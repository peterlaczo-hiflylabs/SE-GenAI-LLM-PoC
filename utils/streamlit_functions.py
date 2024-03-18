import datetime
import hmac
import io
import os
import pandas as pd
import streamlit as st
from utils.blob_storage_handlers import *
from utils.prompts import *
from utils.streamlit_functions import *
from utilities import text_to_html


def format_button_style():

    st.markdown(
    """
    <style>
    button[kind="primary"] {
        background: none!important;
        border: none;
        padding: 0!important;
        color: black !important;
        text-decoration: none;
        cursor: pointer;
        border: none !important;
    }
    button[kind="primary"]:hover {
        text-decoration: none;
        color: black !important;
    }
    button[kind="primary"]:focus {
        outline: none !important;
        box-shadow: none !important;
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                width: 200px !important; # Set the width to your desired value
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
def document_displayer(blob_storage, files, session_state_data: str, row_num):
    if session_state_data != None and session_state_data != "":
        with st.expander("Kiválasztott dokumentum"):
            for element in files:
                if element['name'].split('/')[-1] in session_state_data:
                    if row_num != "":
                        st.write(session_state_data.replace('[', f"{row_num}. sor; ").replace(']',':'))
                    else:
                        st.write(session_state_data.strip('[').replace(']',':'))
                    html_document = text_to_html((select_blob_file(blob_storage,'patient-documents',element)), element['name'])
                    st.markdown(html_document, unsafe_allow_html=True)
                    break

def password_entered():
    """Checks whether a password entered by the user is correct."""
    if hmac.compare_digest(st.session_state["password"], os.environ["streamlit_password"]):
        st.session_state["password_correct"] = True
        del st.session_state["password"]  # Don't store the password.
    else:
        st.session_state["password_correct"] = False

def check_password():
    """Returns `True` if the user had the correct password."""

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("Password incorrect")
    return False

def block_feedback(blob_storage, files, selected_id, table, type, row_num):
    timestamp = table['name'].split('/')[-1].split('_')[-1].split('.')[0]
    try:
        feedback_storage_source = [file for file in files if file['name'].split('/')[1] == 'cache' and f"{type}_feedbacks" in file['name'] and timestamp in file['name']][0]
        feedback_storage = pd.read_csv(io.StringIO(select_blob_file(blob_storage,'patient-documents',feedback_storage_source)), sep=';')
    except:
        feedback_storage = pd.DataFrame()
    if st.checkbox("Visszajelzés adása", key=f"{type}_curr", value=False):
        feedback_name = st.text_input("Kérlek írd be a neved:", value="", key=f"{type}_name")
        cols =  st.columns((1.1,1.5,1.1,1,1))
        check_box_answers = [0] * 5
        check_box_answers[0] = cols[0].checkbox("Általános megjegyzés?")
        check_box_answers[1] = cols[1].checkbox("Helyesen szerepel az anamnézisben?")
        check_box_answers[2] = cols[2].checkbox("Duplikátum")
        check_box_answers[3] = cols[3].checkbox("Helyes dátum?")
        check_box_answers[4] = cols[4].checkbox("Pontos BNO kód?")
        feedback_text = st.text_area("Kérlek írd be a visszajelzésed", value="",key=f"{type}_desc")
        if st.button("Beküldés", key=f"{type}_submitbtn"):
            feedback_storage = pd.concat([feedback_storage,pd.DataFrame([{
                'név': feedback_name,
                'dátum': datetime.datetime.now().strftime("%Y/%m/%d"),
                'időpont': datetime.datetime.now().strftime("%H:%M:%S"),
                'sor': "-" if check_box_answers[0] else row_num,
                'helyes anamnézis': check_box_answers[1],
                'duplikátum': check_box_answers[2],
                'helyes dátum': check_box_answers[3],
                'helyes BNO': check_box_answers[4],
                'leírás': feedback_text
            }], index=[0])],ignore_index=True)
            csv_string_buffer = io.StringIO()
            feedback_storage.to_csv(csv_string_buffer, index=False, sep=';')
            csv_string = csv_string_buffer.getvalue()
            success, error = upload_to_blob_storage(blob_storage,"patient-documents",f"{selected_id}/cache/{selected_id}_{timestamp}_{type}_feedbacks.csv",csv_string)
            if success:
                st.write("FILE UPLOADED")
                feedback_name = ""
                feedback_text = ""
            else:
                st.write(error)
    if st.checkbox("Korábbi visszajelzések megjelenítése",key=f"{type}_prev", value=False):
        if len(feedback_storage)>0:
            st.table(feedback_storage)
        else:
            st.write("Nem található korábbi visszajelzés")
