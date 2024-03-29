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
from utils.CSV_formatter import create_extended_log


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
                width: 200px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def document_displayer(blob_storage, session_state, doc_type):
    match doc_type:
        case "anam":
            session_state_data = session_state.anam_html_table_name
            row_num = session_state.anam_row_index
        case "gyogyszer":
            session_state_data = session_state.gyogyszer_html_table_name
            row_num = session_state.gyogyszer_row_index
        case "chat":
            session_state_data = session_state.chat_html_table_name
            row_num = ""
    if session_state_data != None and session_state_data != "":
        with st.expander("Kiválasztott dokumentum"):
            for element in session_state.files:
                if element['name'].split('/')[-1] in session_state_data or session_state_data.strip('[').strip(']') in element['name'].split('/')[-1]:
                    if row_num != "":
                        st.write(session_state_data.replace('[', f"{row_num}. sor; ").replace(']',':'))
                    else:
                        st.write(session_state_data.strip('[').replace(']',':'))
                    html_document = text_to_html((select_blob_file(blob_storage, session_state.selected_container ,element)), element['name'])
                    st.markdown(html_document, unsafe_allow_html=True)
                    break


def password_entered():
    """Checks whether a password entered by the user is correct."""
    if hmac.compare_digest(st.session_state["password"], os.environ["streamlit_password"]):
        st.session_state["password_correct"] = True
        # del st.session_state["password"]  # Don't store the password.
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

def csv_upload_from_dataframe(blob_storage, storage: pd.DataFrame, session_state, timestamp: str, doc_type):
    csv_string_buffer = io.StringIO()
    storage.to_csv(csv_string_buffer, index=False, sep=';')
    csv_string = csv_string_buffer.getvalue()
    success, error = upload_to_blob_storage(blob_storage, session_state.selected_container,f"{session_state.selected_id}/cache/{session_state.selected_id}_{timestamp}_{doc_type}_feedbacks.csv",csv_string)
    if success and "extended" not in doc_type:
        st.write("FILE UPLOADED")
        
def search_for_log(blob_storage, session_state, doc_type, timestamp):
    feedback_storage_source = [file for file in session_state.files if file['name'].split('/')[1] == 'cache' and f"{doc_type}_feedbacks" in file['name'] and timestamp in file['name']]
    if len(feedback_storage_source) > 0:
            feedback_storage = pd.read_csv(io.StringIO(select_blob_file(blob_storage, session_state.selected_container,feedback_storage_source[0])), sep=';')
    else:
        feedback_storage = pd.DataFrame()
    return feedback_storage

def block_feedback(blob_storage, gen_csv, session_state, doc_type):
    match doc_type:
        case "anam":
            row_num = session_state.anam_row_index
        case "gyogyszer":
            row_num = session_state.gyogyszer_row_index

    not_refreshed = True
    if st.checkbox("Visszajelzés adása", key=f"{doc_type}_curr", value=False):
        session_state.files = [file for file in list_files_in_container(blob_storage, session_state.selected_container) if len(file['name'].split('/')) > 2 and session_state.selected_id in file['name'].split('/')[-1]]
        timestamp = [file for file in st.session_state.files if file['name'].split('/')[1] == 'cache' and 'anamnezis_of' in file['name']][-1]['name'].split('/')[-1].split('_')[-1].split('.')[0]
        not_refreshed = False

        feedback_storage = search_for_log(blob_storage,session_state, doc_type, timestamp)
        extended_feedback_storage = search_for_log(blob_storage,session_state,f"{doc_type}_extended",timestamp)
        feedback_name = st.text_input("Kérlek írd be a neved:", value="", key=f"{doc_type}_name")
        cols =  st.columns((1.1,1.5,1.1,1,1))
        check_box_answers = [""] * 5
        if row_num != "":
            check_box_answers[0] = cols[0].checkbox("Általános megjegyzés?", key=f"{doc_type}_alt")
            if check_box_answers[0] == False:
                check_box_answers[1] = cols[1].checkbox("Helytelenül szerepel a táblában?", key=f"{doc_type}_")
                check_box_answers[2] = cols[2].checkbox("Duplikátum?", key=f"{doc_type}_dupl")
                check_box_answers[3] = cols[3].checkbox("Helytelen dátum?", key=f"{doc_type}_date")
                check_box_answers[4] = cols[4].checkbox("Rossz BNO kód?", key=f"{doc_type}_bno")
        feedback_text = st.text_area("Kérlek írd be a visszajelzésed", value="",key=f"{doc_type}_desc")

        if st.button("Beküldés", key=f"{doc_type}_submitbtn"):
            current_row = pd.DataFrame([{
                'név': feedback_name,
                'dátum': datetime.datetime.now().strftime("%Y/%m/%d"),
                'időpont': datetime.datetime.now().strftime("%H:%M:%S"),
                'sor': "-" if check_box_answers[0] else row_num,
                'helytelen tábla': str(check_box_answers[1]) if check_box_answers[1] != "" else '-',
                'duplikátum?': str(check_box_answers[2]) if check_box_answers[1] != "" else '-',
                'helytelen dátum': str(check_box_answers[3]) if check_box_answers[1] != "" else '-',
                'helytelen BNO': str(check_box_answers[4]) if check_box_answers[1] != "" else '-',
                'leírás': feedback_text
            }], index=[row_num-1 if row_num !=  "" else 0])
            feedback_storage = pd.concat([feedback_storage,current_row],ignore_index=True)
            csv_upload_from_dataframe(blob_storage,feedback_storage, session_state, timestamp, doc_type)
            extended_log = create_extended_log(gen_csv, current_row)
            extended_feedback_storage = pd.concat([extended_feedback_storage,extended_log],ignore_index=True)
            csv_upload_from_dataframe(blob_storage, extended_feedback_storage, session_state, timestamp, f"{doc_type}_extended")

    if st.checkbox("Korábbi visszajelzések megjelenítése",key=f"{doc_type}_prev", value=False):
        if  not_refreshed:
            session_state.files = [file for file in list_files_in_container(blob_storage, session_state.selected_container) if len(file['name'].split('/')) > 2 and session_state.selected_id in file['name'].split('/')[-1]]
            timestamp = [file for file in st.session_state.files if file['name'].split('/')[1] == 'cache' and 'anamnezis_of' in file['name']][-1]['name'].split('/')[-1].split('_')[-1].split('.')[0]
        
        feedback_storage = search_for_log(blob_storage,session_state, doc_type, timestamp)
        if len(feedback_storage) > 0:
            st.table(feedback_storage)
        else:
            st.write("Nem található korábbi visszajelzés")