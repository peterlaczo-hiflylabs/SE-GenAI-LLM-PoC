import io
import os
import time
import openai
import tiktoken

import pandas as pd
import streamlit as st

from copy import deepcopy
from dotenv import load_dotenv
from utils.blob_storage_handlers import *
from utils.prompts import *
from utils.CSV_formatter import format_diagnosis_csv
from utils.streamlit_functions import *
from utils.table_transform import format_table
from io import StringIO

from utilities import (
    extract_text_between_brackets,
    load_txt,
    create_db,
    concat_docs_count_tokens,
    swap_elements
)

load_dotenv()

# Define default variables
MODEL_INPUT_TOKEN_SUMM_LIMIT = 125000
MODEL_MAX_TOKEN_LIMIT = 128000
MAX_TOKENS = MODEL_MAX_TOKEN_LIMIT-MODEL_INPUT_TOKEN_SUMM_LIMIT
MAX_CONTEXT_QUESTIONS = 120
TEMPERATURE = 0


encoding = tiktoken.get_encoding("cl100k_base")

# TODO: change embedder to azure oai
embedder = 'text-embedding-ada-002'
MODEL = 'gpt-4-1106-preview'

account_name = str(os.environ['azure_name'])
key = str(os.environ['azure_key'])
blob_storage = connect_to_storage(account_name, key)

openai_api = str(os.environ['openai_api_key'])
openai.api_key = openai_api
os.environ["OPENAI_API_KEY"] = openai_api


def generate_embeddings(text):
    response = openai.Embedding.create(input=text, model = embedder)
    embeddings = response['data'][0]['embedding']
    return embeddings

def generate_response(messages, MODEL, TEMPERATURE, MAX_TOKENS):
    completion = openai.ChatCompletion.create(
        model=MODEL, 
        messages=messages, 
        temperature=TEMPERATURE, 
        max_tokens=MAX_TOKENS)
    return completion.choices[0]['message']['content']


def retrieve_relevant_chunks(user_input, db, model):

    query_embedded = generate_embeddings(user_input)

    sim_docs = db.max_marginal_relevance_search_by_vector(query_embedded, k = 3)
    results = [doc.metadata['source'].split("\\")[-1] + "-page-" + str(doc.metadata['page'] )+ ": " + doc.page_content.replace("\n", "").replace("\r", "") for doc in sim_docs]
    sources = "\n".join(results)

    return sources

def table_string_generator(docs, generator) -> str:
    sources = retrieve_relevant_chunks(generator,st.session_state.db, MODEL)
    messages =[
    {"role": "system", "content" : "You are a helpful assistant helping people answer their questions related to documents."},
    {"role": "user", "content": table_gen_system_message.format(system_prompt = generator, sources=sources)}
    ]
    full_response = generate_response(messages, MODEL, TEMPERATURE, MAX_TOKENS)
    return full_response.replace('; ',';') if len(full_response.split(";")) > 1 else ""

def upload_table(docs, selected_id, generator, type):
    timestamp = datetime.datetime.now().strftime( "%Y%m%d%H%M%S")
    gyogyszer_docs = []
    if type == "gyogyszer":
        for uploaded_file in st.session_state.files:
            txt_doc_chunks = load_txt(select_blob_file(blob_storage,'patient-documents',uploaded_file),filename=uploaded_file['name'].split('/')[-1])
            gyogyszer_docs.extend(txt_doc_chunks) 
        generated_text = table_string_generator(gyogyszer_docs, generator)
    else:
        generated_text = table_string_generator(docs, generator)
    if generated_text != "":
        match type:
            case "anam":
                upload_to_blob_storage(blob_storage,"patient-documents",f"{selected_id}/cache/{selected_id}_anamnezis_of_{timestamp}.csv",generated_text)
            case "gyogyszer":
                upload_to_blob_storage(blob_storage,"patient-documents",f"{selected_id}/cache/{selected_id}_gyogyszererzekenyseg_{timestamp}.csv",generated_text)
        return True
    else:
        st.write("Sikertelen generálás")
        return False


def talk_to_your_docs():
    if "selected_id" not in st.session_state:
        st.session_state.selected_id = ""
    
    if "docs" not in st.session_state:
        st.session_state.docs = []
    
    if "db" not in st.session_state:
        st.session_state.db = None

    if "files" not in st.session_state:
        st.session_state.files = []

    # - - - - - - - - - - - - - - - -
    # Select id
    # - - - - - - - - - - - - - - - -

    # start = time.time()
    ids = sorted(set([file['name'].split('/')[0] for file in list_files_in_container(blob_storage, "patient-documents")]))
    selected_id = st.selectbox("Válaszd ki az azonosítót:", ids)
    # st.info(f"selected id deltatime:{time.time()- start:.2f} sec")
    # start = time.time()
    if selected_id != st.session_state.selected_id:
        is_authenticated = st.session_state.authenticated
        st.cache_data.clear()
        for key in st.session_state.keys():
            del st.session_state[key]
        st.session_state.authenticated = is_authenticated
        st.session_state.selected_id = selected_id
        #### UPLOAD DOCS #####
        docs = []
        #first filtering, current ID filter
        files = [file for file in list_files_in_container(blob_storage, "patient-documents") if len(file['name'].split('/')) > 2 and selected_id in file['name'].split('/')[-1]]
        #second filtering, chunking sources (txt-s)
        selected_files = [file for file in files if file['name'].split('/')[2] == "filtered" and f"{selected_id}_" in file['name'].split('/')[-1]]
            
        if selected_files:

            for uploaded_file in selected_files:
                txt_doc_chunks = load_txt(select_blob_file(blob_storage,'patient-documents',uploaded_file),filename=uploaded_file['name'].split('/')[-1])
                docs.extend(txt_doc_chunks)
            #### STORE DOCS IN VECTOR DATABASE
            embeddings, st.session_state.db = create_db(docs)
        st.session_state.docs = docs
    st.session_state.files = [file for file in list_files_in_container(blob_storage, "patient-documents") if len(file['name'].split('/')) > 2 and selected_id in file['name'].split('/')[-1]]
    
    # st.info(st.session_state.files)

    # st.info(f"generating database deltatime:{time.time()- start:.2f} sec")
    # start = time.time()

    # - - - - - - - - - - - - - - - -
    # Define Session state elements
    # - - - - - - - - - - - - - - - -

    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = MODEL

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "source_links" not in st.session_state:
        st.session_state.source_links = None

    if "anamnezis_html_table_name" not in st.session_state:
        st.session_state.anamnezis_html_table_name = ""

    if "gyogyszer_html_table_name" not in st.session_state:
        st.session_state.gyogyszer_html_table_name = ""

    if "chat_html_table_name" not in st.session_state:
        st.session_state.chat_html_table_name = ""

    if "anam_row_index" not in st.session_state:
        st.session_state.anam_row_index = ""

    if "gyogyszer_row_index" not in st.session_state:
        st.session_state.gyogyszer_row_index = ""


    #### Clear cache ####
    # if "previous_id" not in st.session_state:
    #     st.session_state.previous_id = selected_id

    # if selected_id != st.session_state.previous_id:
    #     st.session_state.previous_id = selected_id
    #     is_authenticated = st.session_state.authenticated
    #     st.cache_data.clear()
    #     for key in st.session_state.keys():
    #         del st.session_state[key]
    #     st.session_state.authenticated = is_authenticated

    # - - - - - - - - - - - - - - - -
    # Start frontend
    # - - - - - - - - - - - - - - - -

    st.title("Semmelweis X Hiflylabs")
    st.header("Semmelweis GenAI/LLM Anamnézis PoC")
    st.write("Készítette: Hiflylabs")

    WHOLE_DOC, input_tokens = concat_docs_count_tokens(st.session_state.docs, encoding)
    st.write('A paciens dokumentumainak tokenszáma: ' + str(len(input_tokens)))

    # st.info(f"state initializing deltatime:{time.time()- start:.2f} sec")
    # start = time.time()
    
    # - - - - - - - - - - - - - - - -
    # initializing anamnezis table
    # - - - - - - - - - - - - - - - -

    st.subheader("Anamnézis szekció")
    if st.button("Tábla újragenerálása", key = "anam_table_gen_btn"):
        upload_table(st.session_state.docs, selected_id, anam_gen_system_prompt, 'anam')
    # st.info(len(st.session_state.files))
    csv_file = [file for file in st.session_state.files if file['name'].split('/')[1] == 'cache' and 'anamnezis_of' in file['name']]
    if len(csv_file) == 0:
        upload_table(st.session_state.docs, selected_id, anam_gen_system_prompt, 'anam')
        csv_file = [file for file in st.session_state.files if file['name'].split('/')[1] == 'cache' and 'anamnezis_of' in file['name']]
        # st.info([file['name'] for file in st.session_state.files])
        # st.info(f"{len(csv_file)} {csv_file}")
    csv_doc =pd.read_csv(io.StringIO(select_blob_file(blob_storage,'patient-documents',csv_file[-1])), sep=';')
    formatted_csv = format_diagnosis_csv(csv_doc)
    column_names = swap_elements([col for col in formatted_csv.columns],3,4)
    cols = st.columns((1, 4, 3, 2, 4, 4))
    for idx in range(1, len(cols)):
        cols[idx].caption(column_names[idx-1])
    for index, row in formatted_csv.iterrows():
        col1, col2, col3, col4, col5, col6 = st.columns((1, 4, 3, 2, 4, 4))
        col1.write(index + 1)
        col2.write(row['Diagnózis'])
        col3.write(row['Kezdete'])
        col4.write(row['BNO-10'])
        col5.write(row['BNO leírás'])
        do_action = col6.button("forrás dokumentum", key=f"diagnosis_btn_{index}", type="primary")
        if do_action:
            if index + 1 != st.session_state.anam_row_index:
                st.session_state.anam_row_index = index + 1
            if row[column_names[-1]] != st.session_state.anamnezis_html_table_name:
                st.session_state.anamnezis_html_table_name = row[column_names[-1]]
    
    #### feedback and source display ####
    block_feedback(blob_storage, st.session_state.files, selected_id, csv_file[-1], "anamnezis", st.session_state.anam_row_index)
    document_displayer(blob_storage, st.session_state.files, st.session_state.anamnezis_html_table_name, st.session_state.anam_row_index)

    # st.info(f"anam table display deltatime:{time.time()- start:.2f} sec")
    # start = time.time()
    # - - - - - - - - - - - - - - - -
    # Initializing gyogyszererzekenyseg table
    # - - - - - - - - - - - - - - - -

    st.subheader("Gyógyszerérzékenység szekció")
    show_table = True
    if st.button("Tábla újragenerálása", key = "gyogyszer_table_gen_btn"):
        show_table = upload_table(st.session_state.docs, selected_id, gyogyszer_gen_system_prompt, 'gyogyszer')

    csv_file = [file for file in st.session_state.files if file['name'].split('/')[1] == 'cache' and 'gyogyszererzekenyseg' in file['name']]
    if len(csv_file) == 0:
        show_table = upload_table(st.session_state.docs, selected_id, gyogyszer_gen_system_prompt, 'gyogyszer')
        csv_file = [file for file in st.session_state.files if file['name'].split('/')[1] == 'cache' and 'gyogyszererzekenyseg' in file['name']]
    if show_table:
        csv_doc =pd.read_csv(io.StringIO(select_blob_file(blob_storage,'patient-documents',csv_file[-1])), sep=';')
        column_names = [col for col in csv_doc.columns]
        cols = st.columns((1, 2, 2, 2))
        for idx in range(1, len(cols)):
            cols[idx].caption(column_names[idx-1])
        for index, row in csv_doc.iterrows():
            col1, col2, col3, col4 = st.columns((1, 2, 2, 2))
            col1.write(index + 1)
            col2.write(row[column_names[0]])
            col3.write(row[column_names[1]])
            do_action = col4.button("forrás dokumentum" if str(row[column_names[2]]) != 'nan' else "", key=f"gyogyszer_btn_{index}", type="primary")
            if do_action:
                if index + 1 != st.session_state.gyogyszer_row_index:
                    st.session_state.gyogyszer_row_index = index + 1
                if str(row[column_names[2]]) != 'nan' and row[column_names[2]] != st.session_state.gyogyszer_html_table_name:
                    st.session_state.gyogyszer_html_table_name = row[column_names[1]]
                else:
                    st.session_state.gyogyszer_html_table_name = ""

        #### feedback and source display ####
        block_feedback(blob_storage, st.session_state.files, selected_id, csv_file[-1],"gyogyszer", st.session_state.gyogyszer_row_index)
        document_displayer(blob_storage, st.session_state.files, st.session_state.gyogyszer_html_table_name, st.session_state.gyogyszer_row_index)

    # st.info(f"gyogyszer table display deltatime:{time.time()- start:.2f} sec")
    # start = time.time()

    # - - - - - - - - - - - - - - -
    # Chat main part
    # - - - - - - - - - - - - - - -

    st.subheader("Chat szekció")

    msg = st.chat_message('assistant')
    msg.write("Üdvözlöm! 👋 Tegyen fel kérdéseket a kiválasztott pácienssel kapcsolatban!")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if QUERY := st.chat_input("Ide írja a kérdését"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(QUERY)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            if len(input_tokens) <= MODEL_INPUT_TOKEN_SUMM_LIMIT:
                print('include all documents')
                results = [doc.metadata['source'].split("\\")[-1] + "-page-" + str(doc.metadata['page'] )+ ": " + doc.page_content.replace("\n", "").replace("\r", "") for doc in docs]
                sources = "\n".join(results)   
            else:
                sources = retrieve_relevant_chunks(QUERY, st.session_state.db, MODEL)


            messages =[
                        {"role": "system", "content" : "You are a helpful assistant helping people answer their questions related to documents."},
                        {"role": "user", "content": system_message.format(system_prompt = default_system_prompt, sources=sources)},
                        *st.session_state.messages,
                        {"role": "user", "content": question_message.format(question=QUERY)}
                        ]
            
            current_token_count = len(encoding.encode(' '.join([i['content'] for i in messages])))

            while (len(messages)-3 > MAX_CONTEXT_QUESTIONS * 2) or (current_token_count >= MODEL_INPUT_TOKEN_SUMM_LIMIT):

                messages.pop(3)            
                current_token_count = len(encoding.encode(' '.join([i['content'] for i in messages])))

            full_response = generate_response(messages, MODEL, TEMPERATURE, MAX_TOKENS)
            message_placeholder.markdown(full_response)

        # Add user and AI message to chat history
        st.session_state.messages.append({"role": "user", "content": QUERY})
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.chat_html_table_name = ""
    
    #### Chat source display part ####
    if len(st.session_state.messages) > 0:
        source_links = extract_text_between_brackets(st.session_state.messages[-1]['content'])
        if st.session_state.source_links != source_links:
            st.session_state.source_links = None
        sources_expander = st.expander(label='Forrás')
        with sources_expander:
            if len(input_tokens) <= MODEL_INPUT_TOKEN_SUMM_LIMIT:
                #st.write('A válasz generálásához az összes feltöltött dokumentum felhasználásra került.')
                for element_id in range(len(source_links)):
                    if st.button(source_links[element_id],key=f"expander_btn_{element_id}", type="primary"):
                        if source_links[element_id].split('-p')[0] != st.session_state.chat_html_table_name:
                            st.session_state.chat_html_table_name = source_links[element_id].split('-p')[0]
            else:
                st.write("A válasz generálásához az alábbi, relevánsnak ítélt dokumentumok lettek felhasználva:")
                st.text(sources)
        
    document_displayer(blob_storage, st.session_state.files, st.session_state.chat_html_table_name, "")
    # st.info(f"chat deltatime:{time.time()- start:.2f} sec")
    # start = time.time()

def upload_file():
    container_name = st.text_input("Írja be az állomány nevét",key="input_container_name")
    st.markdown("""
        Kérem itt töltse fel az adatokat tartalmazó excel táblát, melyet használni akar.
         A feltöltés csak az előre meghatározott paraméterekkel rendelkező táblák esetében működik.
                
        Header: "NPI|CASE_NO|ADMIT_DATE|CASE_TYPE|DEPT|DESCR|TX_TYPE|VER_NO|SEQ_NO|Ananmézis|Jelen panaszok|Dekurzus|Epikrízis|Egyéb vizsgálatok|Műtéti leírás|Státusz|Javaslat|Therápia"
    """)
    uploaded_file = st.file_uploader("Töltse fel a fájlokat! Elfogadott formátumok: xlsx", 
                     type = 'xlsx', accept_multiple_files=False)
    if st.button("Feltöltés", key="submit_xlsx_btn"):
        df_raw = pd.read_excel(uploaded_file, engine='openpyxl', dtype={'NPI':'object', 'CASE_NO':'object'}).convert_dtypes()
        blob_storage.create_container(container_name)
        format_table(df_raw, blob_storage, container_name)


def app_main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated and not check_password():
        st.stop()
    else:
        st.session_state.authenticated = True

    page_names_to_funcs = {
    "Talk to your document":talk_to_your_docs
    # "Dokumentum feltöltés": upload_file
    }

    st.sidebar.image("img/semmelweis_logo_transparent.png", use_column_width=True)
    window_name = st.sidebar.selectbox("Válaszd ki a használandó funkciót", page_names_to_funcs.keys())
    st.sidebar.title("Leírás")
    st.sidebar.markdown(
        """
        Lépések\n

        1. Talk to your documents

            1.1 Személy kiválasztása

            1.2 Ha a táblázat nem biztosít elég anyagot,
            akkor a chat segítségével lehet további
            adatokat kinyerni a rendszerből.
        2. Dokumentum feltöltés

            2.1 Dokumentum helyes formában való feltöltése
        """
    )
    format_button_style()
    page_names_to_funcs[window_name]()

if __name__ == "__main__":
    app_main()