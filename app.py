import io
import os
import time
import openai
import tiktoken

import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from utils.blob_storage_handlers import *
from utils.prompts import *
from utils.CSV_formatter import format_anamnezis_csv, format_gyogyszer_csv
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

def set_config():
    st.set_page_config(layout="wide")

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

def table_string_generator(docs, generator, input_tokens) -> str:
    if len(input_tokens) + 3000 <= MODEL_INPUT_TOKEN_SUMM_LIMIT:
        print('include all documents')
        results = [doc.metadata['source'].split("\\")[-1] + "-page-" + str(doc.metadata['page'] )+ ": " + doc.page_content.replace("\n", "").replace("\r", "") for doc in docs]
        sources = "\n".join(results)   
    else:
        sources = retrieve_relevant_chunks(generator,st.session_state.db, MODEL)
        # results = [doc.metadata['source'].split("\\")[-1] + "-page-" + str(doc.metadata['page'] )+ ": " + doc.page_content.replace("\n", "").replace("\r", "") for doc in st.session_state.docs]
        # sources = "\n".join(results)  
    messages =[
    {"role": "system", "content" : "You are a helpful assistant helping people answer their questions related to documents."},
    {"role": "user", "content": table_gen_system_message.format(system_prompt = generator, sources=sources)}
    ]
    full_response = generate_response(messages, MODEL, TEMPERATURE, MAX_TOKENS)
    return full_response.replace('; ',';') if len(full_response.split(";")) > 5 else ""

def upload_table(selected_id, generator, document_type, input_tokens):
    timestamp = datetime.datetime.now().strftime( "%Y%m%d%H%M%S")
    gyogyszer_docs = []
    if document_type == "gyogyszer":
        for uploaded_file in st.session_state.files:
            txt_doc_chunks = load_txt(select_blob_file(blob_storage, st.session_state.selected_container,uploaded_file),filename=uploaded_file['name'].split('/')[-1])
            gyogyszer_docs.extend(txt_doc_chunks)
        WHOLE_DOC, gyogyszer_input_tokens = concat_docs_count_tokens(gyogyszer_docs, encoding)
        generated_text = table_string_generator(gyogyszer_docs, generator, gyogyszer_input_tokens)
    else:
        generated_text = table_string_generator(st.session_state.docs, generator, input_tokens)
    if generated_text != "":
        match document_type:
            case "anam":
                upload_to_blob_storage(blob_storage, st.session_state.selected_container,f"{selected_id}/cache/{selected_id}_anamnezis_of_{timestamp}.csv",generated_text)
                st.session_state.anam_row_index = ""
            case "gyogyszer":
                # st.info(generated_text)
                upload_to_blob_storage(blob_storage, st.session_state.selected_container,f"{selected_id}/cache/{selected_id}_gyogyszererzekenyseg_{timestamp}.csv",generated_text)
                st.session_state.gyogyszer_row_index = ""
        st.session_state.files = [file for file in list_files_in_container(blob_storage, st.session_state.selected_container) if len(file['name'].split('/')) > 2 and selected_id in file['name'].split('/')[-1]]
        return True
    else:
        st.write("Nem található releváns információ")
        return False


def talk_to_your_docs():
    if "selected_id" not in st.session_state:
        st.session_state.selected_id = ""

    if "selected_container" not in st.session_state:
        st.session_state.selected_container = ""
    
    if "docs" not in st.session_state:
        st.session_state.docs = []
    
    if "db" not in st.session_state:
        st.session_state.db = None

    if "files" not in st.session_state:
        st.session_state.files = []

    if "id_list" not in st.session_state:
        st.session_state.id_list = []

    # - - - - - - - - - - - - - - - -
    # Select container and id
    # - - - - - - - - - - - - - - - -

    # start = time.time()
    selected_container = st.selectbox("Válassza ki a használni kívánt állományt:", st.session_state.container_list)
    if selected_container != st.session_state.selected_container:
        ids = set([file['name'].split('/')[0] for file in list_files_in_container(blob_storage, selected_container)])
        st.session_state.id_list = sorted(ids)
    selected_id = st.selectbox("Válassza ki az azonosítót:", st.session_state.id_list)
    # st.info(f"selected container + id deltatime:{time.time()- start:.2f} sec")
    # start = time.time()
    if selected_container != st.session_state.selected_container or selected_id != st.session_state.selected_id:
        time.sleep(1.5)
        #### clear cache ####
        is_authenticated = st.session_state.authenticated
        selected_id_list = st.session_state.id_list
        container_list = st.session_state.container_list

        st.cache_data.clear()
        for key in st.session_state.keys():
            if key not in ['authenticated','password_correct']:
                del st.session_state[key]

        st.session_state.authenticated = is_authenticated
        st.session_state.selected_id = selected_id
        st.session_state.selected_container = selected_container
        st.session_state.id_list = selected_id_list
        st.session_state.container_list = container_list

        #### UPLOAD DOCS #####
        docs = []
        #first filtering, current ID filter
        files = [file for file in list_files_in_container(blob_storage, selected_container) if len(file['name'].split('/')) > 2 and selected_id in file['name'].split('/')[-1]]
        #second filtering, chunking sources (txt-s)
        selected_files = [file for file in files if file['name'].split('/')[2] == "filtered" and f"{selected_id}_" in file['name'].split('/')[-1]]
            
        if selected_files:

            for uploaded_file in selected_files:
                txt_doc_chunks = load_txt(select_blob_file(blob_storage,selected_container,uploaded_file),filename=uploaded_file['name'].split('/')[-1])
                docs.extend(txt_doc_chunks)
            #### STORE DOCS IN VECTOR DATABASE
            embeddings, st.session_state.db = create_db(docs)
        st.session_state.docs = docs
        st.session_state.files = [file for file in list_files_in_container(blob_storage, selected_container) if len(file['name'].split('/')) > 2 and selected_id in file['name'].split('/')[-1]]

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

    if "anam_html_table_name" not in st.session_state:
        st.session_state.anam_html_table_name = ""

    if "gyogyszer_html_table_name" not in st.session_state:
        st.session_state.gyogyszer_html_table_name = ""

    if "chat_html_table_name" not in st.session_state:
        st.session_state.chat_html_table_name = ""

    if "anam_row_index" not in st.session_state:
        st.session_state.anam_row_index = ""

    if "gyogyszer_row_index" not in st.session_state:
        st.session_state.gyogyszer_row_index = ""

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
        upload_table(selected_id, anam_gen_system_prompt, 'anam', input_tokens)
    csv_file = [file for file in st.session_state.files if file['name'].split('/')[1] == 'cache' and 'anamnezis_of' in file['name']]
    if len(csv_file) == 0:
        upload_table(selected_id, anam_gen_system_prompt, 'anam', input_tokens)
        csv_file = [file for file in st.session_state.files if file['name'].split('/')[1] == 'cache' and 'anamnezis_of' in file['name']]
    if len(csv_file) > 0:
        csv_doc =pd.read_csv(io.StringIO(select_blob_file(blob_storage,selected_container,csv_file[-1])), sep=';')
        formatted_csv = format_anamnezis_csv(csv_doc)
        column_names = [col for col in formatted_csv.columns]
        cols = st.columns((1, 4, 2, 3, 2, 4, 3))
        for idx in range(1, len(cols)):
            cols[idx].caption(column_names[idx-1])
        for index, row in formatted_csv.iterrows():
            col1, col2, col3, col4, col5, col6, col7 = st.columns((1, 4, 2, 3, 2, 4, 3))
            col1.write(index + 1)
            col2.write(row[column_names[0]])
            col3.write(row[column_names[1]])
            col4.write(row[column_names[2]])
            col5.write(row[column_names[3]])
            col6.write(row[column_names[4]])
            do_action = col7.button("forrás" if len(row[column_names[5]].split('_')) > 1 and len(row[column_names[5]].split('_')) > 1 else "", key=f"diagnosis_btn_{index}", type="primary")
            if do_action:
                if index + 1 != st.session_state.anam_row_index:
                    st.session_state.anam_row_index = index + 1
                if row[column_names[5]] != st.session_state.anam_html_table_name:
                    st.session_state.anam_html_table_name = row[column_names[5]]
        
        #### feedback and source display ####
        block_feedback(blob_storage, formatted_csv, st.session_state, "anam")
        document_displayer(blob_storage, st.session_state, "anam")

    # st.info(f"anam table display deltatime:{time.time()- start:.2f} sec")
    # start = time.time()
    # - - - - - - - - - - - - - - - -
    # Initializing gyogyszererzekenyseg table
    # - - - - - - - - - - - - - - - -

    st.subheader("Gyógyszerérzékenység szekció")
    gen_success = True
    if st.button("Tábla újragenerálása", key = "gyogyszer_table_gen_btn"):
        gen_success = upload_table(selected_id, gyogyszer_gen_system_prompt, 'gyogyszer', input_tokens)

    csv_file = [file for file in st.session_state.files if file['name'].split('/')[1] == 'cache' and 'gyogyszererzekenyseg' in file['name']]
    if len(csv_file) == 0:
        upload_table(selected_id, gyogyszer_gen_system_prompt, 'gyogyszer', input_tokens)
        csv_file = [file for file in st.session_state.files if file['name'].split('/')[1] == 'cache' and 'gyogyszererzekenyseg' in file['name']]
    if len(csv_file) > 0 and gen_success:
        csv_doc =pd.read_csv(io.StringIO(select_blob_file(blob_storage,selected_container,csv_file[-1])), sep=';')
        formatted_csv = format_gyogyszer_csv(csv_doc)
        if len(formatted_csv) > 0:
            column_names = [col for col in formatted_csv.columns]
            cols = st.columns((1, 2, 2, 2, 2))
            for idx in range(1, len(cols)):
                cols[idx].caption(column_names[idx-1])
            for index, row in formatted_csv.iterrows():
                col1, col2, col3, col4, col5 = st.columns((1, 2, 2, 2, 2))
                col1.write(index + 1)
                col2.write(row[column_names[0]])
                col3.write(row[column_names[1]])
                col4.write(row[column_names[2]])

                do_action = col5.button("forrás" if str(row[column_names[3]]) != 'nan' and len(row[column_names[3]].split('_')) > 1  else "", key=f"gyogyszer_btn_{index}", type="primary")
                if do_action:
                    if index + 1 != st.session_state.gyogyszer_row_index:
                        st.session_state.gyogyszer_row_index = index + 1
                    if str(row[column_names[3]]) != 'nan' and row[column_names[3]] != st.session_state.gyogyszer_html_table_name:
                        st.session_state.gyogyszer_html_table_name = row[column_names[3]]

            #### feedback and source display ####
            block_feedback(blob_storage, formatted_csv, st.session_state, "gyogyszer")
            document_displayer(blob_storage, st.session_state, "gyogyszer")
        else:
            st.write("Nem található releváns adat")

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
        
    document_displayer(blob_storage, st.session_state, "chat")
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
        st.session_state.container_list = [container['name'] for container in blob_storage.list_containers()]



def app_main():
    set_config()
    if "container_list" not in st.session_state:
        st.session_state.container_list = [container['name'] for container in blob_storage.list_containers()]

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated and not check_password():
        st.stop()
    else:
        st.session_state.authenticated = True

    page_names_to_funcs = {
    "Talk to your document":talk_to_your_docs,
    "Dokumentum feltöltés": upload_file
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