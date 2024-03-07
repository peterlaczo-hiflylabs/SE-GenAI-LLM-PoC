import io
#import re
import os
import openai
import tiktoken

import numpy as np
import pandas as pd
import streamlit as st

from copy import deepcopy
from dotenv import load_dotenv
from utils.blob_storage_handlers import *
from utils.prompts import system_message, default_system_prompt, question_message
from utils.CSV_formatter import format_csv

from utilities import (
    extract_text_between_brackets,
    text_to_html,
    load_txt,
    create_db,
    concat_docs_count_tokens
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

account_name = str(os.getenv('azure_name'))
key = str(os.getenv('azure_key'))
blob_storage = connect_to_storage(account_name, key)



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


ids = set([file['name'].split('/')[0] for file in list_files_in_container(blob_storage, "patient-documents")])
selected_id = st.selectbox("Válaszd ki az azonosítót:", ids)
# selected_id = '008359041'
#### UPLOAD DOCS #####
docs = []
#first filtering, current ID filter
files = [file for file in list_files_in_container(blob_storage, "patient-documents") if file['name'].split('/')[0] == selected_id and len(file['name'].split('/')) > 2]
#second filtering, chunking sources (txt-s)
selected_files = [file for file in files if file['name'].split('/')[2] == "filtered" and f"{selected_id}_" in file['name'].split('/')[-1]]
    
if selected_files:

    for uploaded_file in selected_files:

        filename = uploaded_file['name'].split('/')[-1]
        
        txt_doc_chunks = load_txt(select_blob_file(blob_storage,'patient-documents',uploaded_file),filename=uploaded_file['name'].split('/')[-1])
        docs.extend(txt_doc_chunks)

    docs_original = deepcopy(docs)

    #### STORE DOCS IN VECTOR DATABASE
    embeddings, db = create_db(docs)


#### Clear cache ####
if "previous_id" not in st.session_state:
    st.session_state.previous_id = selected_id

if selected_id != st.session_state.previous_id:
    st.session_state.previous_id = selected_id
    st.cache_data.clear()
    for key in st.session_state.keys():
        del st.session_state[key]


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

if "html_table_name" not in st.session_state:
    st.session_state.html_table_name = ""


# - - - - - - - - - - - - - - -
# Chat main part
# - - - - - - - - - - - - - - -
    
st.title("Semmelweis X Hiflylabs")
st.header("Semmelweis GenAI/LLM Anamnézis PoC")
st.write("Készítette: Hiflylabs")

st.sidebar.image("img/semmelweis_logo_transparent.png", use_column_width=True)
with st.sidebar:
    openai_api = str(os.getenv('openai_api_key'))
    openai.api_key = openai_api
    os.environ["OPENAI_API_KEY"] = openai_api

st.sidebar.title("Leírás")
st.sidebar.markdown(
    """
   Lépések\n

    1. Személy kiválasztása

    2. Ha a táblázat nem biztosít elég anyagot,
       akkor a chat segítségével lehet további
       adatokat kinyerni a rendszerből.
    """
)

WHOLE_DOC, input_tokens = concat_docs_count_tokens(docs, encoding)
st.write('A paciens dokumentumainak tokenszáma: ' + str(len(input_tokens)))

#showing CSV
csv_file = [file for file in files if file['name'].split('/')[1] == 'cache']
if len(csv_file) != 0:
    csv_doc =pd.read_csv(io.StringIO(select_blob_file(blob_storage,'patient-documents',csv_file[0])), sep=';')
    formatted_csv = format_csv(csv_doc)
    for index, row in formatted_csv.iterrows():
        col1, col2, col3, col4, col5, col6 = st.columns((1, 3, 2, 1, 3, 3))
        col1.write(index)
        col2.write(row['Diagnózis'])
        col3.write(row['Kezdete'])
        col4.write(row['BNO-10'])
        col5.write(row['BNO leírás'])
        do_action = col6.button(row["Forrás(ok) "], key=index, type="secondary")
        if do_action:
            if row["Forrás(ok) "] != st.session_state.html_table_name:
                st.session_state.html_table_name = row["Forrás(ok) "]

html_name_placeholder = st.empty()
html_placeholder = st.empty()

st.markdown(
    """
    <style>
    button[kind="secondary"] {
        background: none!important;
        border: none;
        padding: 0!important;
        color: black !important;
        text-decoration: none;
        cursor: pointer;
        border: none !important;
    }
    button[kind="secondary"]:hover {
        text-decoration: none;
        color: black !important;
    }
    button[kind="secondary"]:focus {
        outline: none !important;
        box-shadow: none !important;
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

msg = st.chat_message('assistant')
msg.write("Üdvözlöm! 👋 Tegyen fel kérdéseket a kiválasztott pácienssel kapcsolatban!")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Good code under this 
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
            sources = retrieve_relevant_chunks(QUERY, db, MODEL)


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

if len(st.session_state.messages) > 0:
    source_links = extract_text_between_brackets(st.session_state.messages[-1]['content'])
    if st.session_state.source_links != source_links:
        st.session_state.source_links = None
    sources_expander = st.expander(label='Forrás')
    with sources_expander:
        if len(input_tokens) <= MODEL_INPUT_TOKEN_SUMM_LIMIT:
            #st.write('A válasz generálásához az összes feltöltött dokumentum felhasználásra került.')
            for element_id in range(len(source_links)):
                if st.button(source_links[element_id],key=f"expander_btn_{element_id}"):
                    if source_links[element_id].split('-p')[0] != st.session_state.html_table_name:
                        st.session_state.html_table_name = source_links[element_id].split('-p')[0]
        else:
            st.write("A válasz generálásához az alábbi, relevánsnak ítélt dokumentumok lettek felhasználva:")
            st.text(sources)
    
if st.session_state.html_table_name != None:
    for element in selected_files:
        if element['name'].split('/')[-1] in st.session_state.html_table_name:
            html_name_placeholder.write(st.session_state.html_table_name.strip('[').replace(']',':'))
            html_document = text_to_html((select_blob_file(blob_storage,'patient-documents',element)), element['name'])
            html_placeholder.markdown(html_document, unsafe_allow_html=True)
            break