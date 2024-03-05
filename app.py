# packages
import io
import re
import os
import openai
import streamlit as st
import numpy as np
import pandas as pd

import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")

from copy import deepcopy
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from azure.storage.blob import BlobServiceClient
from utils import (
    create_db,
    concat_docs_count_tokens,
    add_context_to_doc_chunks
)

load_dotenv()
st.sidebar.image("img/semmelweis_logo_transparent.png", use_column_width=True)
with st.sidebar:
    openai_api = str(os.getenv('openai_api_key'))#st.text_input('OpenAI API kulcs', type = 'password', key = 'openai_key')
    openai.api_key = openai_api
    os.environ["OPENAI_API_KEY"] = openai_api


# openai models, settings
embedder = 'text-embedding-ada-002'

MODEL_INPUT_TOKEN_SUMM_LIMIT = 125000
MODEL_MAX_TOKEN_LIMIT = 128000
MAX_CONTEXT_QUESTIONS = 120

#-------------------------------------------------------------------------------------------------------------------
# functions, prompts
def connect_to_storage(account_name, key):
    blob_service_client = BlobServiceClient.from_connection_string(f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={key};EndpointSuffix=core.windows.net")
    return blob_service_client

def list_files_in_container(blob_service_client, container_name):
    client = blob_service_client.get_container_client(container_name)
    blob_list = client.list_blobs()
    return [blob for blob in blob_list]

def select_blob_file(blob_service_client, container_name, blob):
    client = blob_service_client.get_container_client(container_name)
    blob_file = client.get_blob_client(blob)
    return (blob_file.download_blob()).readall().decode("utf-8")

def text_to_chunk(text):
    DOCS = []
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text_splitted = RecursiveCharacterTextSplitter(chunk_size = 100000, chunk_overlap = 200).split_text(text)
    docs = [Document(page_content=t, metadata={'source' : filename, 'page' : 'all'}) for t in text_splitted]
    docs = add_context_to_doc_chunks(docs)
    DOCS.append(docs)

    DOCS = [item for sublist in DOCS for item in sublist]

    return DOCS

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

#-------------------------------------------------------------------------------------------------------------------

default_system_prompt = """Act as an assistant that helps people with their questions relating to a wide variety of documents. 
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question. 
Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brakets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
If you did not use the information below to answer the question, do not include the source name or any square brackets."""

system_message = """{system_prompt}

Sources:
{sources}

"""

#question_message = """
#Question: {question}
#
#Answer: 
#"""

question_message = """
{question}

Assistant: 
"""


#-------------------------------------------------------------------------------------------------------------------
account_name = str(os.getenv('azure_name'))
key = str(os.getenv('azure_key'))
blob_storage = connect_to_storage(account_name, key)


# streamlit app
st.title("Semmelweis X Hiflylabs")
st.header("Semmelweis GenAI/LLM Anamnézis PoC")
st.write("Készítette: Hiflylabs")
#st.sidebar.image("https://hiflylabs.com/_next/static/media/greenOnDark.35e68199.svg", use_column_width=True)

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
#model parameters
model_name = 'gpt-4-1106-preview'
MODEL = model_name
SYSTEM_MESSAGE = """Act as an assistant who helps people with their questions relating to patient documents. 
Your answer must be based on the facts listed in the sources below, but you can augment the given facts with extra knowledge.
Each source has a name followed by a colon and the actual information, always include the source name for each fact you use in the response. Use square brackets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
If you did not use a piece of information below to answer the question, do not include its source name or any square brackets."""
TEMPERATURE = 0
MAX_TOKENS = MODEL_MAX_TOKEN_LIMIT-MODEL_INPUT_TOKEN_SUMM_LIMIT

ids = set([file['name'].split('/')[0] for file in list_files_in_container(blob_storage, "patient-documents")])
selected_id = st.selectbox("Válaszd ki az azonosítót:", ids)

#### UPLOAD DOCS #####
docs = []
#first filtering, current ID filter
files = [file for file in list_files_in_container(blob_storage, "patient-documents") if file['name'].split('/')[0] == selected_id and len(file['name'].split('/')) > 2]
#second filtering, chunking sources (txt-s)
selected_files = [file for file in files if file['name'].split('/')[2] == "filtered" and f"{selected_id}_" in file['name'].split('/')[-1]]
    
if selected_files:

    if not openai_api:
        st.warning('🔑🔒 A folytatáshoz adja meg az OpenAI API kulcsot az oldalsó panelen 🔑🔒')

    else:
    
        for uploaded_file in selected_files:

            filename = uploaded_file['name'].split('/')[-1]
            
            txt_doc_chunks = text_to_chunk(select_blob_file(blob_storage,'patient-documents',uploaded_file))
            docs.extend(txt_doc_chunks)

        docs_original = deepcopy(docs)

        #### STORE DOCS IN VECTOR DATABASE
        embeddings, db = create_db(docs)

#### END OF UPLOAD PART ####

#### Clear cache ####
if "previous_id" not in st.session_state:
    st.session_state.previous_id = selected_id

if selected_id != st.session_state.previous_id:
    st.session_state.previous_id = selected_id
    st.cache_data.clear()
    for key in st.session_state.keys():
        del st.session_state[key]


#### end of clear cache
    
WHOLE_DOC, input_tokens = concat_docs_count_tokens(docs, encoding)
st.write('Bemeneti tokenszám: ' + str(len(input_tokens)))
# st.write('💰 Approx. cost of processing, not including completion:', str(round(MODEL_COST[MODEL] * (len(input_tokens) + 500) / 1000, 5)), 'USD')

#showing CSV
csv_file = [file for file in files if file['name'].split('/')[1] == 'cache']
if len(csv_file) != 0:
    st.write(pd.read_csv(io.StringIO(select_blob_file(blob_storage,'patient-documents',csv_file[0])), sep=';',))

msg = st.chat_message('assistant')
msg.write("Üdvözlöm! 👋 Tegyen fel kérdéseket a kiválasztott személlyel kapcsolatban!")

### chat elements integration

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = MODEL

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

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

        if len(input_tokens) <= MODEL_INPUT_TOKEN_SUMM_LIMIT: # maybe we can fit everything into the prompt, why not
            print('include all documents')
            results = [doc.metadata['source'].split("\\")[-1] + "-page-" + str(doc.metadata['page'] )+ ": " + doc.page_content.replace("\n", "").replace("\r", "") for doc in docs]
            sources = "\n".join(results)   
        else:
            sources = retrieve_relevant_chunks(QUERY, db, MODEL)


        messages =[
                    {"role": "system", "content" : "You are a helpful assistant helping people answer their questions related to documents."},
                    {"role": "user", "content": system_message.format(system_prompt = SYSTEM_MESSAGE, sources=sources)},
                    *st.session_state.messages,
                    {"role": "user", "content": question_message.format(question=QUERY)}
                    ]
        
        # to always fit in context, either limit historic messages, or count tokens
        # current solution: if we reach model-specific max msg number or token count, remove q-a pairs from beginning until conditions are met
        
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

        sources_expander = st.expander(label='Forrás')
        with sources_expander:
            #st.write('\n')
            if len(input_tokens) <= MODEL_INPUT_TOKEN_SUMM_LIMIT:
                st.write('A válasz generálásához az összes feltöltött dokumentum felhasználásra került.')
            else:
                st.write("A válasz generálásához az alábbi, relevánsnak ítélt dokumentumok lettek felhasználva:")
                st.text(sources)
