# packages
import streamlit as st
import numpy as np
from copy import deepcopy

#from credentials import openai_api
import os
import openai

st.sidebar.image("img/semmelweis_logo_transparent.png", use_column_width=True)
with st.sidebar:
    openai_api = st.text_input('OpenAI API kulcs', type = 'password', key = 'openai_key')
    openai.api_key = openai_api
    os.environ["OPENAI_API_KEY"] = openai_api


import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")

from utils import (
    load_pdf,
    load_HTML,
    load_docx,
    load_txt,
    create_db,
    concat_docs_count_tokens
)

# openai models, settings
embedder = 'text-embedding-ada-002'

MODEL_RELEVANT_DOC_NUMBER = {'gpt-3.5-turbo' : 3,
                            'gpt-3.5-turbo-16k' : 5,
                            'gpt-4' : 5,
                            'gpt-4-1106-preview' : 3}

MODEL_INPUT_TOKEN_SUMM_LIMIT = {'gpt-3.5-turbo' : 3200,
                                'gpt-3.5-turbo-16k' : 14200,
                                'gpt-4' : 7200,
                                'gpt-4-1106-preview' : 125000}

MODEL_MAX_TOKEN_LIMIT = {'gpt-3.5-turbo' : 4097,
                        'gpt-3.5-turbo-16k' : 16385,
                        'gpt-4' : 8192,
                        'gpt-4-1106-preview' : 128000}

MODEL_COST = {'gpt-3.5-turbo' : 0.0015,
              'gpt-3.5-turbo-16k' : 0.003,
              'gpt-4' : 0.03,
              'gpt-4-1106-preview' : 0.01}


MAX_CONTEXT_QUESTIONS = {'gpt-3.5-turbo' : 10,
                        'gpt-3.5-turbo-16k' : 40,
                        'gpt-4' : 20,
                        'gpt-4-1106-preview' : 120}


# functions, prompts
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

    sim_docs = db.max_marginal_relevance_search_by_vector(query_embedded, k = MODEL_RELEVANT_DOC_NUMBER[model])
    results = [doc.metadata['source'].split("\\")[-1] + "-page-" + str(doc.metadata['page'] )+ ": " + doc.page_content.replace("\n", "").replace("\r", "") for doc in sim_docs]
    sources = "\n".join(results)

    return sources


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


# streamlit app
st.title("Semmelweis X Hiflylabs")
st.header("Semmelweis GenAI/LLM Anamnézis PoC")
st.write("Készítette: Hiflylabs")
#st.sidebar.image("https://hiflylabs.com/_next/static/media/greenOnDark.35e68199.svg", use_column_width=True)

st.sidebar.title("Leírás")
st.sidebar.markdown(
    """
    Beállítási lehetőségek\n

    1. OpenAI modell kiválasztása
        - gpt-3.5-turbo
        - gpt-3.5-turbo-16k
        - gpt-4
        - gpt-4-1106-preview

    1. Állítható prompt paraméterek
        - system prompt szövege
        - max kimeneti tokenszám
        - temperature
    """
)

MODEL = st.radio('A használni kívánt OpenAI modell kiválasztása', 
                 ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4', 'gpt-4-1106-preview'], horizontal=True)

prompt_expander = st.expander(label='Prompt beállítások')
with prompt_expander:
    cols=st.columns(2)
    with cols[0]:
        SYSTEM_MESSAGE = st.text_area('System prompt', value = default_system_prompt, height = 400)
    with cols[1]:
        TEMPERATURE = float(st.select_slider('Temperamentum', [str(round(i, 2)) for i in np.linspace(0.0, 2, 101)], value = '0.0')) 
        MAX_TOKENS = st.slider('Maximális kimeneti tokenszám', min_value = 1, max_value = MODEL_MAX_TOKEN_LIMIT[MODEL]-MODEL_INPUT_TOKEN_SUMM_LIMIT[MODEL], value = 512)



#### UPLOAD DOCS #####

DOCUMENTS_TO_CHOOSE_FROM = []
docs = []

uploaded_files = st.file_uploader("Töltse fel a fájlokat! Elfogadott formátumok: PDF, HTML, TXT, DOCX", 
                     type = ['pdf', 'html', 'txt', 'docx'], accept_multiple_files=True)
    
if uploaded_files:

    if not openai_api:
        st.warning('🔑🔒 A folytatáshoz adja meg az OpenAI API kulcsot az oldalsó panelen 🔑🔒')

    else:
    
        for uploaded_file in uploaded_files:

            filename = uploaded_file.name
            DOCUMENTS_TO_CHOOSE_FROM.append(filename)

            if uploaded_file.name.endswith(".pdf"):
                
                pdf_doc_chunks = load_pdf(uploaded_file, filename = filename)
                docs.extend(pdf_doc_chunks)
            
            elif uploaded_file.name.endswith('.txt'):

                txt_doc_chunks = load_txt(uploaded_file, filename = filename)
                docs.extend(txt_doc_chunks)

            elif uploaded_file.name.endswith('.docx'):

                docx_doc_chunks = load_docx(uploaded_file, filename = filename)
                docs.extend(docx_doc_chunks)

            elif uploaded_file.name.endswith('.html'):

                html_doc_chunks = load_HTML(uploaded_file, filename = filename)
                docs.extend(html_doc_chunks)


        docs_original = deepcopy(docs)


        #### STORE DOCS IN VECTOR DATABASE
        embeddings, db = create_db(docs)

#### END OF UPLOAD PART ####


#### Clear cache ####

col1, col2 = st.columns([2, 1])

with col1:
    st.caption("""A chat előzmények törléséhez és egy új beszélgetés kezdeményezéséhez törölje a cache-t.
            Ez a lépés javasolt dokumentumok hozzáadása vagy törlése utána is.""")
with col2:
    if st.button("Clear cache"):
        st.cache_data.clear()
        for key in st.session_state.keys():
            del st.session_state[key]

#### end of clear cache

if len(DOCUMENTS_TO_CHOOSE_FROM) == 0:
        st.write('Töltse fel a dokumentumokat!')

else:
    
    WHOLE_DOC, input_tokens = concat_docs_count_tokens(docs, encoding)
    st.write('Bemeneti tokenszám: ' + str(len(input_tokens)))
    # st.write('💰 Approx. cost of processing, not including completion:', str(round(MODEL_COST[MODEL] * (len(input_tokens) + 500) / 1000, 5)), 'USD')


    msg = st.chat_message('assistant')
    msg.write("Üdvözlöm! 👋 Tegyen fel kérdéseket a feltöltött dokumentumokkal kapcsolatban!")

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

            if len(input_tokens) <= MODEL_INPUT_TOKEN_SUMM_LIMIT[MODEL]: # maybe we can fit everything into the prompt, why not
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

            while (len(messages)-3 > MAX_CONTEXT_QUESTIONS[MODEL] * 2) or (current_token_count >= MODEL_INPUT_TOKEN_SUMM_LIMIT[MODEL]):

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
                if len(input_tokens) <= MODEL_INPUT_TOKEN_SUMM_LIMIT[MODEL]:
                    st.write('A válasz generálásához az összes feltöltött dokumentum felhasználásra került.')
                else:
                    st.write("A válasz generálásához az alábbi, relevánsnak ítélt dokumentumok lettek felhasználva:")
                    st.text(sources)
