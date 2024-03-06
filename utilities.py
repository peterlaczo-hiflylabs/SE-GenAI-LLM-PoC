import re
import docx2txt
import streamlit as st

from io import BytesIO
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


### Splitters for different data sources ###
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100000, chunk_overlap = 200)

def text_to_html(document):
    html = ""
    for paragraph in document.split('\n'):
        html += f"<p>{paragraph}</p>"
    return html


def add_context_to_doc_chunks(_docs):

    # adding the filename to each chunk my help the relevany search

    for i in _docs:
        i.page_content = i.metadata['source'].split("\\")[-1].split('.')[0] + ' --- ' + i.page_content

    return _docs


@st.cache_data()
def load_txt(text, splitter = text_splitter, filename = 'txt'):

    DOCS = []

    text = re.sub(r"\n\s*\n", "\n\n", text)

    text_splitted = splitter.split_text(text)
    docs = [Document(page_content=t, metadata={'source' : filename, 'page' : 'all'}) for t in text_splitted]
    docs = add_context_to_doc_chunks(docs)
    DOCS.append(docs)

    DOCS = [item for sublist in DOCS for item in sublist]

    return DOCS


@st.cache_data()
def load_docx(file, splitter = text_splitter, filename = 'docx'):

    DOCS = []

    text = docx2txt.process(file) 
    text = re.sub(r"\n\s*\n", "\n\n", text)

    text_splitted = splitter.split_text(text)
    docs = [Document(page_content=t, metadata={'source' : filename, 'page' : 'all'}) for t in text_splitted]
    docs = add_context_to_doc_chunks(docs)
    DOCS.append(docs)

    DOCS = [item for sublist in DOCS for item in sublist]

    return DOCS


#@st.cache_data()
def create_db(_docs):

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(_docs, embeddings)

    return embeddings, db


def concat_docs_count_tokens(docs, tiktoken_encoding):

    WHOLE_DOC = ' '.join([i.page_content for i in docs])
    input_tokens = tiktoken_encoding.encode(WHOLE_DOC)

    return WHOLE_DOC, input_tokens
