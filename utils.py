import streamlit as st

from io import BytesIO
from PyPDF2 import PdfReader
import docx2txt
from bs4 import BeautifulSoup
import re

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

### Splitters for different data sources ###
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 100000, chunk_overlap = 200)
html_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, chunk_size=1500, chunk_overlap=200)


def add_context_to_doc_chunks(_docs):

    # adding the filename to each chunk my help the relevany search

    for i in _docs:
        i.page_content = i.metadata['source'].split("\\")[-1].split('.')[0] + ' --- ' + i.page_content

    return _docs


def clean_HTML(html):

    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


@st.cache_data()
def load_pdf(pdf_as_bytes, splitter = text_splitter, filename = 'pdf'):

    pdf_as_bytes = PdfReader(pdf_as_bytes)

    #text = ''
    DOCS = []

    for pagenum, page in enumerate(pdf_as_bytes.pages):

        page_text = page.extract_text()

        #text += page_text

        #text_splitted = splitter.split_text(page_text)
        docs = [Document(page_content=page_text, metadata={'source' : filename, 'page' : str(pagenum+1)})]
        docs = add_context_to_doc_chunks(docs)
        DOCS.append(docs)

    DOCS = [item for sublist in DOCS for item in sublist]

    return DOCS#, text



@st.cache_data()
def load_txt(file, splitter = text_splitter, filename = 'txt'):

    DOCS = []

    text = file.read().decode("utf-8")
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

@st.cache_data()
def load_HTML(file, splitter = html_splitter, filename = 'html'):

    DOCS = []

    text = file.read().decode("utf-8")
    text = clean_HTML(text)
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
