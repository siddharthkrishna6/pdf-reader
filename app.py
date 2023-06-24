from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import find_dotenv, load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

from PyPDF2 import PdfReader
import requests
import streamlit as st
import os


OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = "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"

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()

# PDF URL - training data
github_file_url = 'https://raw.githubusercontent.com/siddharthkrishna6/pdf-reader/blob/main/patient%202%20(14%20files%20merged).pdf'


# defining the prompt
template = """
You are a helpful assistant that can answer questions about healthcare
based on the patient interviews transcript: {docs}

Only use the factual information from the transcript to answer the question.

If you feel like you don't have enough information to answer the question, say "I don't know".

Your answers should be verbose and detailed.
"""


def download_file_from_github(url, destination):
    response = requests.get(url)
    with open(destination, "wb") as f:
        f.write(response.content)


def main():
    load_dotenv()
    st.set_page_config(page_title="IBS Interpreter")
    st.header("IBS Interpreter will provide answers from patient interviews 💬")
    query = st.text_input("Ask a question about the patients:")

    # Download the PDF from GitHub
    download_file_from_github(github_file_url, "temp_pdf.pdf")

    # Read the downloaded PDF
    try:
        transcript = PdfReader("temp_pdf.pdf")
        text = ""
        for page in range(transcript.getNumPages()):
            text += transcript.getPage(page).extractText()
    except Exception as e:
        st.write(f"Error reading PDF: {str(e)}")
        return

    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    db = FAISS.from_texts(chunks, embeddings)

    # show user input
    if query:
        docs = db.similarity_search(query)
        docs_page_content = " ".join([d.page_content for d in docs])
        chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.2)

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            template)
        human_template = "Answer the following question: {question}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            human_template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt])
        chain = LLMChain(llm=chat, prompt=chat_prompt)
        response = chain.run(question=query, docs=docs_page_content)
        response = response.replace("\n", "")
        st.write(response)

   
