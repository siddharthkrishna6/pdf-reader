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

from io import BytesIO
import requests
import streamlit as st
import os
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage


OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = "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"

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()

# PDF URL - training data
pdf_url = 'https://example.com/path/to/pdf.pdf'


# defining the prompt
template = """
You are a helpful assistant that can answer questions about healthcare
based on the patient interviews transcript: {docs}

Only use the factual information from the transcript to answer the question.

If you feel like you don't have enough information to answer the question, say "I don't know".

Your answers should be verbose and detailed.
"""


def extract_text_from_pdf(pdf_bytes):
    resource_manager = PDFResourceManager()
    text_stream = BytesIO()
    laparams = LAParams()

    with TextConverter(resource_manager, text_stream, laparams=laparams) as device:
        interpreter = PDFPageInterpreter(resource_manager, device)
        for page in PDFPage.get_pages(pdf_bytes):
            interpreter.process_page(page)

    return text_stream.getvalue().decode()


def main():
    load_dotenv()
    st.set_page_config(page_title="IBS Interpreter")
    st.header("IBS Interpreter will provide answers from patient interviews 💬")
    query = st.text_input("Ask a question about the patients:")

    # Download the PDF from the web
    response = requests.get(pdf_url)
    pdf_bytes = BytesIO(response.content)

    # Read the downloaded PDF
    text = extract_text_from_pdf(pdf_bytes)

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


if __name__ == '__main__':
    main()
