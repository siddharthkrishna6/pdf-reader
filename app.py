from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
import streamlit as st
import PyPDF2
from PyPDF2 import PdfReader
import textwrap


OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = "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"


load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    if pdf is not None:
        transcript = PdfReader(pdf)
        text = ""
        for page in transcript.pages:
            text += page.extract_text()

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
        query = st.text_input("Ask a question about your PDF:")
        if query:
            docs = db.similarity_search(query)
            docs_page_content = " ".join([d.page_content for d in docs])
            chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.2)
            # Template to use for the system message prompt
            template = """
        You are a helpful assistant that that can answer questions about healthcare
        based on the patient interviews transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """
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
