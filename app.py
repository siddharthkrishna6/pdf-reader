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
from PyPDF2 import PdfReader
import textwrap


OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = "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"


load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()


def get_response_from_query(db, query, k=4):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.2)

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the patient interviews transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


# Example usage:
#video_url = "https://www.youtube.com/watch?v=L_Guz73e6fw"
#db = text_reader()


transcript = PdfReader(
    '/Users/siddharthkrishna/Downloads/patient 2 (14 files merged).pdf')
#transcript = loader.load()
raw_text = ''
for i, page in enumerate(transcript.pages):
    text = page.extract_text()
    if text:
        raw_text += text

text_splitter = CharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_text(raw_text)

db = FAISS.from_texts(docs, embeddings)


query = "What is patient 3's main complaint?"
response, docs = get_response_from_query(db, query)
print(textwrap.fill(response, width=50))
