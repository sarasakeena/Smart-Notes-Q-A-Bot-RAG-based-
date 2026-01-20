from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
import os

docs=[]

for file in os.listdir("notes"):
    if file.endswith(".txt"):
        loader = TextLoader(f"notes/{file}")
        docs.extend(loader.load())


splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)

db = Chroma.from_documents(
    chunks,
    embedding=MistralAIEmbeddings(model="mistral-embed"),
    persist_directory="./chroma_db"
)
db.persist()
print("✅ Notes ingested with Mistral embeddings")


