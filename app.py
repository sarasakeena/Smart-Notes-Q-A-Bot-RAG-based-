import streamlit as st
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Smart Notes Bot", page_icon="🧠")
st.title("🧠 Smart Notes Q&A Bot")

# ---------- Initialize / Load LLM & DB ----------
@st.cache_resource
def load_chain():
    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0
    )

    embeddings = MistralAIEmbeddings(model="mistral-embed")

    # Load or create Chroma DB
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
Answer using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
""")

    return {
        "db": db,
        "retriever": retriever,
        "chain": (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
    }

resources = load_chain()
db = resources["db"]
retriever = resources["retriever"]
chain = resources["chain"]

# ---------- Session State for Chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Upload Notes ----------
uploaded_file = st.file_uploader("Upload your notes (.txt)", type=["txt"])
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    # Add chunks to Chroma
    db.add_texts(chunks)
    st.success("Notes added to database!")

    # Update retriever to include new notes
    retriever = db.as_retriever(search_kwargs={"k": 3})
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template("""
Answer using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
""")
        | ChatMistralAI(model="mistral-small-latest", temperature=0)
    )

# ---------- Chat Input ----------
user_input = st.chat_input("Ask something from your notes...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(user_input).content
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
