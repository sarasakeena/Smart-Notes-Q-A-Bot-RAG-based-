from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0
)

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=MistralAIEmbeddings(model="mistral-embed")
)

retriever = db.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""")

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

print("💬 Smart Notes Bot (type 'exit' to quit)")
while True:
    q = input("You: ")
    if q.lower() == "exit":
        break
    result = chain.invoke(q)
    print("AI:", result.content)
