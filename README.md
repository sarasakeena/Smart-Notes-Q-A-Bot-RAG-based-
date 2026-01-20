# 🧠 Smart Notes Q&A Bot (RAG-based)

A **Retrieval-Augmented Generation (RAG) chatbot** built using **LangChain** and **Mistral AI**, designed for querying your personal notes.  
Upload your notes and ask questions — the bot answers using **only your content**, preventing hallucinations.

---

## **Features**

- **Dynamic Notes Upload**: Upload `.txt` files (support for PDF/DOCX in future upgrades)
- **RAG-based Answers**: Answers are grounded strictly in the uploaded notes
- **Semantic Search**: Uses **Mistral embeddings** for meaning-based retrieval
- **Top-k Retrieval**: Returns the most relevant note chunks
- **Interactive Chat Interface**: Built with **Streamlit** with persistent chat history
- **Persistent Vector Database**: Uses **Chroma** to store embeddings and speed up retrieval
- **Controlled LLM Behavior**: Uses prompt templates and `temperature=0` for factual answers

---

## **Demo Screenshot**

<img width="1038" height="542" alt="image" src="https://github.com/user-attachments/assets/2d42a3a9-bca5-447b-b3f6-d9e3bb83dab5" />

---

## **Getting Started**

### **1. Clone the Repository**

```bash
git clone https://github.com/<your-username>/Smart-Notes-Q-A-Bot-RAG-based-.git
cd Smart-Notes-Q-A-Bot-RAG-based-

##2. Create a Virtual Environment & Install Dependencies
2. Create a Virtual Environment & Install Dependencies
python -m venv env

# On Windows
env\Scripts\activate

# On Mac/Linux
source env/bin/activate

pip install -r requirements.txt

3. Add Your Environment Variables

Create a .env file in the project root:

MISTRAL_API_KEY=your_api_key_here

Do not push .env to GitHub.

4. Run the App
streamlit run app.py


Upload your notes in .txt format

Ask questions and get answers directly from your notes

Folder Structure
Smart-Notes-Q-A-Bot-RAG-based-/
│
├─ app.py                  # Streamlit app
├─ chroma_db/              # Persistent vector DB
├─ notes/                  # Optional folder for raw notes
├─ requirements.txt        # Dependencies
├─ README.md               # Project README
├─ .gitignore              # Files to ignore in Git
└─ .env                    # Your API key (do NOT push)

Technologies Used

Python 3.x

LangChain
 – RAG pipeline

Mistral AI
 – Embeddings & LLM

Chroma
 – Vector DB

Streamlit
 – Web interface

License

This project is licensed under the MIT License. 

Future Improvements

Support PDF/DOCX uploads

Show source references for answers

Add conversation memory for multi-turn chat

Author

Sara Sakeena Abdul Muthalib

GitHub: https://github.com/sarasakeena

LinkedIn: https://www.linkedin.com/in/sara-sakeena-07ba09353/
