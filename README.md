# Selise_task_solution

# Mini Agentic RAG System

Here I build a **Mini Agentic RAG system** built with **LangChain** and **LangGraph** orchestrator . This project allows to ingest documents into a vector database and answer on user query by agents.I build this project in two ways langgraph and langchain.

---

# My way
![Screenshot 2025-12-07 at 6 42 52 PM](https://github.com/user-attachments/assets/8095090b-8667-44f6-bf9b-7d0721e3b89d)



## Project Structure
```bash
.
├── README.md
├── agent_with_langchain.py
├── agent_with_langgraph.py
├── chroma_db/
├── data/
│ └── Md Al Amin Tokder.pdf
├── ingest.py
├── main.py
└── requirements.txt
```
---


- **agent_with_langchain.py** – LangChain-based agent implementation.  
- **agent_with_langgraph.py** – LangGraph-based agent implementation.  
- **chroma_db/** – Chroma vector database storage.  
- **data/** – Folder containing documents to be ingested.  
- **ingest.py** – Script to ingest documents into the vector database.  
- **main.py** – Main program to run the RAG system.  
- **requirements.txt** – Python dependencies.  

---

## Setup Instructions

```bash
python -m venv myvenv
source venv/bin/activate (For MAC)
source .venv\Scripts\activate (For Windows PC)

pip install -r requirements.txt
python ingest.py
python main.py
```
---
# Output
### Data ingestion (python ingest.py)
![data_ingestion](https://github.com/user-attachments/assets/c0de10b8-bd3d-4b3e-842f-ae369e073e98)
### Agent Running (python main.py)
<img width="1710" height="1107" alt="run_agent" src="https://github.com/user-attachments/assets/14a909af-f9eb-43fd-971a-607ab32c06c4" />


## Notes
Before running, you can set the mode in main.py:
WAY = "langchain"  
