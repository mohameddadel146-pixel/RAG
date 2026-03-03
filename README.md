# Egypt Economy RAG Assistant

A Retrieval-Augmented Generation (RAG) system for querying information from the **"Economy of Egypt" PDF** using:

- **LangChain**
- **Chroma Vector Store**
- **Model2vec Embeddings**
- **ChatOllama LLM**
- **Gradio GUI** for a simple web interface

---

## Features

- Load and process PDF documents.
- Split PDF into semantic chunks for retrieval.
- Generate vector embeddings using Model2vec.
- Store and query data efficiently using Chroma.
- Ask questions to a LLM augmented with document retrieval.
- Clean, responsive Gradio interface.

---

## Requirements

- Python 3.11+
- Virtual environment 
- VS Code installed (optional but recommended)
- Ollama installed for LLM (`ChatOllama`)

---

## Setup Instructions (Windows)

### 1️⃣ Create and activate a virtual environment

```powershell
# Navigate to your project folder
cd C:\Users\Mohamed\Desktop\ITI-MO

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
& .venv\Scripts\Activate.ps1
2️⃣ Install Python dependencies
pip install --upgrade pip
pip install langchain-community langchain-text-splitters model2vec langchain-chroma langchain-ollama gradio

Optionally, save requirements for future use:

pip freeze > requirements.txt
3️⃣ Place your PDF

Put your PDF at:

C:\Users\Mohamed\Documents\Economy_of_Egypt.pdf

Make sure the path in your code matches.

4️⃣ Pull the Ollama LLM
ollama pull llama3:8b

Make sure your Ollama API/server is running.

run the code

Type a question in the textbox and click Ask.

The RAG system will fetch relevant information from the PDF and generate a response.

Example Questions
What is Egypt’s global economic ranking as of 2025?
What type of economy does Egypt have?

What major international economic groups is Egypt a member of?

What was the inflation rate (CPI) in 2025?

What is Egypt’s unemployment rate in Q1 2025?

What is the current government debt as a percentage of GDP?

What is Egypt’s HDI ranking in 2025?

What are Egypt’s main industries?

What were Egypt’s total exports in 2023?

Who are Egypt’s top export partners?