from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import Model2vecEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import gradio as gr
import os

# -------------------------------
# 1️⃣ Load PDF
file_path = r"C:\Users\Mohamed\Documents\Economy_of_Egypt.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()  # load all pages as documents

# -------------------------------
# 2️⃣ Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(docs)

# -------------------------------
# 3️⃣ Initialize embeddings
embeddings = Model2vecEmbeddings("minishlab/potion-base-8M")

# -------------------------------
# 4️⃣ Create vector store with chunks
vector_store = Chroma.from_documents(
    documents=chunks,                     
    embedding=embeddings,                 
    collection_name="egypt_economy",      
    persist_directory="./chroma_langchain_db"  
)

# -------------------------------
# 5️⃣ Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# -------------------------------
# 6️⃣ Initialize LLM
llm = ChatOllama(
    model="llama3:8b",
    temperature=0
)

# -------------------------------
# 7️⃣ Create RAG chain
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""")

rag_chain = (
    {"context": retriever, "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)

# -------------------------------
# 8️⃣ Ask a question
query = "What is Egypt’s global economic ranking as of 2025?"
response = rag_chain.invoke(query)

print("Answer:\n", response)

# 2️⃣ Define functions for Gradio
def ask_question(query):
    """Ask a question to the RAG system."""
    if not query.strip():
        return "Please type a question."
    response = rag_chain.invoke(query)
    return response

# -------------------------------
# 3️⃣ Create Gradio GUI
with gr.Blocks(css="""
    body {background-color: #f7f9fc; font-family: 'Arial', sans-serif;}
    .gradio-container {max-width: 700px; margin: auto; padding: 20px; border-radius: 12px; box-shadow: 0px 6px 20px rgba(0,0,0,0.1);}
    .gr-button {background-color: #4B70FF; color: white; border-radius: 8px; font-size: 16px;}
    .gr-textbox {border-radius: 8px; font-size: 16px;}
""") as demo:
    
    gr.Markdown("## 📘 Egypt Economy RAG Assistant", elem_id="title")
    
    with gr.Row():
        ask_input = gr.Textbox(
            label="Ask a question",
            placeholder="Type your question here...",
            lines=2
        )
        ask_button = gr.Button("Ask")

    answer_output = gr.Textbox(
        label="Answer",
        interactive=False,
        lines=6
    )

    # Connect button to function
    ask_button.click(fn=ask_question, inputs=ask_input, outputs=answer_output)

# -------------------------------
# 4️⃣ Launch GUI
if __name__ == "__main__":
    demo.launch()