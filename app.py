from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever  # Your test's success path
from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from typing import Dict
import os
import uvicorn

app = FastAPI(title="Compliance Doc Analyzer")

llm = ChatOllama(model="mistral", temperature=0.5)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

store: Dict[str, BaseChatMessageHistory] = {}
local_vectorstore = None
local_retriever = None

prompt = ChatPromptTemplate.from_template(
    "Analyzer: Context: {context}\nHistory: {history}\nQuery: {input}\nAnswer:"
)

@tool
def enrich_regulation(query: str) -> str:
    """Enrich query with latest regulatory update (mock for demo)."""
    return f"Update for '{query}': GDPR 2025 — fines €20M max."

llm_with_tools = llm.bind_tools([enrich_regulation])

class ComplianceMemory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages = []
        self.entities = {}

    def add_messages(self, messages):
        self.messages.extend(messages)
        for msg in messages:
            if "fines" in msg.content.lower():
                self.entities["focus"] = "fines"
            if "EU" in msg.content:
                self.entities["region"] = "EU"

    def clear(self):  # <-- ADD THIS
        self.messages = []
        self.entities = {}

    def messages(self):
        ent_str = "\n".join([f"{k}: {v}" for k, v in self.entities.items()])
        return self.messages[-10:] + [AIMessage(content=f"Prefs: {ent_str}")]
    
def get_memory(session_id: str):
    if session_id not in store:
        store[session_id] = ComplianceMemory(session_id)
    return store[session_id]

def build_hybrid_retriever():
    global local_retriever
    if local_retriever is None:
        return RunnablePassthrough()
    sample_docs = TextLoader("gdpr_guideline.txt").load()
    sample_splits = splitter.split_documents(sample_docs)
    bm25 = BM25Retriever.from_documents(sample_splits)
    hybrid = EnsembleRetriever(retrievers=[local_retriever, bm25], weights=[0.7, 0.3])
    return hybrid

base_chain = (
    {"context": build_hybrid_retriever(), "input": RunnablePassthrough(), "history": RunnablePassthrough()}  # Add this for memory
    | prompt
    | llm_with_tools
    | StrOutputParser()
)

memory_chain = RunnableWithMessageHistory(
    base_chain, get_memory,
    input_messages_key="input", history_messages_key="history"
)

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"

@app.post("/query")
async def query_docs(request: QueryRequest):
    try:
        config = {"configurable": {"session_id": request.session_id}}
        response = memory_chain.invoke({"input": request.query}, config=config)
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_res = enrich_regulation.invoke(response.tool_calls[0]['args'])
            response += f"\nUpdate: {tool_res}"
        return {"response": response, "entities": get_memory(request.session_id).entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        path = f"docs/{file.filename}"
        os.makedirs("docs", exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
        loader = TextLoader(path) if path.endswith('.txt') else PyPDFLoader(path)
        new_docs = loader.load()
        new_splits = splitter.split_documents(new_docs)
        global local_vectorstore, local_retriever
        if local_vectorstore is None:
            local_vectorstore = FAISS.from_documents(new_splits, embeddings)
        else:
            local_vectorstore.merge_documents(new_splits)
        local_retriever = local_vectorstore.as_retriever(k=3)
        return {"status": "Ingested", "chunks": len(new_splits)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)