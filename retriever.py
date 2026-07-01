import os
import datasets
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever


os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

docs = [Document(
    page_content='\n'.join([
        f"Name: {guest['name']}",
        f"Relation: {guest['relation']}",
        f"Description: {guest['description']}",
        f"Email: {guest['email']}"
    ]),
    metadata = {"name": guest['name']}
)
for guest in guest_dataset]

#BM25 retriever
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3

#Semantic retriever
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                   model_kwargs={'device': device})
vectorstore  = FAISS.from_documents(documents=docs,embedding=embeddings)
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#Hybrid retriever
hybrid_retriever  = EnsembleRetriever(
    retrievers  =[bm25_retriever,semantic_retriever],
    weights = [0.3,0.7]
)


@tool
def guest_info_tool(query:str)-> str:
    """Search the guest list and return information about a person by name or relation."""
    results = hybrid_retriever.invoke(query)
    if results:
        return '\n\n'.join([doc.page_content for doc in results[:3]])
    else:
        return "No matching guest information found."
    

tools = [guest_info_tool]

if __name__ == "__main__":
    print(guest_info_tool.invoke('tell me about ada'))