import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "google/flan-t5-small"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        
        # Load free local LLM (CPU-friendly)
        tokenizer = AutoTokenizer.from_pretrained(llm_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(llm_model)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
        self.llm = HuggingFacePipeline(pipeline=pipe)
        print(f"[INFO] Local HuggingFace LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."

        # Truncate context to 5000 characters to avoid model input overflow
        context = context[:5000]

        prompt = f"Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"
        
        response = self.llm(prompt)  # returns string for HuggingFacePipeline
        if isinstance(response, list) and "generated_text" in response[0]:
            return response[0]["generated_text"]
        elif isinstance(response, str):
            return response
        else:
            return str(response)


if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is taylorshift attention?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
