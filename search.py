from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

def build_hybrid_retriever(raw_text):
    embeddings = OllamaEmbeddings(model = "nomic-embed-text")

    # Connect to DB
    vectorstore = Chroma(
        collection_name = "research_papers",
        embedding_function = embeddings,
        persist_directory = "./chroma_db"
    )

    vector_retriever = vectorstore.as_retriever(search_kwargs = {"k" : 3})
    doc_for_bm25 = [Document(page_content = raw_text)]

    # Initialize BM25 and returns top 3 keyword matches
    bm25_retriever = BM25Retriever.from_documents(doc_for_bm25)
    bm25_retriever.k = 3

    # Ensemble both retrievers with equal weights
    hybrid_retriever = EnsembleRetriever(
        retrievers = [bm25_retriever, vector_retriever],
        weights = [0.5, 0.5]
    )
    return hybrid_retriever

if __name__ == "__main__":
    sample_text = "The forces on the blade elements are solely determined by the lift and drag coefficients"
    retriever = build_hybrid_retriever(sample_text)

    test_query = "How is the force on the blade determined"
    print (f"\nExecuting query : '{test_query}'")

    results = retriever.invoke(test_query)

    print ("\n Top search results")
    for i, doc in enumerate(results):
        print(f"\n Result : {i+1}")
        print(f"\n Content : {doc.page_content[:200]}")