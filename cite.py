from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def cite_answers(query, paper_title="ALL"):
    """
        Retrieves content and cites the sources accordingly
    """

    print (f"searching database for {query}")
    embeddings = OllamaEmbeddings(model = "nomic-embed-text")
    vectorstore = Chroma(
        collection_name = "research_papers",
        embedding_function = embeddings,
        persist_directory = "./chroma_db"
    )
    search_kwargs = {"k": 4}
    if paper_title != "ALL":
        print(f"Filtering for paper: {paper_title}")
        search_kwargs["filter"] = {"title": paper_title}
        
    docs = vectorstore.similarity_search(query=query, **search_kwargs)


    print("[2] Formatting Context with Source IDs...")
    formatted_context = ""
    source_mapping = {} 
    unique_papers_seen = {} 
    current_id = 1
    
    for doc in docs:
        title = doc.metadata.get("title", "Unknown Source")
        year = doc.metadata.get("year", "Unknown Year")
        
        # If we haven't seen this paper yet in this search, give it a new ID
        if title not in unique_papers_seen:
            unique_papers_seen[title] = current_id
            source_mapping[current_id] = f"{title} ({year})"
            current_id += 1
            
        # Apply the correct ID to the chunk
        source_id = unique_papers_seen[title]
        formatted_context += f"Source ID: [{source_id}]\nContent: {doc.page_content}\n\n"

    print("[3] Generating Cited Response...")

    prompt = PromptTemplate.from_template(
        """
            You are an expert research assistant. Answer the user's question
            using only the provided context. 
            Critical Instruction:
            Everytime you use state a fact from the context, you must append the corresponding
            source ID at the end of the sentence.
            Example : "The speed of the system increased by 40% [2]."
            Do not create a bibliography or reference list at the end. I will handle that.
            Question : {question}
            Context : {context}
            Answer:
        """
    )
    llm = ChatOllama(model = "llama3.2")
    chain = prompt | llm | StrOutputParser()

    raw_answer = chain.invoke({
        "question" : query,
        "context" : formatted_context
    })

    final_output = f"{raw_answer}\n\n"
    final_output += "------------------\n"
    final_output += "References: \n"

    for source_id, citation in source_mapping.items():
        if f"[{source_id}]" in raw_answer:
            final_output += f"[{source_id}] {citation}\n"

    return final_output


if __name__ == "__main__":
    test_query = "What are the insights of using nickel and iron?"
    result = cite_answers(test_query)
    print ("------------\n")
    print (result)