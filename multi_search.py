from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def multi_search(complex_query):
    """
        Breaks a complex query into sub queries and searches
        the db for each one and combines the results
    """
    print("Analyzing the query...")
    llm = ChatOllama(model = "llama3.2")

    # Break the question into sub queries by llm
    decompose_prompt = PromptTemplate.from_template("""
    You are an expert research assistant. Break the following complex question into
    simpler and distinct search queries. Output only the 2 queries, one per line
    Complex Question : {question}                                                                                                        
""")
    # Convert llm's raw message into string
    decompose_chain = decompose_prompt | llm | StrOutputParser()
    raw_sub_queries = decompose_chain.invoke({"question": complex_query})

    # Split string into list of queries
    sub_queries = [q.strip() for q in raw_sub_queries.split('\n') if q.strip()]
    print(f"Generated Sub-queries: \n - {sub_queries[0]} \n - {sub_queries[1]}")

    # Multi search
    embeddings = OllamaEmbeddings(model = "nomic-embed-text")
    vectorstore = Chroma(
        collection_name = "research_papers",
        embedding_function = embeddings,
        persist_directory = "./chroma_db"
    )

    combined_context = ""
    for i, sub_query in enumerate(sub_queries):
        docs = vectorstore.similarity_search(query=sub_query, k=2)
        combined_context += f" Context from sub query {i+1}\n"
        combined_context += "\n".join([d.page_content for d in docs]) + "\n\n"

    # Combining into final answer
    print ("Combining final answer")
    combined_prompt = PromptTemplate.from_template("""
    You are an expert research assistant. Answer the original complex questions
    using only the provided combined context. The context was gathered by running multiple sub-queries.
    Combine the information intelligently.
    Original complex question : {question}
    Combined context : {context}
    Final answer :                                                                                                                                                                                                                                                                
""")
    
    combined_chain = combined_prompt | llm | StrOutputParser()
    final_answer = combined_chain.invoke({
        "question" : complex_query,
        "context" : combined_context
    })

    return final_answer

if __name__ == "__main__":
    test_question = "How are the results vary from paper 1 and to those of paper 2"
    result = multi_search(test_question)

    print ("--------------------")
    print (result)
    