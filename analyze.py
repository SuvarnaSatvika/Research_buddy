from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate

def compare_papers(topic, paper1_title, paper2_title):
    """
        Makes the LLM compare two papers side-by-side using metadata filter
    """
    print (f"Comparing both papers on {topic}")

    # Initialize embedding model and database
    embeddings = OllamaEmbeddings(model = "nomic-embed-text")
    vectorstore = Chroma(
        collection_name = "research_papers",
        embedding_function = embeddings,
        persist_directory = "./chroma_db" 
    )

    # Retrieve chunks from both papers using metadata filtering
    docs1 = vectorstore.similarity_search(query = topic, k = 3, filter = {"title":paper1_title})
    context1 = "\n\n".join([d.page_content for d in docs1])

    docs2 = vectorstore.similarity_search(query = topic, k = 3, filter = {"title":paper2_title})
    context2 = "\n\n".join([d.page_content for d in docs2])

    # Defining template
    prompt = PromptTemplate.from_template("""
    You are an expert research assistant. Analyze the differences
    between the given two papers regarding the following topic : {topic}
    
    Paper 1 - {paper1_title} Context : {context1}
    
    Paper 2 = {paper2_title} Context : {context2}
                                          
    Output a detailed contrastive analysis highlighting their 
    methodological differences. You must use a Markdown table to
    compare them side-by-side. Keep it concise and academic.                                                                                  
""")
    # Initialize the local model
    llm = ChatOllama(model = "llama3.2")

    print ("Synthesizing comparison")
    chain = prompt | llm
    response = chain.invoke({
        "topic" : topic,
        "paper1_title" : paper1_title,
        "paper2_title" : paper2_title,
        "context1" : context1,
        "context2" : context2
    })

    return response.content

def timeline_tracker(topic):
    """
        Sorts retrieved chunks chronologically to track how the topic evolved over time
    """
    print (f"Building timeline for {topic}")
    embeddings = OllamaEmbeddings(model = "nomic-embed-text")
    vectorstore = Chroma(
        collection_name = "research_papers",
        embedding_function = embeddings,
        persist_directory = "./chroma_db"
    )

    docs = vectorstore.similarity_search(query = topic, k = 6)
    sorted_docs = sorted(docs, key = lambda x : str(x.metadata.get('year', '0000')))

    # Format the context for clarity for llm
    chronological_context = ""
    for d in sorted_docs:
        chronological_context += f"Year : {d.metadata.get('year')} | Source : {d.metadata.get('title')}\n{d.page_content}\n\n"

    prompt = PromptTemplate.from_template("""
    You are an expert research assistant.
    Based on the following chronological excerpts, track the evolution
    of the topic - {topic}
    Chronological Context : 
    {context}
    Provide a historical timeline summarizing how this concept
    has changed over time. Use bullet points mapped to specific years if applicable.                                                                                                                                                                                                                                                            
""")
    
    llm = ChatOllama(model = "llama3.2")
    print("Generating Timeline")
    chain = prompt | llm
    response = chain.invoke({
        "topic" : topic,
        "context" : chronological_context
    })

    return response.content

if __name__ == "__main__":
    topic_to_compare = "spin waves properties in iron and nickel"
    paper1 = "article 1"
    paper2 = "article 2"

    #print (compare_papers(topic_to_compare, paper1, paper2))
    print (timeline_tracker(topic_to_compare))