from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def compare_multiple_papers(topic: str, paper_titles: list[str]) -> str:
    """
    Dynamically compares any number of papers (2, 3, 5, etc.) on a target topic.
    """
    print(f"\n[1] Initializing comparison of {len(paper_titles)} papers on topic: '{topic}'")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        collection_name="research_papers",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    combined_contexts = ""

    # Gather targeted context for each individual paper
    for title in paper_titles:
        if title == "ALL":
            continue

        print(f"    -> Retrieving context for: {title}")
        docs = vectorstore.similarity_search(
            query=topic,
            k=2,
            filter={"title": title}
        )

        paper_text = "\n".join([doc.page_content for doc in docs]) if docs else "No direct content found for this topic."
        combined_contexts += f"\n### Paper: {title}\n{paper_text}\n"

    if not combined_contexts.strip():
        return "Could not find relevant excerpts for the selected papers on this topic."

    print("[2] Generating contrastive analysis matrix...")

    prompt = PromptTemplate.from_template("""
    You are an expert academic research assistant. 
    Analyze and contrast the provided research excerpts across the specified papers regarding the following topic:
    "{topic}"

    Research Excerpts:
    {contexts}

    Instructions:
    1. Provide a side-by-side contrastive analysis highlighting key differences, methodologies, assumptions, and outcomes.
    2. Format the primary comparison using a clear, structured Markdown table with columns:
       | Paper | Methodology / Approach | Key Findings | Limitations / Trade-offs |
    3. Follow the table with a concise synthesis of how these works diverge or build upon one another.
    """)

    llm = ChatOllama(model="llama3.2")
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "topic": topic,
        "contexts": combined_contexts
    })

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