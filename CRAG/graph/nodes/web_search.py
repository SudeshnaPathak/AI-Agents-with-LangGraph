from typing import Any, Dict
from langchain.schema import Document 
from langchain_tavily import TavilySearch
from graph.state import GraphState
from dotenv import load_dotenv
load_dotenv()

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"] # Get the question from the state
    print(f"Question: {question}")
    documents = state["documents"] # Get the documents from the state
    print("---Documents Retrieved---")

    tavily_results = web_search_tool.invoke({"query" : question}) # Perform web search using Tavily
    joined_tavily_result = "\n".join([tavily_result["content"] for tavily_result in tavily_results])
    print(f"Web search results: {joined_tavily_result}")

    # Create Document objects from the web search results
    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    # Update the state with the web search results
    return {
        "documents": documents,
        "question": question
    }


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None}) # No relevant documents