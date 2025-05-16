from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever

#retriever node
def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"] # Get the question from the state
    print(f"Question: {question}")
    documents = retriever.invoke(question) # Retrieve documents using the retriever using similarity search
    print("---Documents Retrieved---")
    return {"documents": documents, "question": question} #update the state with the documents