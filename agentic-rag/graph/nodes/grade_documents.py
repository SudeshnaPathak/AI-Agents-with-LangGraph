from typing import Any , Dict
from graph.state import GraphState
from graph.nodes.retriever import retrieval_grader

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question.If any document is not relevant, we will set a flag to run web search.


    Args:
        state (dict): The current state of the graph
    
        
    Returns:
        state (dict) : Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    documents = state["documents"] # Get the documents from the state
    question = state["question"] # Get the question from the state
    print(f"Question: {question}")

    filtered_docs = []
    web_search = False

    for doc in documents:
        # Check if the document is relevant to the question
        score = retrieval_grader.invoke({"document": doc, "question": question})
        grade = score["binary_score"]
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
    # Update the state with the filtered documents and web_search flag
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search,
    }
