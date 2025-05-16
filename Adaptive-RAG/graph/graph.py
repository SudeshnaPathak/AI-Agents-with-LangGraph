from dotenv import load_dotenv
from langgraph.graph import END , StateGraph
from graph.consts import RETRIEVE , GENERATE , GRADE_DOCUMENTS , WEBSEARCH
load_dotenv()
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader
from graph.chains.router import RouteQuery , question_router
from graph.nodes import generate , grade_documents , retrieve , web_search
from graph.state import GraphState

def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == "vectorstore":
        print("---DECISION: ROUTE QUESTION TO RAG---")
        return RETRIEVE
    else:
        print("---DECISION: ROUTE QUESTION TO WEBSEARCH---")
        return WEBSEARCH

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE
    
def grade_generation_grounded_in_documents_and_question(state: GraphState)-> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke(
            {"question": question, "generation": generation}
        )
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS , RE-TRY---")
        return "not suported"
    
workflow = StateGraph(GraphState)
#Add nodes to the workflow
workflow.add_node(RETRIEVE , retrieve)
workflow.add_node(GRADE_DOCUMENTS , grade_documents)
workflow.add_node(GENERATE , generate)
workflow.add_node(WEBSEARCH , web_search)
#Set Conditional Entry Point
workflow.set_conditional_entry_point(
    route_question,
    {
        RETRIEVE: RETRIEVE,
        WEBSEARCH: WEBSEARCH,
    },
)

#Add edges to the workflow
workflow.add_edge(RETRIEVE , GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
    )

workflow.add_edge(WEBSEARCH , GENERATE)
workflow.add_edge(GENERATE , END)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")




