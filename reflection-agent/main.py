from dotenv import load_dotenv
import os
from typing import List , Sequence
from langchain_core.messages import BaseMessage , HumanMessage
from langgraph.graph import END , MessageGraph
from chains import generate_chain , reflect_chain
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
#nodes
REFLECT = "reflect"
GENERATE = "generate"

#state : refers to the current state of the graph/conversation/history , the agent appends the messages/response to the state
#The state is a list of messages
def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})


def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]#To trick the llm into thinking it is a human message

builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("This is the main entry point for the reflection agent.")
    print("Hello LangGraph")
    # initail state : To make the llm understand that it is a human message
    inputs = HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)
    response = graph.invoke(inputs)
    print("Response: ", response)