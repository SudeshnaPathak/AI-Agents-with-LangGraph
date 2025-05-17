import operator
from typing import Annotated , TypedDict , Union
from langchain_core.agents import AgentAction , AgentFinish

class AgentState(TypedDict):
    """
    The state of the agent.
    """
    input: str
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction , str]] , operator.add] 
    #type: list[tuple[AgentAction , str]] = [] , operator.add means that the list is mutable and can be modified in place
    #AgentAction contains all details about which tool to use and what input to give it 
    #AgentFinish contains the final output of the agent