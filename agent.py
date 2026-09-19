import os
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver



class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Inside build_agent(checkpointer) or default initialization:
def build_agent(checkpointer=None):
    if checkpointer is None:
        checkpointer = MemorySaver()
        
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3
    )

    def call_model(state: AgentState):
        system = SystemMessage(content="You are CareerMate, an AI Career Assistant. Provide concise, mobile-friendly advice.")
        response = llm.invoke([system] + state["messages"])
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("assistant", call_model)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)

    return builder.compile(checkpointer=checkpointer)