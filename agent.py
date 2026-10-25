import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import chat_pdf  # the shared multi-provider fallback chain (Groq -> SambaNova -> NVIDIA ... -> Gemini 2.5 Flash)

SYSTEM = "You are CareerMate, an AI Career Assistant. Provide concise, mobile-friendly advice."
KEEP_MESSAGES = 6          # only the last few messages go to the model (fewer tokens per call)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p if isinstance(p, str) else str(p.get("text", "")) if isinstance(p, dict) else str(getattr(p, "text", ""))
                       for p in content)
    return str(content)


def _to_chat(messages: list[BaseMessage]) -> list[dict]:
    out = [{"role": "system", "content": SYSTEM}]
    for m in messages[-KEEP_MESSAGES:]:
        out.append({"role": "assistant" if m.type == "ai" else "user", "content": _text(m.content)})
    if out[1:] and out[1]["role"] == "assistant":      # providers want the first turn to be the user's
        out.pop(1)
    return out


def build_agent(checkpointer=None):
    if checkpointer is None:
        checkpointer = MemorySaver()

    def call_model(state: AgentState):
        # chat_pdf.llm tries every configured provider / model / key until one answers.
        text = chat_pdf.llm(_to_chat(state["messages"]), max_tokens=900)
        return {"messages": [AIMessage(content=text)]}

    builder = StateGraph(AgentState)
    builder.add_node("assistant", call_model)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)
    return builder.compile(checkpointer=checkpointer)
