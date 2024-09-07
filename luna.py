from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import os
from typing import Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, AIMessage

from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from tools.time_tool import time_tool




# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

BASE_PROMPT_FILE_PATH = "base_prompt.txt"



class State(TypedDict):
    messages: Annotated[list, add_messages]




def main():

    graph_builder = StateGraph(State)

    # tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1))
    tools = [time_tool]
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-exp-0827")
    llm_with_tools = llm.bind_tools(tools)

    base_prompt = ""
    with open(BASE_PROMPT_FILE_PATH, 'r') as file:
        base_prompt = file.read()

    def chatbot(state: State):
        system_prompt = [("system", base_prompt)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(system_prompt)]}

    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=[time_tool])
    graph_builder.add_node("tools", tool_node)


    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        for event in graph.stream({"messages": [("user", user_input)]}, config=config, stream_mode="values"):
            last_message = event["messages"][-1]
            if isinstance(last_message, AIMessage) and not getattr(last_message, 'tool_calls', None) and last_message.content:
                print("Luna: " + last_message.content.rstrip())



if __name__ == "__main__":
    main() 