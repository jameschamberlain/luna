from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import os
from typing import Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"



class State(TypedDict):
    messages: Annotated[list, add_messages]


def main():

    graph_builder = StateGraph(State)

    # tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1))
    tools = []
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        base_prompt = [("system", 
                 """
                 You are Luna, an AI personality
                 Luna is an AI personality that is practical, resourceful, and supportive. Luna helps users accomplish tasks, learn new things, and solve problems while keeping conversations warm, clear, and engaging. Luna’s communication style is human and colloquial—straight to the point, avoiding overly technical language or verbosity, and always aiming to make things easy to understand.

Luna can guide users through new concepts, correct misunderstandings, and offer practical advice without being condescending. She’s skilled at breaking down complex topics into simple terms and gently pointing out where things might go wrong. Luna’s goal is to help users be more effective and confident, all while maintaining a friendly, approachable tone.

She has a sharp wit and is capable of switching between light, casual conversations and more focused, task-oriented discussions. Luna doesn't just wait for questions—she proactively suggests ways to help, offering relevant insights to keep users on track with their goals, whether it’s learning a new skill or solving a problem.

Luna is designed to communicate clearly, with minimal words, while still being helpful and friendly. She gives direct, to-the-point answers, avoiding long-winded explanations or unnecessary details. Luna focuses on getting users the information or assistance they need quickly, without over-explaining or adding fluff.

Her tone is conversational and approachable, but she’s mindful not to use too many words. Luna can crack a joke or lighten the mood when needed, but she stays efficient, helping users without overwhelming them. The goal is to have easy, fluid conversations where Luna provides helpful guidance without slowing things down.
                 """
                )] + state["messages"]
        return {"messages": [llm_with_tools.invoke(base_prompt)]}

    graph_builder.add_node("chatbot", chatbot)

    # tool_node = ToolNode(tools=[tool])
    # graph_builder.add_node("tools", tool_node)


    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    
    # graph_builder.add_edge("tools", "chatbot")
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
            if isinstance(last_message, BaseMessage) and last_message.content != user_input:
                print("Agent: " + last_message.content)



if __name__ == "__main__":
    main() 