import os
from dotenv import load_dotenv
from retriever import guest_info_tool
from langchain_groq import ChatGroq
from typing import TypedDict,Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from tools import search_tool,get_weather_info,extract_text
from langgraph.graph import StateGraph,START
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.messages import HumanMessage


load_dotenv()
api_key = os.environ['GROQ_API_KEY']
llm = ChatGroq(model="llama-3.3-70b-versatile",
               temperature=0,
               api_key=api_key)

tools = [guest_info_tool,get_weather_info,extract_text,search_tool]
llm_with_tools = llm.bind_tools(tools=tools)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage],add_messages]

def assistant(state:AgentState):
    return {'messages': [llm_with_tools.invoke(state['messages'])]}

builder = StateGraph(AgentState) #Empty graph cretaed

builder.add_node('assistant',assistant)
builder.add_node('tools',ToolNode(tools))

builder.add_edge(START,'assistant')
builder.add_conditional_edges('assistant',tools_condition)
builder.add_edge('tools','assistant')

agent = builder.compile()

if __name__ == "__main__":
    request = "I've attached tonight's invitation (invitation.png). Read the guest list from it, remind me how I know each guest, find one recent development in science or tech connected to their legacy so I have conversation starters, and let me know if I'll need an umbrella in London this evening."
    response  = agent.invoke({"messages": [HumanMessage(content=request)]})
    print(f"User's Request:")
    print(f"{request}\n")
    print("Agent's Response:")
    print(response['messages'][-1].content)
