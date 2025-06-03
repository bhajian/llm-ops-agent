# app/fmp_agent.py
"""
Financial helper that uses an autonomous LangChain agent to interact with our
MCP-FMP FastAPI server. This agent uses an LLM to decide which financial
tools to call based on the user's query.
"""
from __future__ import annotations
from functools import lru_cache

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.tools.fmp_tools import get_fmp_tools
from app.llm import get_llm

@lru_cache
def get_fmp_agent() -> AgentExecutor:
    """
    Creates and returns a cached singleton of the FMP AgentExecutor.
    The agent uses an LLM to reason about which tools to use from the provided list.
    """
    tools = get_fmp_tools()
    
    # Use a low temperature for the LLM to make its tool usage more predictable and factual.
    llm = get_llm(temperature=0) 

    # This is the core prompt that defines the agent's behavior and personality.
    # It instructs the agent on how to reason and use its tools.
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert financial assistant. You must use the available tools to answer the user's questions about stocks and financial data."
            "\nKey Instructions:"
            "\n1. For any question about a specific company, you MUST first use the 'search_symbol' tool to find the correct stock ticker."
            "\n2. Once you have the ticker symbol, use the other tools to find the requested information."
            "\n3. If a user asks a vague question, ask for clarification."
            "\n4. Do not make up information; only use the data provided by the tools."
        )),
        # The MessagesPlaceholder allows the agent to consider chat history.
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        # The agent_scratchpad is where the agent stores its thoughts and tool outputs.
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # This function creates the agent from the LLM, tools, and prompt.
    # It's designed to work with models that support function calling.
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # The AgentExecutor is the runtime that powers the agent. It calls the agent,
    # executes the chosen tools, and loops until an answer is found.
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True, # Set to True for debugging to see the agent's step-by-step thoughts
        return_intermediate_steps=False,
        handle_parsing_errors=True # Makes the agent more robust
    )
    
    return agent_executor