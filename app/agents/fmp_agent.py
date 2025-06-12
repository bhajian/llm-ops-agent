# app/agents/fmp_agent.py
"""
Financial helper that uses an autonomous LangChain agent to interact with our
MCP-FMP FastAPI server. This agent uses an LLM to decide which financial
tools to call based on the user's query.
"""
from __future__ import annotations
from functools import lru_cache # Keep import for consistency, but not used on this function
# Revert imports: Use initialize_agent and AgentType again
from langchain.agents import AgentExecutor, initialize_agent, AgentType, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Keep for clarity, but less critical here
from app.tools.fmp_tools import get_fmp_tools
from app.llm import get_llm
from app.tools.datetime_tools import get_current_datetime_tool

# REMOVED @lru_cache here to ensure a fresh agent is always created
def get_fmp_agent() -> AgentExecutor:
    """
    Creates and returns an AgentExecutor using initialize_agent with ReAct.
    The agent uses an LLM to reason about which tools to call based on the user's query.
    """
    fmp_tools = get_fmp_tools()
    
    # Combine FMP tools with the new datetime tool
    # initialize_agent handles StructuredTools well, so no explicit Tool wrapping needed.
    tools = fmp_tools + [get_current_datetime_tool]

    # Pass streaming=True to the get_llm() call here
    llm = get_llm(temperature=0, streaming=True) # Ensure LLM supports streaming for async agent operations

    # Define the system message content directly
    system_message_content = (
        "You are an expert financial assistant. Use the provided tools to answer questions about stock prices, company financials, and market data. "
        "You must answer questions by using the available tools and providing factual data. "
        "Strictly adhere to the following rules for tool usage and response generation:"
        "\n1. Always attempt to use a tool if the query is financial or asks for date/time. Do not answer from general knowledge if a tool can provide the information."
        "\n2. For stock-related queries, always try to use the 'search_symbol' tool first to resolve company names to ticker symbols if the ticker is not explicitly provided. "
        "Then, use the appropriate FMP tool (e.g., 'get_stock_quote') with the resolved ticker."
        "\n3. Use the 'get_current_datetime' tool when the user asks for the current date or time, or if a financial query requires knowing the current date (e.e.g., 'What is XYZ stock price today?')."
        "\n4. Do not make up information; only use the data provided by the tools. If a tool returns no data for a specific request, state that you couldn't retrieve the information."
        "\n5. If a user asks a vague question, ask for clarification (e.g., 'What company are you asking about?')."
        "\n6. Always provide a concise disclaimer about the volatility of stock prices and the importance of professional financial advice, but keep it after presenting the data. Do NOT include this disclaimer if the query was for datetime only."
        # The agent type automatically adds observation/thought steps for ReAct
    )

    # Use initialize_agent with AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
    # This type does not require bind_tools, but uses a ReAct-style prompt.
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, # Reverted to this agent type
        verbose=True, # Keep verbose to see agent's thought process
        handle_parsing_errors=True,
        agent_kwargs={"system_message": system_message_content}, # Pass system message as agent_kwargs
        # llm_kwargs={"stop_sequences": []}, # Not needed for initialize_agent, as LLM handles this via get_llm config
        # IMPORTANT: max_iterations might be needed to prevent infinite loops on complex tasks
        # max_iterations=5,
        # early_stopping_method="generate"
    )

    return agent_executor
