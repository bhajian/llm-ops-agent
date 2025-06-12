# app/agents/fmp_agent.py
"""
Financial helper that uses an autonomous LangChain agent to interact with our
MCP-FMP FastAPI server. This agent uses an LLM to decide which financial
tools to call based on the user's query.
"""
from __future__ import annotations
from functools import lru_cache
# IMPORT CHANGE: Use initialize_agent and AgentType for Bedrock compatibility
from langchain.agents import AgentExecutor, initialize_agent, AgentType, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.tools.fmp_tools import get_fmp_tools
from app.llm import get_llm
from app.tools.datetime_tools import get_current_datetime_tool

@lru_cache
def get_fmp_agent() -> AgentExecutor:
    """
    Creates and returns a cached singleton of the FMP AgentExecutor.
    The agent uses an LLM to reason about which tools to use from the provided list.
    """
    fmp_tools = get_fmp_tools()
    
    # Combine FMP tools with the new datetime tool
    # Convert StructuredTools to basic Tools for initialize_agent if necessary
    # initialize_agent with STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION can often handle StructuredTools directly
    # but sometimes explicit conversion to Tool is safer if tool arguments are simple.
    # We'll pass StructuredTools directly first and see if it works.
    tools = fmp_tools + [get_current_datetime_tool]

    llm = get_llm(temperature=0) # Use the Bedrock LLM

    # With AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    # the prompt is implicitly handled by the agent type's internal logic,
    # but we still define a system message for overall behavior.
    # The agent will inject its own thought/action/observation steps.

    # NOTE: The system prompt here is LESS CRITICAL for tool formatting
    # compared to create_openai_functions_agent, as the agent type
    # generates its own internal prompt for ReAct. However, it still
    # sets the overall persona and high-level rules.
    system_message_content = (
        "You are an expert financial assistant. You must use the available tools to answer the user's questions about stocks and financial data. "
        "Use the tools provided to answer questions."
        "\nKey Instructions:"
        "\n1. For any question about a specific company, you MUST first use the 'search_symbol' tool to find the correct stock ticker if one is not explicitly provided. "
        "\n2. Use the 'get_current_datetime' tool when the user asks for the current date or time, or if a financial query requires knowing the current date (e.g., 'What is XYZ stock price today?')."
        "\n3. Do not make up information; only use the data provided by the tools. If a tool returns no data for a specific request, state that you couldn't retrieve the information."
        "\n4. If a user asks a vague question, ask for clarification (e.g., 'What company are you asking about?')."
        "\n5. Always provide a concise disclaimer about the volatility of stock prices and the importance of professional financial advice, but keep it after presenting the data."
        # The agent type automatically adds observation/thought steps for ReAct
    )

    # initialize_agent is an older way to create agents, but necessary for
    # non-OpenAI function calling models with a general ReAct approach.
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, # This is the key change
        verbose=True, # Keep verbose to see agent's thought process
        handle_parsing_errors=True,
        agent_kwargs={"system_message": system_message_content}, # Pass system message
        # IMPORTANT: max_iterations might be needed to prevent infinite loops on complex tasks
        # max_iterations=5,
        # early_stopping_method="generate"
    )

    return agent_executor
