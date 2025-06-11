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
from app.tools.fmp_tools import get_fmp_tools # Ensure this provides tools with good descriptions
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
    # It instructs the agent on how to reason and use its tools, including output formatting.
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert financial assistant. You must use the available tools to answer the user's questions about stocks and financial data."
            "\nKey Instructions:"
            "\n1. For any question about a specific company, you MUST first use the 'search_symbol' tool to find the correct stock ticker."
            "\n   When using 'search_symbol', you will receive a list of matching companies. Your top priority for selection should be:"
            "\n   - **Primary**: A symbol with 'currency' as 'USD' AND 'exchange' as 'NASDAQ'."
            "\n   - **Secondary**: If NASDAQ is not found, look for other major US exchanges with 'currency' as 'USD', such as 'NYSE', 'CBOE', or 'AMEX'."
            "\n   - **Tertiary**: If no major US exchanges are found, pick the first most relevant or commonly traded symbol in 'USD' currency. Avoid derivative products (like 'ETC', 'ETF', 'Yield Shares') unless specifically asked."
            "\n   - If 'search_symbol' returns no results or no suitable US exchange, state that you couldn't find a relevant stock symbol."
            "\n2. Once you have the most relevant ticker symbol, use the other tools (e.g., 'get_stock_quote' for price, 'get_earnings_surprises' for earnings, etc.) to find the requested information."
            "\n3. When presenting financial data (like stock quotes, earnings, etc.), always present the actual numerical values clearly and concisely from the tool output. Do NOT use placeholders like '[current price]' or '[number of shares traded]'. Extract the precise data points and format them clearly."
            "\n4. If a user asks a vague question, ask for clarification (e.g., 'What company are you asking about?')."
            "\n5. Do not make up information; only use the data provided by the tools. If a tool returns no data for a specific request, state that you couldn't retrieve the information."
            "\n6. Always provide a concise disclaimer about the volatility of stock prices and the importance of professional financial advice, but keep it after presenting the data."
        )),
        # The MessagesPlaceholder allows the agent to consider chat history.
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        # The agent_scratchpad is where the agent stores its thoughts and tool outputs.
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # This creates the agent, binding the LLM, tools, and prompt together.
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
