# app/graph_orchestrator.py
# ────────────────────────────────────────────────────────────
"""
LangGraph based Orchestrator for the multi-agent application.
This defines the state, nodes (LLM calls, tool calls), and edges
for routing user queries to specialized agents and synthesizing responses.
"""

from __future__ import annotations # type: ignore

import asyncio
import json
import re
import warnings
from typing import Dict, List, Literal, Optional, TypedDict, Union, Any

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field, ValidationError

# Local application imports (ADJUSTED PATHS)
from app.llm import get_llm
from app.chat_agent import chat_with_memory # Corrected import path for chat_agent
from app.tools.fmp_tools import get_fmp_tools # Tool remains in tools/
from app.tools.datetime_tool import get_current_datetime_tool # Corrected import path for datetime_tool
from app.agents.rag_agent import get_weaviate_retriever, condense_question_for_rag, answer_question_from_context # RAG agent in agents/
from app.agents.finance_agent import extract_ticker_symbol, synthesize_finance_answer # Finance agent in agents/
from app.agents.k8s_agent import extract_k8s_tool_call, synthesize_k8s_answer # NEW: K8s agent functions

from app.tools.openshift_tools import scale_deployment, get_pods # NEW: Direct import of K8s tools for invocation

from langgraph.graph import END, StateGraph


# ───────────────── 1. Define Graph State ────────────────────────────────
# The state for the graph, which will be passed between nodes.
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        user_query: The current user's input.
        chat_history: A list of previous user/AI messages.
        intent: The identified intent of the user's query (e.g., "CHAT", "FINANCE", "COT_RAG", "K8S").
        ticker_symbol: Extracted stock ticker symbol, if applicable.
        finance_tool_output: Output from the financial tool call.
        rag_condensed_question: Standalone question for RAG retrieval.
        rag_context: Retrieved documents for RAG.
        k8s_tool_call: Pydantic model containing tool name and args for K8s.
        k8s_tool_output: Output from the K8s tool call.
        final_answer: The final answer to be presented to the user.
        error_message: Any error message encountered during processing.
    """
    user_query: str
    chat_history: List[Dict[str, str]]
    intent: Optional[str]
    ticker_symbol: Optional[str]
    finance_tool_output: Optional[str]
    rag_condensed_question: Optional[str]
    rag_context: Optional[str]
    k8s_tool_call: Optional[Dict[str, Any]] # Store the parsed K8s tool call
    k8s_tool_output: Optional[str]
    final_answer: Optional[str]
    error_message: Optional[str]


# ───────────────── 2. LLM and Parser Instances (shared) ────────────────
# Use consistent LLMs for routing, argument extraction, and synthesis
_orchestrator_llm = get_llm(temperature=0.0) # Very low temp for deterministic routing/parsing


# ───────────────── 3. Graph Nodes (Functions) ──────────────────────────

async def call_router_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 1: Classifies the intent of the user's query.
    This is an LLM-powered router with fallback rules.
    """
    print("---NODE: CALLING ROUTER---")
    user_query = state["user_query"]
    chat_history = state["chat_history"]
    normalized_user_query = user_query.lower().strip()

    # Rule-Based Routing (Prioritized for speed and reliability)
    # Rule 1: RAG queries
    rag_keywords = ["who is", "what is", "tell me about", "information about",
                    "summarize", "describe", "explain", "documents", "document", "facts"]
    if any(keyword in normalized_user_query for keyword in rag_keywords):
        print(f"🧭 Rule-based router chose: COT_RAG for query: '{user_query}'")
        return {"intent": "COT_RAG"}

    # Rule 2: Finance queries
    finance_keywords = ["stock", "price", "company value", "market cap", "earnings",
                        "dividend", "financials", "analyst", "target", "eps",
                        "date today", "current time", "what time is it", "what date is it"]
    if any(keyword in normalized_user_query for keyword in finance_keywords):
        print(f"🧭 Rule-based router chose: FINANCE for query: '{user_query}'")
        return {"intent": "FINANCE"}

    # Rule 3: K8s queries
    k8s_keywords = ["scale deployment", "get pods", "openshift", "kubernetes", "k8s", "deployments", "pods"]
    if any(keyword in normalized_user_query for keyword in k8s_keywords):
        print(f"🧭 Rule-based router chose: K8S for query: '{user_query}'")
        return {"intent": "K8S"}


    # Fallback to LLM Router for General CHAT
    _ROUTER_PROMPT = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are an expert at classifying user intent. "
            "Your sole task is to determine if the user's query is purely conversational. "
            "If it is a greeting, a general question (e.g., 'how are you?'), or asking for factual information you don't have tools for (e.g., 'What is the capital of France?', 'What is the weather like?'), then and ONLY then, output 'CHAT'. "
            "For all other types of queries (e.g., financial, document lookup, K8s commands), you are NOT to output 'CHAT'. "
            "Your response MUST be **EXACTLY** the word 'CHAT' and nothing else. NO examples, NO explanations, NO punctuation, NO other words."
            "\n\nStrict examples:"
            "\nUser: 'Hi'"
            "\nAssistant: CHAT"
            "\nUser: 'How are you today?'"
            "\nAssistant: CHAT"
            "\nUser: 'Tell me a joke.'"
            "\nAssistant: CHAT"
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}"),
    ])

    messages_for_router = _ROUTER_PROMPT.format_messages(
        input=user_query,
        chat_history=[HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in chat_history]
    )

    try:
        llm_response = await asyncio.to_thread(_orchestrator_llm.invoke, messages_for_router)
        raw_content = llm_response.content.strip().upper() if hasattr(llm_response, 'content') else str(llm_response).strip().upper()

        if raw_content == "CHAT":
            print(f"🧭 LLM router chose: CHAT (from raw response: '{raw_content}') for query: '{user_query}'")
            return {"intent": "CHAT"}
        else:
            warnings.warn(
                f"LLM router produced unexpected keyword: '{raw_content}'. "
                f"Falling back to CHAT for query: '{user_query}'."
            )
            return {"intent": "CHAT"}
    except Exception as e:
        warnings.warn(f"Router LLM (fallback) failed for query '{user_query}': {e}. Falling back to CHAT.")
        return {"intent": "CHAT"}


async def call_chat_agent_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 2: Handles general conversational queries.
    """
    print("---NODE: CALLING CHAT AGENT---")
    user_query = state["user_query"]
    chat_history = state["chat_history"]
    try:
        answer = chat_with_memory(user_query, chat_history)
        return {"final_answer": answer}
    except Exception as e:
        warnings.warn(f"Chat agent failed: {e}")
        return {"final_answer": "I'm sorry, I couldn't process your chat request.", "error_message": str(e)}


async def extract_ticker_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 3 (Finance Path): Extracts the stock ticker symbol from the query.
    Uses the utility function from app.agents.finance_agent.
    """
    print("---NODE: EXTRACTING TICKER---")
    user_query = state["user_query"]
    try:
        ticker = await extract_ticker_symbol(user_query)
        return {"ticker_symbol": ticker}
    except Exception as e:
        warnings.warn(f"Ticker extraction failed: {e}")
        return {"ticker_symbol": None, "error_message": f"Failed to extract ticker: {e}"}


async def call_finance_tool_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 4 (Finance Path): Calls the appropriate FMP tool based on the query.
    """
    print("---NODE: CALLING FINANCE TOOL---")
    user_query = state["user_query"]
    ticker = state.get("ticker_symbol")
    normalized_user_query = user_query.lower().strip()

    if "date today" in normalized_user_query or "current time" in normalized_user_query or \
       "what time is it" in normalized_user_query or "what date is it" in normalized_user_query:
        tool_output = await asyncio.to_thread(get_current_datetime_tool.func)
        return {"finance_tool_output": tool_output}

    if not ticker:
        return {"finance_tool_output": None, "error_message": "No ticker symbol provided for financial query."}

    tool_func = None
    tool_name = ""
    fmp_tools = get_fmp_tools()
    fmp_tool_map = {t.name: t.func for t in fmp_tools}

    if "price" in normalized_user_query or "stock quote" in normalized_user_query or "eps" in normalized_user_query:
        tool_func = fmp_tool_map.get("get_stock_quote")
        tool_name = "get_stock_quote"
    elif "analyst estimate" in normalized_user_query:
        tool_func = fmp_tool_map.get("get_analyst_estimate")
        tool_name = "get_analyst_estimate"
    elif "price target" in normalized_user_query:
        tool_func = fmp_tool_map.get("get_price_target")
        tool_name = "get_price_target"
    elif "historical prices" in normalized_user_query:
        tool_func = fmp_tool_map.get("get_historical_prices")
        tool_name = "get_historical_prices"
    elif "grades summary" in normalized_user_query:
        tool_func = fmp_tool_map.get("get_grades_summary")
        tool_name = "get_grades_summary"
    elif "corporate calendar" in normalized_user_query:
        tool_func = fmp_tool_map.get("get_corporate_calendar")
        tool_name = "get_corporate_calendar"
    elif "dividend calendar" in normalized_user_query:
        tool_func = fmp_tool_map.get("get_dividend_calendar")
        tool_name = "get_dividend_calendar"
    elif "earnings surprises" in normalized_user_query:
        tool_func = fmp_tool_map.get("get_earnings_surprises")
        tool_name = "get_earnings_surprises"
    elif "enterprise valuation" in normalized_user_query:
        tool_func = fmp_tool_map.get("get_enterprise_valuation")
        tool_name = "get_enterprise_valuation"
    else:
        return {"finance_tool_output": None, "error_message": "Could not determine specific financial data type from query."}

    if not tool_func:
        return {"finance_tool_output": None, "error_message": f"No tool found for '{tool_name}' or type of data requested."}

    try:
        if tool_name == "get_historical_prices":
            tool_output = await asyncio.to_thread(tool_func, symbol=ticker, series_type="line", from_date="2024-01-01")
        else:
            tool_output = await asyncio.to_thread(tool_func, symbol=ticker)
        print(f"📊 Tool '{tool_name}' output: {tool_output}")
        return {"finance_tool_output": tool_output}
    except Exception as e:
        warnings.warn(f"Error calling FMP tool '{tool_name}' for {ticker}: {e}")
        return {"finance_tool_output": None, "error_message": f"Error retrieving {tool_name} for {ticker}: {e}"}


async def finance_synthesis_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 5 (Finance Path): Synthesizes the final human-readable answer for finance queries.
    Uses the utility function from app.agents.finance_agent.
    """
    print("---NODE: FINANCE SYNTHESIS---")
    user_query = state["user_query"]
    tool_output = state.get("finance_tool_output")
    error_message = state.get("error_message")

    if error_message:
        return {"final_answer": f"I'm sorry, I encountered an error: {error_message}. Please try again."}
    if not tool_output:
        return {"final_answer": "I'm sorry, I couldn't retrieve the requested financial data."}

    normalized_user_query = user_query.lower().strip()
    if "date today" in normalized_user_query or "current time" in normalized_user_query or \
       "what time is it" in normalized_user_query or "what date is it" in normalized_user_query:
        return {"final_answer": tool_output}

    try:
        answer = await synthesize_finance_answer(user_query, tool_output)
        return {"final_answer": answer}
    except Exception as e:
        warnings.warn(f"Finance synthesis failed: {e}")
        return {"final_answer": "I'm sorry, I couldn't synthesize the financial data into a coherent answer.", "error_message": str(e)}


async def rag_condense_question_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 6 (RAG Path): Condenses chat history and user query into a standalone question for RAG.
    Uses the function from app.agents.rag_agent.
    """
    print("---NODE: RAG CONDENSE QUESTION---")
    user_query = state["user_query"]
    chat_history = state["chat_history"]
    try:
        # NOTE: rag_agent's condense_question_for_rag expects raw chat_history from GraphState
        condensed_q = await condense_question_for_rag(user_query, chat_history) 
        print(f"🔍 RAG: Condensed question: '{condensed_q}'")
        return {"rag_condensed_question": condensed_q}
    except Exception as e:
        warnings.warn(f"RAG question condensation failed: {e}")
        return {"rag_condensed_question": None, "error_message": f"Failed to condense question for RAG: {e}"}


async def rag_retrieve_documents_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 7 (RAG Path): Retrieves relevant documents from Weaviate.
    Uses the retriever from app.agents.rag_agent.
    """
    print("---NODE: RAG RETRIEVE DOCUMENTS---")
    condensed_question = state["rag_condensed_question"]
    if not condensed_question:
        return {"rag_context": None, "error_message": "No condensed question for retrieval."}
    
    try:
        retriever = get_weaviate_retriever()
        retrieved_docs = await asyncio.to_thread(retriever.get_relevant_documents, condensed_question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        print(f"🔍 RAG: Retrieved {len(retrieved_docs)} documents.")
        return {"rag_context": context}
    except Exception as e:
        warnings.warn(f"RAG document retrieval failed: {e}")
        return {"rag_context": None, "error_message": f"Failed to retrieve documents: {e}"}


async def rag_answer_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 8 (RAG Path): Synthesizes the final answer based on retrieved context.
    Uses the function from app.agents.rag_agent.
    """
    print("---NODE: RAG ANSWER SYNTHESIS---")
    user_query = state["user_query"]
    rag_context = state.get("rag_context")
    error_message = state.get("error_message")

    if error_message:
        return {"final_answer": f"I'm sorry, I encountered an error during RAG: {error_message}. Please try again."}
    if not rag_context:
        return {"final_answer": "I could not find relevant information in the documents to answer your question."}

    try:
        answer = await answer_question_from_context(user_query, rag_context)
        return {"final_answer": answer}
    except Exception as e:
        warnings.warn(f"RAG answer synthesis failed: {e}")
        return {"final_answer": "I'm sorry, I couldn't synthesize the answer from the documents.", "error_message": str(e)}


# --- K8S Nodes ---
async def extract_k8s_args_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 9 (K8s Path): Extracts the K8s tool to call and its arguments.
    Uses the utility function from app.agents.k8s_agent.
    """
    print("---NODE: EXTRACTING K8S ARGS---")
    user_query = state["user_query"]
    try:
        k8s_tool_call = await extract_k8s_tool_call(user_query)
        return {"k8s_tool_call": k8s_tool_call.dict() if k8s_tool_call else None}
    except Exception as e:
        warnings.warn(f"K8s argument extraction failed: {e}")
        return {"k8s_tool_call": None, "error_message": f"Failed to extract K8s args: {e}"}


async def call_k8s_tool_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 10 (K8s Path): Calls the identified OpenShift/K8s tool.
    """
    print("---NODE: CALLING K8S TOOL---")
    k8s_tool_info = state.get("k8s_tool_call")
    user_query = state["user_query"]
    
    if not k8s_tool_info or not k8s_tool_info.get("tool_name"):
        return {"k8s_tool_output": None, "error_message": "No valid K8s tool call determined."}

    tool_name = k8s_tool_info["tool_name"]
    tool_args = k8s_tool_info.get("tool_args", {})

    tool_func = None
    if tool_name == "scale_deployment":
        tool_func = scale_deployment
    elif tool_name == "get_pods":
        tool_func = get_pods
    else:
        return {"k8s_tool_output": None, "error_message": f"Unknown K8s tool: {tool_name}"}

    try:
        tool_output = await asyncio.to_thread(tool_func, **tool_args)
        print(f"🛠️ K8s Tool '{tool_name}' output: {tool_output}")
        return {"k8s_tool_output": tool_output}
    except Exception as e:
        warnings.warn(f"Error calling K8s tool '{tool_name}': {e}")
        return {"k8s_tool_output": None, "error_message": f"Error executing K8s tool '{tool_name}': {e}"}


async def k8s_synthesis_node(state: GraphState) -> Dict[str, Any]:
    """
    Node 11 (K8s Path): Synthesizes the final human-readable answer for K8s queries.
    Uses the utility function from app.agents.k8s_agent.
    """
    print("---NODE: K8S SYNTHESIS---")
    user_query = state["user_query"]
    k8s_tool_output = state.get("k8s_tool_output")
    error_message = state.get("error_message")
    tool_name = state.get("k8s_tool_call", {}).get("tool_name", "unknown_k8s_tool")

    if error_message:
        return {"final_answer": f"I'm sorry, I encountered an error: {error_message}. Please try again."}
    if k8s_tool_output is None:
        return {"final_answer": "I'm sorry, I couldn't get a result from the OpenShift tool."}

    try:
        answer = await synthesize_k8s_answer(user_query, tool_name, k8s_tool_output)
        return {"final_answer": answer}
    except Exception as e:
        warnings.warn(f"K8s synthesis failed: {e}")
        return {"final_answer": "I'm sorry, I couldn't synthesize the K8s operation result.", "error_message": str(e)}


# ───────────────── 4. Conditional Edges (Functions) ──────────────────────

# Store node functions in variables to reference them in routes
_NODE_CHAT = call_chat_agent_node
_NODE_FINANCE_EXTRACT_TICKER = extract_ticker_node
_NODE_FINANCE_CALL_TOOL = call_finance_tool_node
_NODE_FINANCE_SYNTHESIS = finance_synthesis_node
_NODE_RAG_CONDENSE = rag_condense_question_node
_NODE_RAG_RETRIEVE = rag_retrieve_documents_node
_NODE_RAG_ANSWER = rag_answer_node
_NODE_K8S_EXTRACT_ARGS = extract_k8s_args_node
_NODE_K8S_CALL_TOOL = call_k8s_tool_node
_NODE_K8S_SYNTHESIS = k8s_synthesis_node


def route_by_intent(state: GraphState) -> str:
    """
    Routes based on the intent identified by the router node.
    Returns the string name of the next node.
    """
    intent = state.get("intent")
    print(f"---ROUTING: Intent '{intent}'---")
    if intent == "FINANCE":
        return "extract_ticker_node"
    elif intent == "COT_RAG":
        return "rag_condense_question_node"
    elif intent == "K8S":
        return "extract_k8s_args_node"
    elif intent == "CHAT":
        return "call_chat_agent_node"
    else:
        return "call_chat_agent_node"


def route_finance_subtask(state: GraphState) -> str:
    """
    Routes within the finance path. Returns the string name of the next node.
    """
    user_query = state["user_query"].lower().strip()
    ticker = state.get("ticker_symbol")
    error_message = state.get("error_message")

    if "date today" in user_query or "current time" in user_query or \
       "what time is it" in user_query or "what date is it" in user_query:
        print("---ROUTING: Finance is direct datetime---")
        return "call_finance_tool_node"

    if ticker and not error_message:
        print(f"---ROUTING: Finance with ticker '{ticker}'---")
        return "call_finance_tool_node"
    else:
        print("---ROUTING: Finance failed ticker extraction, synthesizing error---")
        state["finance_tool_output"] = None
        state["error_message"] = state.get("error_message", "Could not identify a valid stock ticker.")
        return "finance_synthesis_node"


def route_rag_retrieval_outcome(state: GraphState) -> str:
    """
    Routes within the RAG path. Returns the string name of the next node or END.
    """
    rag_context = state.get("rag_context")
    error_message = state.get("error_message")

    if rag_context and not error_message:
        print("---ROUTING: RAG documents retrieved---")
        return "rag_answer_node"
    else:
        print("---ROUTING: RAG no documents or error, synthesizing error---")
        state["final_answer"] = state.get("error_message", "I could not find relevant information in the documents.")
        return END


def route_k8s_tool_outcome(state: GraphState) -> str:
    """
    Routes within the K8s path. Returns the string name of the next node.
    """
    k8s_tool_info = state.get("k8s_tool_call")
    error_message = state.get("error_message")

    if k8s_tool_info and not error_message:
        print(f"---ROUTING: K8s tool '{k8s_tool_info.get('tool_name')}' identified---")
        return "call_k8s_tool_node"
    else:
        print("---ROUTING: K8s failed tool extraction or error, synthesizing error---")
        state["k8s_tool_output"] = None
        state["error_message"] = state.get("error_message", "I could not determine the specific OpenShift operation or its parameters from your request.")
        return "k8s_synthesis_node"


# ───────────────── 5. Build the Graph ──────────────────────────────────
workflow = StateGraph(GraphState)

# Add nodes to the graph using the explicitly named variables
workflow.add_node("router_node", call_router_node)
workflow.add_node("call_chat_agent_node", _NODE_CHAT) # Using the named variable
workflow.add_node("extract_ticker_node", _NODE_FINANCE_EXTRACT_TICKER)
workflow.add_node("call_finance_tool_node", _NODE_FINANCE_CALL_TOOL)
workflow.add_node("finance_synthesis_node", _NODE_FINANCE_SYNTHESIS)
workflow.add_node("rag_condense_question_node", _NODE_RAG_CONDENSE)
workflow.add_node("rag_retrieve_documents_node", _NODE_RAG_RETRIEVE)
workflow.add_node("rag_answer_node", _NODE_RAG_ANSWER)
workflow.add_node("extract_k8s_args_node", _NODE_K8S_EXTRACT_ARGS)
workflow.add_node("call_k8s_tool_node", _NODE_K8S_CALL_TOOL)
workflow.add_node("k8s_synthesis_node", _NODE_K8S_SYNTHESIS)


# Set the entry point
workflow.set_entry_point("router_node")

# Add conditional edges from the router - mapping string names to node names (as returned by route_by_intent)
workflow.add_conditional_edges(
    "router_node",
    route_by_intent, # This function still returns string names
    {
        "FINANCE": "extract_ticker_node",
        "COT_RAG": "rag_condense_question_node",
        "K8S": "extract_k8s_args_node",
        "CHAT": "call_chat_agent_node",
    }
)

# Finance path routing
workflow.add_conditional_edges(
    "extract_ticker_node",
    route_finance_subtask, # This function returns string names
    {
        "call_finance_tool_node": "call_finance_tool_node",
        "finance_synthesis_node": "finance_synthesis_node",
    }
)
workflow.add_edge("call_finance_tool_node", "finance_synthesis_node")


# RAG path routing
workflow.add_edge("rag_condense_question_node", "rag_retrieve_documents_node")
workflow.add_conditional_edges(
    "rag_retrieve_documents_node",
    route_rag_retrieval_outcome, # This function returns string names or END
    {
        "rag_answer_node": "rag_answer_node",
        END: END
    }
)
workflow.add_edge("rag_answer_node", END)


# K8s path routing
workflow.add_conditional_edges(
    "extract_k8s_args_node",
    route_k8s_tool_outcome, # This function returns string names
    {
        "call_k8s_tool_node": "call_k8s_tool_node",
        "k8s_synthesis_node": "k8s_synthesis_node",
    }
)
workflow.add_edge("call_k8s_tool_node", "k8s_synthesis_node")


# All synthesis nodes and the chat agent node lead to END
workflow.add_edge("finance_synthesis_node", END)
workflow.add_edge("k8s_synthesis_node", END)
workflow.add_edge("call_chat_agent_node", END)


# Compile the graph into an executable runnable
app = workflow.compile()
