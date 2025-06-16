# app/graph.py
from langgraph.graph import StateGraph, END
from app.langgraph_nodes.planner_node import planner_node
from app.langgraph_nodes.rag_node import rag_node
from app.langgraph_nodes.reasoner_node import reasoner_node
from app.langgraph_nodes.memory_node import memory_node
from app.langgraph_nodes.datetime_node import datetime_node

# Define the graph
builder = StateGraph()

# Register nodes
builder.add_node("planner", planner_node)
builder.add_node("rag", rag_node)
builder.add_node("memory", memory_node)
builder.add_node("datetime", datetime_node)
builder.add_node("reasoner", reasoner_node)

# Dynamic branching based on planner output
def planner_routes(state):
    """Expected state['planner_result'] to be a list of strings like ['rag', 'memory']"""
    return state.get("planner_result", [])

builder.add_conditional_edges("planner", planner_routes, {
    "rag": "rag",
    "memory": "memory",
    "datetime": "datetime",
})

# Edge back to reasoner
builder.add_edge("rag", "reasoner")
builder.add_edge("memory", "reasoner")
builder.add_edge("datetime", "reasoner")

# Start and end
builder.set_entry_point("planner")
builder.set_finish_point("reasoner")

# Compile the app
graph_app = builder.compile()
