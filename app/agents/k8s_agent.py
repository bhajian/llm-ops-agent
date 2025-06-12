# --- app/k8s_agent.py ---
# Changed imports: Remove initialize_agent, AgentType. Add AgentExecutor, create_tool_calling_agent
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Added for prompt
from langchain_core.tools import Tool # Ensures Tool is imported correctly
from app.llm import get_llm
from app.tools.openshift_tools import scale_deployment, get_pods

def get_k8s_agent():
    tools = [
        Tool(name="ScaleDeployment", func=scale_deployment, description="Scale an OpenShift deployment"),
        Tool(name="GetPods", func=get_pods, description="List pods in a namespace")
    ]
    llm = get_llm()

    # Define a prompt for the k8s agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for OpenShift Kubernetes cluster management. Use the provided tools to respond to user queries."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create the agent using create_tool_calling_agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Use AgentExecutor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True, # Set to True for debugging
        return_intermediate_steps=False,
        handle_parsing_errors=True
    )
    return agent_executor
