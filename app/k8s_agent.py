# --- app/k8s_agent.py ---
from langchain.agents import initialize_agent, Tool, AgentType
from app.llm import get_llm
from app.tools.openshift_tools import scale_deployment, get_pods

def get_k8s_agent():
    tools = [
        Tool(name="ScaleDeployment", func=scale_deployment, description="Scale an OpenShift deployment"),
        Tool(name="GetPods", func=get_pods, description="List pods in a namespace")
    ]
    llm = get_llm()
    return initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
