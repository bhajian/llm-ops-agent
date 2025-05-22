import os
from langchain.agents import initialize_agent, AgentType
from langchain.agents.agent_toolkits import Tool
from langchain.chat_models import ChatOpenAI
from app.tools.openshift_tools import scale_deployment
from app.tools.aap_tools import run_playbook

def create_agent_executor():
    tools = [
        Tool(name="ScaleDeployment", func=scale_deployment, description="Scale an OpenShift deployment"),
        Tool(name="RunPlaybook", func=run_playbook, description="Trigger an AAP playbook"),
    ]

    llm = ChatOpenAI(
        base_url=os.getenv("VLLM_BASE_URL"),
        api_key="fake",  # vLLM doesn't require real keys
        model_name="llama3-70b-chat"
    )
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
    return agent
