# llm-ops-agent
project_root/
├── app/
│   ├── main.py                  # FastAPI server with /chat endpoint
│   ├── config.py                # .env config loader
│   ├── llm.py                   # LLM factory (LM Studio / vLLM switch)
│   ├── agent_router.py          # Agent selector (RAG agent vs Ops agent)
│   ├── rag_agent.py             # RAG agent using Weaviate
│   ├── k8s_agent.py             # Kubernetes/Openshift automation agent
│   ├── tools/
│   │   ├── openshift_tools.py   # MCP-integrated OpenShift automation tools
│   │   └── vector_utils.py      # Weaviate doc ingestion/search utilities
├── requirements.txt
├── Dockerfile
├── .env
└── README.md

# --- app/main.py ---
from fastapi import FastAPI, Request
from app.config import get_settings
from app.agent_router import route_query

app = FastAPI()
settings = get_settings()

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    query = body.get("query", "")
    response = await route_query(query)
    return {"response": response}


# --- app/config.py ---
import os
from dotenv import load_dotenv
load_dotenv()

def get_settings():
    return {
        "llm_backend": os.getenv("LLM_BACKEND", "lmstudio"),
        "llm_base_url": os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
        "weaviate_url": os.getenv("WEAVIATE_URL", "http://localhost:8080")
    }


# --- app/llm.py ---
from app.config import get_settings

def get_llm():
    settings = get_settings()
    if settings["llm_backend"] == "lmstudio":
        from langchain_community.chat_models import ChatOpenAI
        return ChatOpenAI(
            base_url=settings["llm_base_url"],
            api_key="none",
            model_name="llama3"
        )
    elif settings["llm_backend"] == "openai":
        from langchain.chat_models import ChatOpenAI
        return ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        raise ValueError("Unsupported LLM backend")


# --- app/rag_agent.py ---
from app.llm import get_llm
from langchain.vectorstores import Weaviate
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import weaviate
from app.config import get_settings

def get_rag_chain():
    settings = get_settings()
    client = weaviate.Client(url=settings["weaviate_url"])
    db = Weaviate(client, "Document", "content", OpenAIEmbeddings())
    retriever = db.as_retriever()
    return RetrievalQA.from_chain_type(llm=get_llm(), retriever=retriever)


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


# --- app/agent_router.py ---
from app.rag_agent import get_rag_chain
from app.k8s_agent import get_k8s_agent

async def route_query(query: str):
    if any(word in query.lower() for word in ["pod", "deployment", "namespace", "openshift", "kubernetes"]):
        agent = get_k8s_agent()
        return await agent.arun(query)
    else:
        rag = get_rag_chain()
        return rag.run(query)


# --- app/tools/openshift_tools.py ---
from kubernetes import client, config

config.load_kube_config()

# Tool 1: scale deployment

def scale_deployment(deployment_name: str, replicas: int, namespace: str = "default"):
    api = client.AppsV1Api()
    body = {"spec": {"replicas": replicas}}
    api.patch_namespaced_deployment_scale(deployment_name, namespace, body)
    return f"Scaled {deployment_name} to {replicas} replicas."

# Tool 2: get pods

def get_pods(namespace: str = "default"):
    api = client.CoreV1Api()
    pods = api.list_namespaced_pod(namespace)
    return ", ".join([pod.metadata.name for pod in pods.items])


# --- requirements.txt ---
fastapi
uvicorn
langchain
openai
weaviate-client
python-dotenv
kubernetes
langchain-community
