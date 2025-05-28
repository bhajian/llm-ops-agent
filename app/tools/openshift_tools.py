import os
import requests

MCP_BASE_URL = os.getenv("MCP_BASE_URL", "http://mcp-server:9000")
MCP_AUTH = (os.getenv("MCP_USERNAME", "admin"), os.getenv("MCP_PASSWORD", "secret"))

def scale_deployment(deployment_name: str, replicas: int, namespace: str = "default") -> str:
    url = f"{MCP_BASE_URL}/openshift/scale"
    payload = {
        "deployment": deployment_name,
        "namespace": namespace,
        "replicas": replicas
    }
    try:
        res = requests.post(url, json=payload, auth=MCP_AUTH)
        res.raise_for_status()
        return f"✅ MCP scaled '{deployment_name}' to {replicas} in namespace '{namespace}'"
    except requests.RequestException as e:
        return f"❌ Failed to scale via MCP: {e}"

def get_pods(namespace: str = "default") -> str:
    url = f"{MCP_BASE_URL}/openshift/pods?namespace={namespace}"
    try:
        res = requests.get(url, auth=MCP_AUTH)
        res.raise_for_status()
        pods = res.json().get("pods", [])
        if not pods:
            return "No pods found."
        return f"Pods in '{namespace}':\n- " + "\n- ".join(pods)
    except requests.RequestException as e:
        return f"❌ Failed to fetch pods from MCP: {e}"
