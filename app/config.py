# app/config.py
import os

def get_settings():
    return {
        "redis_host":    os.getenv("REDIS_HOST", "redis"),
        "weaviate_url":  os.getenv("WEAVIATE_URL", "http://weaviate:8080"),

        # micro-control-planes
        "mcp_fmp_url":   os.getenv("MCP_FMP_URL",  "http://mcp-fmp:9000"),
        "mcp_k8s_url":   os.getenv("MCP_K8S_URL",  "http://mcp-openshift:9000"),
        "mcp_username":  os.getenv("MCP_USERNAME", "admin"),
        "mcp_password":  os.getenv("MCP_PASSWORD", "secret"),

        # LLM
        "llm_base_url":  os.getenv("LLM_BASE_URL", ""),
    }
