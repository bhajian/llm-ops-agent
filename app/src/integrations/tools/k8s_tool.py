from typing import Any, ClassVar

from kubernetes import client, config
from langchain.tools import BaseTool


class K8sTool(BaseTool):
    name: ClassVar[str] = "k8s"
    description: ClassVar[str] = (
        "Basic Kubernetes/OpenShift ops. "
        "Actions: scale_deployment, list_pods."
    )

    def __init__(self):
        super().__init__()
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        self.apps = client.AppsV1Api()
        self.core = client.CoreV1Api()

    # helpers -------------------------------------------------------
    def _scale_deployment(self, deployment: str, replicas: int, namespace: str = "default"):
        self.apps.patch_namespaced_deployment_scale(
            deployment, namespace, {"spec": {"replicas": replicas}}
        )
        return f"Scaled {deployment} to {replicas} replicas in {namespace}"

    def _list_pods(self, namespace: str = "default"):
        return [p.metadata.name for p in self.core.list_namespaced_pod(namespace).items]

    # BaseTool interface -------------------------------------------
    def _run(self, action: str, **kwargs: Any) -> str:
        if action == "scale_deployment":
            return self._scale_deployment(**kwargs)
        if action == "list_pods":
            return str(self._list_pods(**kwargs))
        raise ValueError("Unsupported action for K8sTool")

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Async not supported for K8sTool.")
