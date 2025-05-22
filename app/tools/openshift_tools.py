from kubernetes import client, config
def scale_deployment(deployment_name: str, replicas: int, namespace="default"):
    config.load_kube_config()
    api = client.AppsV1Api()
    body = {"spec": {"replicas": replicas}}
    api.patch_namespaced_deployment_scale(deployment_name, namespace, body)
    return f"Scaled {deployment_name} to {replicas} replicas."
