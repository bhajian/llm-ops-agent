from kubernetes import client, config
import os

def get_k8s_client():
    configuration = client.Configuration()
    configuration.host = os.getenv("OCP_API")
    configuration.verify_ssl = False
    configuration.api_key = {"authorization": "Bearer " + os.getenv("OCP_TOKEN")}
    return configuration

def scale_deployment(namespace, name, replicas):
    configuration = get_k8s_client()
    with client.ApiClient(configuration) as api:
        api_instance = client.AppsV1Api(api)
        body = {"spec": {"replicas": replicas}}
        api_instance.patch_namespaced_deployment_scale(name=name, namespace=namespace, body=body)

def list_pods(namespace):
    configuration = get_k8s_client()
    with client.ApiClient(configuration) as api:
        v1 = client.CoreV1Api(api)
        return [pod.metadata.name for pod in v1.list_namespaced_pod(namespace).items]
