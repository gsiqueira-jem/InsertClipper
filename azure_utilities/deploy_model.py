from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.ai.ml.entities._deployment.resource_requirements import ResourceRequirements

endpoint = ManagedOnlineEndpoint(
    name="onnx-endpoint",
    auth_mode="key"
)

deployment = ManagedOnlineDeployment(
    name="onnx-deploy",
    endpoint_name=endpoint.name,
    model=model,
    environment=env,
    code_configuration={"code": ".", "scoring_script": "score.py"},
    instance_type="Standard_DS2_v2",
    instance_count=1
)

ml_client.online_endpoints.begin_create_or_update(endpoint)
ml_client.online_deployments.begin_create_or_update(deployment)

# Set default deployment
ml_client.online_endpoints.begin_update(
    ManagedOnlineEndpoint(name="onnx-endpoint", defaults={"deployment_name": "onnx-deploy"})
)
