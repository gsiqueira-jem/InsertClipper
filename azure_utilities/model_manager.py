from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model, Environment, ManagedOnlineEndpoint, ManagedOnlineDeployment
from config.read_config import load_config


def create_configs():
    config = load_config()

    azure_cf = config["azure"]
    model_cf = config["model"]
    env_cf = config["environment"]

    return azure_cf, model_cf, env_cf

def create_client(azure_cf):
    ml_client = MLClient(DefaultAzureCredential(), 
                        azure_cf["subscription_id"], 
                        azure_cf["resource_group"], 
                        azure_cf["workspace_name"])
    return ml_client

def create_env(env_cf):
    env = Environment(
        name=env_cf["name"],
        conda_file=env_cf["conda_file"]
    )
    return env

def register_model(ml_client, model_cf):
    model = Model(
        path=model_cf["path"],
        name=model_cf["name"],
        description=model_cf["description"],
        type="custom_model"
    )

    ml_client.models.create_or_update(model)
    return model

def deploy_model(ml_client, model, env):
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

    ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
    ml_client.online_deployments.begin_create_or_update(deployment).wait()

    # Set default deployment
    ml_client.online_endpoints.begin_update(
        ManagedOnlineEndpoint(name="onnx-endpoint", defaults={"deployment_name": "onnx-deploy"})
    ).wait()

def register_and_deploy_model():
    azure_cf, model_cf, env_cf = create_configs()
    ml_client = create_client(azure_cf)
    env = create_env(env_cf)
    model = register_model(ml_client, model_cf)
    deploy_model(ml_client, model, env)

if __name__ == "__main__":
    register_and_deploy_model()