from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model, Environment, ManagedOnlineEndpoint, ManagedOnlineDeployment, CodeConfiguration
from config.read_config import load_config


def create_configs():
    config = load_config()

    azure_cf = config["azure"]
    model_cf = config["model"]
    env_cf = config["environment"]
    api_cf = config["endpoint"]

    return azure_cf, model_cf, env_cf, api_cf

def create_client(azure_cf):
    ml_client = MLClient(DefaultAzureCredential(), 
                        azure_cf["subscription_id"], 
                        azure_cf["resource_group"], 
                        azure_cf["workspace_name"])
    return ml_client

def create_env(env_cf):
    env = Environment(
        name=env_cf["name"],
        image=env_cf["image"],
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

def deploy_model(ml_client, model, env, api_cf):
    endpoint = ManagedOnlineEndpoint(
        name=api_cf["name"],
        auth_mode=api_cf["auth"]
    )
    
    deployment = ManagedOnlineDeployment(
        name=api_cf["deploy_name"],
        endpoint_name=endpoint.name,
        model=model,
        environment=env,
        code_configuration=CodeConfiguration(code=".", scoring_script="src/score.py"),
        instance_type=api_cf["instance_type"],
        instance_count=1
    )

    ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
    ml_client.online_deployments.begin_create_or_update(deployment).wait()

    endpoint = ml_client.online_endpoints.get(name=api_cf["name"])  # Fetch the existing endpoint

    # Set traffic to the deployment
    endpoint.traffic = {
        api_cf["deploy_name"]: 100  # 100% of the traffic is routed to this deployment
    }

    # Update the endpoint with the new traffic configuration
    ml_client.online_endpoints.begin_create_or_update(endpoint).wait()



def register_and_deploy_model():
    azure_cf, model_cf, env_cf, api_cf = create_configs()
    print("Creating ML Client")
    ml_client = create_client(azure_cf)
    print("Creating Conda Environment")
    env = create_env(env_cf)
    print("Registering Model")
    model = register_model(ml_client, model_cf)
    print("Deploying Model")
    deploy_model(ml_client, model, env, api_cf)
    print("Model Deployed")

if __name__ == "__main__":
    register_and_deploy_model()