from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model
from config.read_config import load_config

config = load_config()
azure_cf = config["azure"]
model_cf = config["model"]

ml_client = MLClient(DefaultAzureCredential(), 
                     azure_cf["subscription_id"], 
                     azure_cf["resource_group"], 
                     azure_cf["workspace_name"])

model = Model(path=model_cf["path"],
              name=model_cf["name"],
              description=model_cf["description"],
              type="custom_model")  # or 'onnx' depending on your SDK version

ml_client.models.create_or_update(model)