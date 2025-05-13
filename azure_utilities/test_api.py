import requests
import json

# Replace with your scoring URI and API key (if applicable)
scoring_uri = "https://ambient-extractor-endpoint.northcentralus.inference.ml.azure.com/score"
api_key = "DFiVIiGC4soOntTKXzTV9Oq3CfvznvfdmU9CwuzmQuZXA2FxMqOSJQQJ99BEAAAAAAAAAAAAINFRAZML27A2"  # Replace with your actual API key

# Example input data to send to the model
input_data = {
    "input": ["Room 1", "10x3", "Women Restroom", "Closet", "Check Plans"]  # Replace with actual data that your model expects
}

# Prepare headers for the request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"  # Use API key or other authentication method if required
}

# Send a POST request to the endpoint
response = requests.post(scoring_uri, headers=headers, json=input_data, timeout=120)

# Check if the response is successful
if response.status_code == 200:
    print("Response from the model:")
    print(response.json())  # The result from the model
else:
    print(f"Error: {response.status_code}")
    print(response.text)
