import os
import onnxruntime as rt
import json
from transformers import AutoTokenizer

def init():
    global session
    model_name = "ambient_classifier.onnx"
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), model_name)
    session = rt.InferenceSession(model_path)

def run(data):
    # Extract text data from input
    # Ensure the input is in the correct format (data should be a dictionary)
    try:
        # Check if data is a string, if so, try to load as JSON
        if isinstance(data, str):
            data = json.loads(data)
        
        # Extract the input field, which is expected to be a list of strings
        texts = data["input"]
        
    except Exception as e:
        return json.dumps({"error": f"Failed to parse input data: {str(e)}"})

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize the input text
    encodings = tokenizer(
        texts,
        padding="max_length", 
        truncation=True, 
        max_length=128,
        return_tensors="np" 
    )
    
    # Convert tokenized inputs to numpy arrays
    input_ids = encodings['input_ids']  
    attention_mask = encodings['attention_mask']

    # Prepare the inputs for ONNX Runtime
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    # Run inference with ONNX
    output = session.run(None, inputs)

    # Extract the result (since we have a sigmoid activation, we expect the output to be probabilities)
    result = (output[0] > 0.5).astype(int)

    # Return the result as JSON, converting to a list for easy JSON serialization
    return json.dumps({"result": result.tolist()})
