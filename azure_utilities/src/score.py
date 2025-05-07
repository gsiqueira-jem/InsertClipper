import os
import onnxruntime as rt
import numpy as np
import json

def init():
    global session
    model_name = "ambient_classifier.onnx"
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), model_name)

    session = rt.InferenceSession("model/model.onnx")

def run(data):
    inputs = np.array(data["input"], dtype=np.float32)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: inputs})
    return json.dumps({"result": outputs[0].tolist()})