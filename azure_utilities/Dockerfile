# Use the Azure ML base image for inference
FROM mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest

# Install necessary dependencies including PyTorch
RUN pip install --upgrade pip && \
    pip install onnxruntime numpy transformers torch

# Set the working directory to the Azure ML source directory
WORKDIR /var/azureml-app/azure_utilities/src

# Copy the scoring script into the container
COPY src/score.py .

# Expose the port used by Azure ML Inference Server (typically 5001)
EXPOSE 5001

# Entry point for the container: Start the Azure ML Inference Server
CMD ["python", "-m", "azureml.core.webservice.score"]