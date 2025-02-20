# Start from a Python 3.11 image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the Python app script and other necessary files into the working directory
COPY app.py /app/
COPY requirements.txt /app/
COPY kokoro_config.json /app/

# Copy the folders into the working directory
COPY onnx/ /app/onnx/
COPY voices/ /app/voices/

# Install the required dependencies from the requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Gradio uses (default is 7860)
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Command to run the app when the container starts
CMD ["python", "app.py"]
