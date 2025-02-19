# Kokoro TTS served using Grindo, containerised in Docker
This repo is submitted as fulfillment of the Red DragonAI - AI in Production course project (run: Feb 2025).<br />

<p align="center">
  <img src="images/demo.jpg" alt="Kokoro TTS" />
</p>

## Overview

- **Kokoro TTS** was chosen as the model. The `model_q8f16.onnx` was selected for its small model size and high speed during inference. The ONNX model ensures framework agnosticism.
- **Grindo** was selected as the frontend to provide users with an interactive interface to engage with the model.
- **Docker** was chosen as the container solution to ensure an environment-agnostic deployment.

## Setup Instructions

To start the app, follow these steps:

1. **Build the Docker image:**

   ```bash
   docker build -t kokoro-gradio-app:v1 .
2. **Run the container:**
    ```bash
    docker run -it -p 7860:7860 kokoro-gradio-app:v1
After running the container, the app can be accessed at http://localhost:7860.
