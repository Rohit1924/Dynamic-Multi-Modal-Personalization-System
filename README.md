# Dynamic Multi-Modal Personalization System

This is my internship project for generating personalized marketing content using AI. The system takes a customer's interests and a product name, then generates a marketing caption and a product image.

I used **SportX** as a sample brand and created two sample customer profiles to test the project.

## Technologies Used

- Python
- FastAPI for the backend API
- Streamlit for the user interface
- Qwen3-1.7B for text generation
- Stable Diffusion 1.5 for image generation
- Qdrant for retrieving brand guidelines (RAG)
- Google Colab for development and testing

## How It Works

1. Select a customer and enter a product name.
2. The system reads the customer's interests and browsing history.
3. Qdrant retrieves the relevant SportX brand guidelines.
4. Qwen3 generates a personalized marketing caption.
5. Stable Diffusion generates a product image.
6. The caption and image are returned to the user.

The backend also includes basic error logging, fallback responses if generation fails, and a small job queue to handle multiple requests.

## How to Run

A GPU is recommended because the project uses text and image generation models.

Start the backend:

```bash
pip install -r requirements-backend.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

Start Streamlit in another terminal:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deployment Status

**This project is not live deployed.**

I developed the project using Google Colab because running Qwen3 and Stable Diffusion requires GPU resources. A free Colab GPU is temporary and cannot be used as a reliable always-running public server.

I have not used paid GPU hosting, so I am submitting the source code and notebook on GitHub instead of a live demo link.

## Limitations

- Image generation can take more than 3 seconds, depending on the GPU.
- The job queue and Qdrant data are stored in memory and reset when the backend restarts.
- This is a working prototype for learning and demonstration, not a production deployment.
