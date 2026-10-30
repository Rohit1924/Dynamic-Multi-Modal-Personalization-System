"""Caption and image generation for the SportX demo."""

import logging
import time
from pathlib import Path

from PIL import Image, ImageDraw

logger = logging.getLogger("sportx")
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "generated_images"
IMAGE_DIR.mkdir(exist_ok=True)

# Models are loaded only when the first request needs them.
text_model = None
image_model = None
vector_store = None


def get_brand_rules(customer, product):
    global vector_store

    if vector_store is None:
        from langchain_core.documents import Document
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_qdrant import QdrantVectorStore
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        brand_text = (BASE_DIR / "brand_guidelines.txt").read_text(encoding="utf-8")
        doc = Document(page_content=brand_text, metadata={"brand": "SportX"})
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
        chunks = splitter.split_documents([doc])
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            location=":memory:",
            collection_name="sportx_brand",
        )

    search_text = (
        f"Guidelines for {product}. "
        f"Interests: {', '.join(customer['interests'])}. "
        f"Viewed: {', '.join(customer['recently_viewed'])}."
    )
    docs = vector_store.similarity_search(search_text, k=3)
    return "\n".join(doc.page_content for doc in docs)


def make_caption(customer, product):
    global text_model
    from transformers import pipeline

    if text_model is None:
        text_model = pipeline(
            "text-generation",
            model="Qwen/Qwen3-1.7B",
            device_map="auto",
            dtype="auto",
        )

    rules = get_brand_rules(customer, product)
    messages = [
        {
            "role": "system",
            "content": (
                "You write short marketing captions for SportX. "
                "Always mention SportX. Do not invent prices, discounts, "
                "features or competing brands. Return only the caption.\n"
                f"Brand guidelines:\n{rules}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Write one caption for {product}. "
                f"Customer interests: {', '.join(customer['interests'])}. "
                f"Recently viewed: {', '.join(customer['recently_viewed'])}."
            ),
        },
    ]
    response = text_model(
        messages,
        max_new_tokens=80,
        do_sample=False,
        tokenizer_encode_kwargs={"enable_thinking": False},
    )
    caption = response[0]["generated_text"][-1]["content"].strip()

    # A basic check; the prompt alone does not guarantee brand compliance.
    if not caption or "sportx" not in caption.lower():
        raise ValueError("Generated caption did not include the brand name")
    return caption


def make_image_prompt(customer, product):
    interests = " ".join(customer["interests"]).lower()
    if "running" in interests or "fitness" in interests:
        setting = "outdoor running track, active fitness lifestyle"
    else:
        setting = "modern city street, casual fashion lifestyle"

    return (
        f"Realistic commercial photo of {product}, {setting}, "
        "single pair of sneakers, clean composition, natural lighting, "
        "detailed product photography, no text, no logo, no watermark"
    )


def make_image(customer, product, job_id):
    global image_model
    import torch
    from diffusers import StableDiffusionPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is needed for image generation")

    if image_model is None:
        image_model = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
        )
        image_model.enable_model_cpu_offload()

    image = image_model(
        prompt=make_image_prompt(customer, product),
        num_inference_steps=20,
        height=512,
        width=512,
    ).images[0]

    filename = f"marketing_{job_id}.png"
    image.save(IMAGE_DIR / filename)
    return filename


def fallback_image():
    filename = "fallback.png"
    path = IMAGE_DIR / filename
    if not path.exists():
        image = Image.new("RGB", (512, 512), "#f1f5f9")
        draw = ImageDraw.Draw(image)
        draw.text((218, 245), "SportX", fill="#0f172a")
        image.save(path)
    return filename


def generate_content(customer, product, job_id):
    start = time.perf_counter()
    errors = []

    try:
        caption = make_caption(customer, product)
    except Exception:
        logger.exception("Caption generation failed for customer %s", customer["customer_id"])
        caption = f"Explore {product} with SportX."
        errors.append("caption_generation_failed")

    try:
        image_filename = make_image(customer, product, job_id)
    except Exception:
        logger.exception("Image generation failed for customer %s", customer["customer_id"])
        image_filename = fallback_image()
        errors.append("image_generation_failed")

    return {
        "customer_id": customer["customer_id"],
        "product": product,
        "caption": caption,
        "image_url": f"/images/{image_filename}",
        "status": "partial_success" if errors else "success",
        "errors": errors,
        "generation_time_seconds": round(time.perf_counter() - start, 2),
    }
