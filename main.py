"""FastAPI backend for the SportX marketing project."""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from generator import IMAGE_DIR, generate_content

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sportx")

BASE_DIR = Path(__file__).resolve().parent
customers = json.loads((BASE_DIR / "customers.json").read_text(encoding="utf-8"))

app = FastAPI(title="SportX AI Marketing API", version="1.0")
app.mount("/images", StaticFiles(directory=str(IMAGE_DIR)), name="images")

# One worker prevents several image models from using the same GPU at once.
worker = ThreadPoolExecutor(max_workers=1)
queue_slots = threading.BoundedSemaphore(3)
jobs = {}
jobs_lock = threading.Lock()


class MarketingRequest(BaseModel):
    customer_id: int
    product: str = Field(min_length=1, max_length=100)

    @field_validator("product")
    @classmethod
    def check_product(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Product cannot be empty")
        return value


@app.get("/")
def home():
    return {"message": "SportX AI Marketing API is running"}


@app.get("/customers")
def list_customers():
    return customers


def release_slot(_future):
    queue_slots.release()


@app.post("/jobs/submit", status_code=202)
def submit_job(request: MarketingRequest):
    customer = next(
        (item for item in customers if item["customer_id"] == request.customer_id),
        None,
    )
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")

    if not queue_slots.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Queue is full. Try again later.")

    job_id = str(uuid4())
    try:
        future = worker.submit(generate_content, customer, request.product, job_id)
        with jobs_lock:
            jobs[job_id] = future
        future.add_done_callback(release_slot)
    except Exception:
        queue_slots.release()
        logger.exception("Could not submit job")
        raise HTTPException(status_code=500, detail="Could not submit job")

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        future = jobs.get(job_id)

    if future is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if not future.done():
        return {"job_id": job_id, "status": "running" if future.running() else "queued"}

    try:
        return {"job_id": job_id, **future.result()}
    except Exception:
        logger.exception("Job %s failed", job_id)
        return {"job_id": job_id, "status": "failed", "errors": ["unexpected_error"]}
