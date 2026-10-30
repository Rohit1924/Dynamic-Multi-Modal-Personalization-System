"""Small Streamlit frontend for the SportX API."""

import os
import time

import requests
import streamlit as st

st.set_page_config(page_title="SportX Marketing Generator", page_icon="👟")
st.title("SportX Marketing Generator")
st.write("Generate a marketing caption and product image for a customer profile.")

# Set API_BASE_URL in Streamlit Cloud secrets after deploying the backend.
api_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
try:
    api_url = st.secrets.get("API_BASE_URL", api_url).rstrip("/")
except FileNotFoundError:
    pass


@st.cache_data(ttl=60)
def load_customers(url):
    response = requests.get(f"{url}/customers", timeout=15)
    response.raise_for_status()
    return response.json()


try:
    customers = load_customers(api_url)
except requests.RequestException:
    st.error("Backend is not available. Check API_BASE_URL and make sure the API is running.")
    st.stop()

customer_names = {
    item["customer_id"]: f"Customer {item['customer_id']} — {', '.join(item['interests'])}"
    for item in customers
}
customer_id = st.selectbox("Customer", options=list(customer_names), format_func=customer_names.get)
product = st.text_input("Product", value="Running Shoes")

if "current_job" not in st.session_state:
    st.session_state.current_job = None

if st.button("Generate content", type="primary"):
    if not product.strip():
        st.warning("Please enter a product name.")
    else:
        try:
            response = requests.post(
                f"{api_url}/jobs/submit",
                json={"customer_id": customer_id, "product": product.strip()},
                timeout=20,
            )
            if response.status_code == 429:
                st.warning("The AI queue is full. Please try again later.")
            else:
                response.raise_for_status()
                st.session_state.current_job = response.json()["job_id"]
        except requests.RequestException as error:
            st.error(f"Could not submit request: {error}")

if st.session_state.current_job:
    job_id = st.session_state.current_job
    st.info(f"Job ID: {job_id}")
    status_box = st.empty()

    # Poll for up to five minutes. This is a demo, not a real-time production UI.
    result = None
    for _ in range(60):
        try:
            response = requests.get(f"{api_url}/jobs/{job_id}", timeout=15)
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as error:
            status_box.error(f"Could not check job: {error}")
            break

        if result.get("status") not in ("queued", "running"):
            break
        status_box.info(f"Status: {result['status']} — generating content...")
        time.sleep(5)

    if result and result.get("status") in ("success", "partial_success"):
        status_box.empty()
        st.subheader("Generated Caption")
        st.write(result["caption"])
        st.subheader("Generated Image")
        st.image(f"{api_url}{result['image_url']}", use_container_width=True)
        st.caption(f"Generation time: {result['generation_time_seconds']} seconds")
        if result.get("errors"):
            st.warning("Fallback used: " + ", ".join(result["errors"]))
        st.session_state.current_job = None
    elif result and result.get("status") == "failed":
        status_box.error("Generation failed. Please try again.")
        st.session_state.current_job = None
    elif result and result.get("status") in ("queued", "running"):
        status_box.warning("Still processing. Click below to check again.")
        if st.button("Check result again"):
            st.rerun()
