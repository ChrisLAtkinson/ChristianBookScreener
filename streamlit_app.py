import streamlit as st
import pandas as pd
import openai
import requests
from bs4 import BeautifulSoup
import time
import random

# Initialize OpenAI API
try:
    API_KEY = st.secrets["openai"]["api_key"]
    openai.api_key = API_KEY
except KeyError:
    st.error("OpenAI API key not found. Please add it to Streamlit secrets.")
    st.stop()

# LGBTQ Keywords
LGBTQ_KEYWORDS = [
    "LGBTQ", "LGBT", "gay", "lesbian", "transgender", "queer", "nonbinary", "bisexual",
    "homosexual", "same-sex", "same-gender", "two-spirit", "genderqueer", "intersex",
    "asexual", "aromantic", "gender non-conforming", "pansexual", "genderfluid",
    "drag queen", "drag king", "polyamorous", "coming out", "rainbow family",
    "chosen family", "queer representation", "gender dysphoria", "pride",
    "sexual orientation", "gender identity", "androgynous", "LGBTQ themes",
    "inclusive love", "non-heteronormative", "rainbow flag", "gay dads", "lesbian moms",
    "queer relationships", "LGBTQIA", "gender affirmation", "non-cisgender",
    "sapphic", "wlw", "mlm", "gender transition", "closeted", "allies", "pride month",
    "rainbow literature", "queer literature", "gender spectrum", "fluid gender",
    "identity exploration", "sexual fluidity", "trans rights", "two dads", "two moms"
]

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame(columns=["Title", "Synopsis", "Review", "LGBTQ Content", "Confidence Level"])
if "processed_batches" not in st.session_state:
    st.session_state.processed_batches = set()

# Helper functions
def fetch_synopsis_with_gpt(title, max_retries=3):
    prompt = f"Provide a short synopsis for the book titled '{title}'."
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
                temperature=0.7,
            )
            return response["choices"][0]["message"]["content"].strip()
        except openai.error.RateLimitError:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            st.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            st.warning(f"Error fetching synopsis: {e}")
            break
    return "Failed to fetch synopsis."

def analyze_lgbtq_content(text):
    if not text:
        return False
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in LGBTQ_KEYWORDS)

def process_title(title):
    synopsis = fetch_synopsis_with_gpt(title)
    lgbtq_content = analyze_lgbtq_content(synopsis)
    return {
        "Title": title,
        "Synopsis": synopsis,
        "Review": "",
        "LGBTQ Content": lgbtq_content,
        "Confidence Level": "Low (GPT)" if synopsis != "Failed to fetch synopsis." else "Failed"
    }

def process_batch(batch_number, titles):
    if batch_number in st.session_state.processed_batches:
        st.info(f"Batch {batch_number + 1} already processed.")
        return

    st.write(f"Processing Batch {batch_number + 1}...")
    batch_progress = st.progress(0)
    results = []

    for idx, title in enumerate(titles):
        result = process_title(title)
        results.append(result)

        # Update progress bar
        batch_progress.progress((idx + 1) / len(titles))
        time.sleep(0.2)  # Small delay between requests

    batch_df = pd.DataFrame(results)
    st.session_state.results = pd.concat([st.session_state.results, batch_df], ignore_index=True)
    st.session_state.processed_batches.add(batch_number)

    st.write(f"Batch {batch_number + 1} Results:")
    st.dataframe(batch_df)

    st.download_button(
        label=f"Download Batch {batch_number + 1} Results",
        data=batch_df.to_csv(index=False),
        file_name=f"batch_{batch_number + 1}_results.csv",
        mime="text/csv",
        key=f"batch_{batch_number + 1}_download"
    )

# Streamlit UI
st.title("LGBTQ Book Identifier")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    books = pd.read_csv(uploaded_file)
    if "Title" not in books.columns:
        st.error("Uploaded file must contain a 'Title' column.")
    else:
        titles = books["Title"].dropna().tolist()
        batch_size = 25  # Reduced batch size
        batches = [titles[i:i + batch_size] for i in range(0, len(titles), batch_size)]

        st.write(f"Total Batches: {len(batches)}")

        selected_batches = st.multiselect(
            "Select Batches to Process:",
            options=list(range(1, len(batches) + 1)),
            format_func=lambda x: f"Batch {x}",
        )
        selected_batch_indices = [batch - 1 for batch in selected_batches]

        if st.button("Process Selected Batches"):
            for batch_index in selected_batch_indices:
                process_batch(batch_index, batches[batch_index])
            st.success("Selected batches processed successfully.")

        if not st.session_state.results.empty:
            cumulative_df = st.session_state.results[
                ["Title", "Synopsis", "Review", "LGBTQ Content", "Confidence Level"]
            ]
            st.write("Cumulative Results:")
            st.dataframe(cumulative_df)

            st.download_button(
                label="Download All Results",
                data=cumulative_df.to_csv(index=False),
                file_name="cumulative_lgbtq_analysis_results.csv",
                mime="text/csv",
                key="cumulative_download"
            )
