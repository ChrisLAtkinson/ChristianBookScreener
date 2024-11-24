import streamlit as st
import pandas as pd
from openai import OpenAI, RateLimitError
import requests
from bs4 import BeautifulSoup
import concurrent.futures

# Initialize OpenAI API
try:
    API_KEY = st.secrets["openai"]["api_key"]
    openai_client = OpenAI(api_key=API_KEY)
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
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False  # Tracks if processing is done

def search_qbd_online(title):
    try:
        url = "https://qbdatabase.wpcomstaging.com/"
        response = requests.get(url, params={"s": title})
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            results = soup.find_all("article")
            for result in results:
                if title.lower() in result.text.lower():
                    return True
        return False
    except Exception as e:
        return False

def search_scholastic_online(title):
    try:
        url = "https://clubs.scholastic.com/search"
        response = requests.get(url, params={"q": title})
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            results = soup.find_all("div", class_="product-tile")
            for result in results:
                if title.lower() in result.text.lower():
                    return True
        return False
    except Exception as e:
        return False

def fetch_synopsis_with_gpt(title):
    prompt = f"Provide a short synopsis for the book titled '{title}'."
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except RateLimitError:
        return "Failed to fetch synopsis."

def analyze_lgbtq_content(text):
    if not text:
        return False
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in LGBTQ_KEYWORDS)

def process_title(title):
    """
    Processes a single title, combining database searches and GPT results.
    """
    if search_qbd_online(title):
        return {
            "Title": title,
            "Synopsis": "Identified via QBD",
            "Review": "Identified via QBD",
            "LGBTQ Content": True,
            "Confidence Level": "High (QBD)"
        }
    if search_scholastic_online(title):
        return {
            "Title": title,
            "Synopsis": "Identified via Scholastic",
            "Review": "Identified via Scholastic",
            "LGBTQ Content": False,
            "Confidence Level": "Moderate (Scholastic)"
        }

    synopsis = fetch_synopsis_with_gpt(title)
    lgbtq_content = analyze_lgbtq_content(synopsis)
    return {
        "Title": title,
        "Synopsis": synopsis,
        "Review": "",
        "LGBTQ Content": lgbtq_content,
        "Confidence Level": "Low (GPT)"
    }

def process_batch(batch_number, titles):
    """
    Processes a batch of titles using parallel processing for efficiency.
    """
    if batch_number in st.session_state.processed_batches:
        return  # Skip already processed batches

    st.write(f"Processing Batch {batch_number + 1}...")

    batch_progress = st.progress(0)
    results = []

    for idx, title in enumerate(titles):
        result = process_title(title)
        results.append(result)
        batch_progress.progress((idx + 1) / len(titles))

    batch_df = pd.DataFrame(results)
    batch_df = batch_df[["Title", "Synopsis", "Review", "LGBTQ Content", "Confidence Level"]]
    st.session_state.results = pd.concat([st.session_state.results, batch_df], ignore_index=True)
    st.session_state.processed_batches.add(batch_number)

    st.download_button(
        label=f"Download Batch {batch_number + 1} Results",
        data=batch_df.to_csv(index=False),
        file_name=f"batch_{batch_number + 1}_results.csv",
        mime="text/csv",
        key=f"batch_{batch_number + 1}_download",  # Unique key
    )

# UI
st.title("LGBTQ Book Identifier")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    books = pd.read_csv(uploaded_file)
    if "Title" not in books.columns:
        st.error("Uploaded file must contain a 'Title' column.")
    else:
        titles = books["Title"].dropna().tolist()
        batch_size = 500  # Batch size
        batches = [titles[i:i + batch_size] for i in range(0, len(titles), batch_size)]

        start_batch = st.selectbox("Select Starting Batch:", options=list(range(1, len(batches) + 1)), index=0)
        start_batch_index = start_batch - 1

        if st.button("Start Processing"):
            for batch_number, batch in enumerate(batches[start_batch_index:], start=start_batch_index):
                process_batch(batch_number, batch)

            st.session_state.processing_complete = True

        if st.session_state.processing_complete:
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
                key="cumulative_download",  # Unique key
            )
