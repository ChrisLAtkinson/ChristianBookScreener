import streamlit as st
import pandas as pd
from openai import OpenAI, RateLimitError
import requests
from bs4 import BeautifulSoup
import time

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

def search_qbd_online(title):
    """
    Searches the Queer Books Database for a given title.
    """
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
        st.warning(f"Error searching QBD: {e}")
        return False

def search_scholastic_online(title):
    """
    Searches Scholastic's Clubs database for a given title.
    """
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
        st.warning(f"Error searching Scholastic: {e}")
        return False

def fetch_synopsis_with_gpt(title):
    """
    Fetches a synopsis using OpenAI GPT.
    """
    prompt = f"Provide a short synopsis for the book titled '{title}'."
    for _ in range(3):  # Retry logic
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
            time.sleep(2)  # Backoff
    return "Failed to fetch synopsis."

def fetch_reviews_with_gpt(title):
    """
    Fetches a detailed review using OpenAI GPT.
    """
    prompt = f"Provide a detailed review of the book titled '{title}'. Focus on themes and audience reception."
    for _ in range(3):  # Retry logic
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            time.sleep(2)  # Backoff
    return "Failed to fetch review."

def analyze_lgbtq_content(text):
    """
    Analyzes text for LGBTQ-related keywords.
    """
    if not text:
        return False
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in LGBTQ_KEYWORDS)

def process_batch(batch_number, titles):
    """
    Processes a batch of titles.
    """
    if batch_number in st.session_state.processed_batches:
        return  # Skip already processed batches

    results = []
    for title in titles:
        if search_qbd_online(title):
            results.append({
                "Title": title,
                "Synopsis": "Identified via QBD",
                "Review": "Identified via QBD",
                "LGBTQ Content": True,
                "Confidence Level": "High (QBD)"
            })
            continue

        if search_scholastic_online(title):
            results.append({
                "Title": title,
                "Synopsis": "Identified via Scholastic",
                "Review": "Identified via Scholastic",
                "LGBTQ Content": False,
                "Confidence Level": "Moderate (Scholastic)"
            })
            continue

        synopsis = fetch_synopsis_with_gpt(title)
        review = fetch_reviews_with_gpt(title)
        content_combined = f"{synopsis} {review}"
        lgbtq_content = analyze_lgbtq_content(content_combined)

        results.append({
            "Title": title,
            "Synopsis": synopsis,
            "Review": review,
            "LGBTQ Content": lgbtq_content,
            "Confidence Level": "Low (GPT)"
        })

    # Save results to session state
    batch_df = pd.DataFrame(results)
    st.session_state.results = pd.concat([st.session_state.results, batch_df], ignore_index=True)
    st.session_state.processed_batches.add(batch_number)

# UI
st.title("LGBTQ Book Identifier")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    books = pd.read_csv(uploaded_file)
    if "Title" not in books.columns:
        st.error("Uploaded file must contain a 'Title' column.")
    else:
        titles = books["Title"].dropna().tolist()
        batch_size = 100
        batches = [titles[i:i + batch_size] for i in range(0, len(titles), batch_size)]

        for i, batch in enumerate(batches):
            process_batch(i, batch)
            st.write(f"Batch {i + 1} results:")
            st.dataframe(st.session_state.results)

        # Download cumulative results
        csv_data = st.session_state.results.to_csv(index=False)
        st.download_button("Download All Results", csv_data, "lgbtq_analysis_results.csv", "text/csv")
