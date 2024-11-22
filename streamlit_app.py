import streamlit as st
import pandas as pd
from openai import OpenAI, RateLimitError
import time
import requests
from bs4 import BeautifulSoup

# Ensure OpenAI API key is loaded from Streamlit secrets
try:
    API_KEY = st.secrets["openai"]["api_key"]
    openai_client = OpenAI(api_key=API_KEY)
except KeyError:
    st.error(
        "OpenAI API key not found in Streamlit secrets. "
        "Please add the key to `.streamlit/secrets.toml` (local) or the Secrets Manager (Streamlit Cloud)."
    )
    st.stop()

# LGBTQ Keywords for analysis (expanded)
LGBTQ_KEYWORDS = [
    "LGBTQ", "gay", "lesbian", "transgender", "queer", "nonbinary", "bisexual",
    "LGBT", "homosexual", "dads", "moms", "parents", "family", "pride", "same-sex"
]

def search_qbd_online(title):
    """
    Searches the Queer Books Database online for the given book title.

    Args:
        title: The book title to search.

    Returns:
        True if the book exists in the database, False otherwise.
    """
    try:
        url = "https://qbdatabase.wpcomstaging.com/"
        response = requests.get(url, params={"s": title})  # Use the search parameter
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            results = soup.find_all("article")
            for result in results:
                if title.lower() in result.text.lower():
                    return True
        return False
    except Exception as e:
        st.warning(f"Error searching Queer Books Database: {e}")
        return False

def fetch_synopsis_with_gpt(book_title, max_retries=3):
    for retry_count in range(max_retries):
        try:
            prompt = f"Provide a short synopsis for the book titled '{book_title}'."
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
        except RateLimitError as e:
            wait_time = 2 ** retry_count
            st.warning(f"Rate limit exceeded, retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    return "Failed to fetch synopsis after multiple attempts."

def fetch_reviews_with_gpt(book_title, max_retries=3):
    for retry_count in range(max_retries):
        try:
            prompt = (
                f"Provide the most in-depth, critical review available for the book titled '{book_title}'. "
                "Focus on themes, character development, and audience reception."
            )
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
        except RateLimitError as e:
            wait_time = 2 ** retry_count
            st.warning(f"Rate limit exceeded, retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    return "Failed to fetch review after multiple attempts."

def analyze_lgbtq_content(text):
    if not text:
        return False
    lower_text = text.lower()
    for keyword in LGBTQ_KEYWORDS:
        if keyword.lower() in lower_text:
            return True
    return False

def process_batch(titles_batch):
    results = []
    for title in titles_batch:
        # Check online database first
        if search_qbd_online(title):
            results.append({
                "Title": title,
                "Synopsis": "Identified via Queer Books Database",
                "Review": "Identified via Queer Books Database",
                "LGBTQ Content": True
            })
        else:
            synopsis = fetch_synopsis_with_gpt(title)
            review = fetch_reviews_with_gpt(title)
            combined_text = f"{synopsis} {review}"
            has_lgbtq_content = analyze_lgbtq_content(combined_text)
            results.append({
                "Title": title,
                "Synopsis": synopsis,
                "Review": review,
                "LGBTQ Content": has_lgbtq_content
            })
    return results

# Streamlit app UI
st.title("LGBTQ Book Identifier with OpenAI GPT")
st.markdown(
    """
    Upload a CSV file containing book titles (with a column named 'Title').
    The app will analyze each title to identify LGBTQ themes or characters by searching online databases
    and analyzing synopses and reviews, processing in batches of 100 titles.
    """
)

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        books = pd.read_csv(uploaded_file)
        if "Title" not in books.columns:
            st.error("The uploaded CSV must contain a column named 'Title'.")
        else:
            st.success("File uploaded successfully!")
            titles = books["Title"].dropna().tolist()
            st.write(f"Found {len(titles)} book titles.")

            # Process the titles in batches of 100
            batch_size = 100
            batches = [titles[i:i + batch_size] for i in range(0, len(titles), batch_size)]

            cumulative_results = []
            for batch_number, batch in enumerate(batches):
                st.write(f"Processing batch {batch_number + 1} of {len(batches)}...")
                progress = st.progress(0)
                batch_results = []

                for i, title in enumerate(batch):
                    batch_results.extend(process_batch([title]))
                    progress.progress((i + 1) / len(batch))

                cumulative_results.extend(batch_results)
                st.markdown("---")

            cumulative_df = pd.DataFrame(cumulative_results)
            st.write("All batches processed! Here's the complete result:")
            st.dataframe(cumulative_df)

            csv = cumulative_df.to_csv(index=False)
            st.download_button(
                label="Download Complete Results as CSV",
                data=csv,
                file_name="lgbtq_analysis_results.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
