import streamlit as st
import pandas as pd
from openai import OpenAI, RateLimitError
import time

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

# LGBTQ Keywords for analysis
LGBTQ_KEYWORDS = ["LGBTQ", "gay", "lesbian", "transgender", "queer", "nonbinary", "bisexual", "LGBT"]

def fetch_synopsis_with_gpt(book_title, max_retries=3):
    """
    Fetches a book synopsis using OpenAI GPT API with retry logic.

    Args:
        book_title: The title of the book.
        max_retries: The maximum number of retries in case of rate limits.

    Returns:
        The synopsis if successful, otherwise an error message.
    """

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

def analyze_lgbtq_content(text):
    """
    Analyzes the text for LGBTQ keywords.

    Args:
        text: The text to analyze.

    Returns:
        True if LGBTQ keywords are found, False otherwise.
    """

    if not text:
        return False
    lower_text = text.lower()
    for keyword in LGBTQ_KEYWORDS:
        if keyword.lower() in lower_text:
            return True
    return False

def process_batch(titles_batch):
    """
    Processes a batch of titles by fetching synopses and analyzing LGBTQ content.

    Args:
        titles_batch: A list of book titles.

    Returns:
        A list of dictionaries containing title, synopsis, and LGBTQ content flag.
    """

    results = []
    for title in titles_batch:
        synopsis = fetch_synopsis_with_gpt(title)
        has_lgbtq_content = analyze_lgbtq_content(synopsis)
        results.append({"Title": title, "Synopsis": synopsis, "LGBTQ Content": has_lgbtq_content})
    return results

# Streamlit app UI
st.title("LGBTQ Book Identifier with OpenAI GPT")
st.markdown(
    """
    Upload a CSV file containing book titles (with a column named 'Title').
    The app will analyze each title to identify LGBTQ themes or characters, processing in batches of 100 titles.
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

            for batch_number, batch in enumerate(batches):
                st.write(f"Processing batch {batch_number + 1} of {len(batches)}...")
                with st.spinner("Processing titles..."):
                    batch_results = process_batch(batch)

                batch_df = pd.DataFrame(batch_results)
                st.write(f"Batch {batch_number + 1} results:")
                st.dataframe(batch_df)

                # ... (add download button for batch results)

            # ... (add download button for cumulative results)

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
