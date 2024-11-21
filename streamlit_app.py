import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

# LGBTQ Keywords for analysis
LGBTQ_KEYWORDS = ["LGBTQ", "gay", "lesbian", "transgender", "queer", "nonbinary", "bisexual", "LGBT"]

# Function to fetch synopsis using DuckDuckGo scraping
def fetch_duckduckgo_synopsis(book_title):
    """
    Fetch a book synopsis using DuckDuckGo scraping.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://duckduckgo.com/html/?q={book_title} book synopsis"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for link in soup.find_all("a", class_="result__a"):
            snippet = link.get_text(strip=True)
            results.append(snippet)
        return results[0] if results else "No synopsis found."
    except Exception as e:
        return f"Error fetching synopsis: {e}"

# Function to analyze synopsis for LGBTQ content
def analyze_lgbtq_content(text):
    """
    Check if any LGBTQ keywords are present in the text.
    """
    if not text:
        return False
    lower_text = text.lower()
    for keyword in LGBTQ_KEYWORDS:
        if keyword.lower() in lower_text:
            return True
    return False

# Process a single batch of titles
def process_batch(titles_batch):
    """
    Process a batch of titles by fetching synopses and analyzing LGBTQ content.
    """
    results = []
    for title in titles_batch:
        synopsis = fetch_duckduckgo_synopsis(title)
        has_lgbtq_content = analyze_lgbtq_content(synopsis)
        results.append({"Title": title, "Synopsis": synopsis, "LGBTQ Content": has_lgbtq_content})
        time.sleep(1)  # Avoid overloading the server or getting rate-limited
    return results

# Streamlit app UI
st.title("LGBTQ Book Identifier with Batch Processing")
st.markdown(
    """
    Upload a CSV file containing book titles (with a column named 'Title').
    The app will analyze each title to identify LGBTQ themes or characters, processing in batches of 100 titles.
    """
)

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        # Read the uploaded CSV
        books = pd.read_csv(uploaded_file)
        if "Title" not in books.columns:
            st.error("The uploaded CSV must contain a column named 'Title'.")
        else:
            st.success("File uploaded successfully!")
            titles = books["Title"].dropna().tolist()
            st.write(f"Found {len(titles)} book titles.")
            
            # Process the titles in batches of 100
            batch_size = 100
            total_batches = (len(titles) + batch_size - 1) // batch_size
            results = []

            for batch_number in range(total_batches):
                st.write(f"Processing batch {batch_number + 1} of {total_batches}...")
                start_index = batch_number * batch_size
                end_index = min((batch_number + 1) * batch_size, len(titles))
                batch = titles[start_index:end_index]

                # Process the current batch
                with st.spinner(f"Processing titles {start_index + 1} to {end_index}..."):
                    batch_results = process_batch(batch)
                    results.extend(batch_results)

                # Update progress
                progress = (batch_number + 1) / total_batches
                st.progress(progress)

                # Pause briefly before starting the next batch
                if batch_number + 1 < total_batches:
                    time.sleep(5)  # Short pause between batches

            # Convert results to DataFrame
            result_df = pd.DataFrame(results)
            st.write("Analysis Complete!")
            st.dataframe(result_df)

            # Download results as CSV
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="lgbtq_books_results.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
