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

# Process a single batch of titles with a real-time progress bar
def process_batch(titles_batch, batch_progress_bar):
    """
    Process a batch of titles by fetching synopses and analyzing LGBTQ content.
    """
    results = []
    for i, title in enumerate(titles_batch):
        synopsis = fetch_duckduckgo_synopsis(title)
        has_lgbtq_content = analyze_lgbtq_content(synopsis)
        results.append({"Title": title, "Synopsis": synopsis, "LGBTQ Content": has_lgbtq_content})

        # Update the per-batch progress bar
        batch_progress_bar.progress((i + 1) / len(titles_batch))
        time.sleep(1)  # Avoid overloading the server or getting rate-limited
    return results

# Streamlit app UI
st.title("LGBTQ Book Identifier with Batch Downloads")
st.markdown(
    """
    Upload a CSV file containing book titles (with a column named 'Title').
    The app will analyze each title to identify LGBTQ themes or characters, processing in batches of 100 titles.
    After each batch, you can download the results for that batch.
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
            cumulative_results = []

            for batch_number in range(total_batches):
                st.write(f"Processing batch {batch_number + 1} of {total_batches}...")
                start_index = batch_number * batch_size
                end_index = min((batch_number + 1) * batch_size, len(titles))
                batch = titles[start_index:end_index]

                # Initialize a progress bar for the current batch
                batch_progress_bar = st.progress(0)

                # Process the current batch
                with st.spinner(f"Processing titles {start_index + 1} to {end_index}..."):
                    batch_results = process_batch(batch, batch_progress_bar)
                    cumulative_results.extend(batch_results)

                # Convert batch results to DataFrame
                batch_df = pd.DataFrame(batch_results)
                st.write(f"Batch {batch_number + 1} results:")
                st.dataframe(batch_df)

                # Batch download button
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    label=f"Download Batch {batch_number + 1} Results as CSV",
                    data=csv,
                    file_name=f"lgbtq_books_batch_{batch_number + 1}.csv",
                    mime="text/csv",
                )

                # Pause briefly before starting the next batch
                if batch_number + 1 < total_batches:
                    st.write("Pausing briefly before the next batch...")
                    time.sleep(5)  # Short pause between batches

            # Download cumulative results
            cumulative_df = pd.DataFrame(cumulative_results)
            st.write("All batches processed! Download cumulative results below:")
            st.dataframe(cumulative_df)
            cumulative_csv = cumulative_df.to_csv(index=False)
            st.download_button(
                label="Download Cumulative Results as CSV",
                data=cumulative_csv,
                file_name="lgbtq_books_all_batches.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
