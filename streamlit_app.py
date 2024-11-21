import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import concurrent.futures

# LGBTQ Keywords for analysis
LGBTQ_KEYWORDS = ["LGBTQ", "gay", "lesbian", "transgender", "queer", "nonbinary", "bisexual", "LGBT"]

# Function to fetch synopsis using DuckDuckGo scraping
def fetch_duckduckgo_synopsis(book_title):
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
    if not text:
        return False
    lower_text = text.lower()
    for keyword in LGBTQ_KEYWORDS:
        if keyword.lower() in lower_text:
            return True
    return False

# Function for parallel processing
def fetch_and_analyze_parallel(title):
    synopsis = fetch_duckduckgo_synopsis(title)
    lgbtq_content = analyze_lgbtq_content(synopsis)
    return {"Title": title, "Synopsis": synopsis, "LGBTQ Content": lgbtq_content}

# Streamlit app UI
st.title("LGBTQ Book Identifier with DuckDuckGo (Optimized)")
st.markdown(
    """
    Upload a CSV file containing book titles (with a column named 'Title').
    The app will analyze each title to identify LGBTQ themes or characters.
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

            # Process the titles
            if st.button("Start Analysis"):
                results = []
                total_titles = len(titles)

                with st.spinner("Analyzing titles... This may take some time."):
                    progress_bar = st.progress(0)  # Initialize progress bar

                    # Parallel processing
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        futures = [executor.submit(fetch_and_analyze_parallel, title) for title in titles]
                        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                            results.append(future.result())
                            progress_bar.progress((idx + 1) / total_titles)

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
