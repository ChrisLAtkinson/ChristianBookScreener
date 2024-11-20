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
        
        # Extract search results
        results = []
        for link in soup.find_all("a", class_="result__a"):
            snippet = link.get_text(strip=True)
            results.append(snippet)
        
        # Return the first result or a fallback message
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

# Streamlit app UI
st.title("LGBTQ Book Identifier with DuckDuckGo")
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
                with st.spinner("Analyzing titles... This may take some time."):
                    for title in titles:
                        synopsis = fetch_duckduckgo_synopsis(title)
                        has_lgbtq_content = analyze_lgbtq_content(synopsis)
                        results.append(
                            {"Title": title, "Synopsis": synopsis, "LGBTQ Content": has_lgbtq_content}
                        )
                        # Add a small delay to avoid hitting rate limits
                        time.sleep(1)

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
