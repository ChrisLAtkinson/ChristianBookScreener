import streamlit as st
import pandas as pd
from serpapi import GoogleSearch

# LGBTQ Keywords for analysis
LGBTQ_KEYWORDS = ["LGBTQ", "gay", "lesbian", "transgender", "queer", "nonbinary", "bisexual", "LGBT"]

# Function to fetch synopsis from SerpAPI
def fetch_synopsis(book_title, api_key):
    """
    Fetch a book synopsis using SerpAPI.
    """
    try:
        search = GoogleSearch({
            "q": f"{book_title} book synopsis",
            "api_key": api_key,
        })
        results = search.get_dict()
        # Get the snippet from organic results
        if "organic_results" in results and len(results["organic_results"]) > 0:
            return results["organic_results"][0].get("snippet", "No synopsis found.")
        return "No synopsis found."
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
st.title("LGBTQ Book Identifier")
st.markdown(
    """
    Upload a CSV file containing book titles (with a column named 'Title').
    The app will analyze each title to identify LGBTQ themes or characters.
    You need a [SerpAPI](https://serpapi.com/) key to fetch Google search results.
    """
)

# API Key Input
api_key = st.text_input("Enter your SerpAPI Key", type="password")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if api_key and uploaded_file:
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
                        synopsis = fetch_synopsis(title, api_key)
                        has_lgbtq_content = analyze_lgbtq_content(synopsis)
                        results.append(
                            {"Title": title, "Synopsis": synopsis, "LGBTQ Content": has_lgbtq_content}
                        )

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
    if not api_key:
        st.info("Please enter your SerpAPI Key.")
    if not uploaded_file:
        st.info("Please upload a CSV file to proceed.")
