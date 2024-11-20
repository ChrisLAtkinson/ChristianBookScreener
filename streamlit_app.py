import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import nltk

# Download NLTK resources
nltk.download("punkt")

# LGBTQ Keywords for analysis
LGBTQ_KEYWORDS = ["LGBTQ", "gay", "lesbian", "transgender", "queer", "nonbinary", "bisexual", "LGBT"]

# Function to fetch synopsis from online sources
def fetch_synopsis(book_title):
    try:
        # Perform Google search
        query = f"{book_title} book synopsis"
        for url in search(query, num=1, stop=1):
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            # Extract paragraphs for synopsis
            paragraphs = soup.find_all("p")
            return " ".join([p.text for p in paragraphs[:3]])  # First 3 paragraphs as synopsis
    except Exception as e:
        return f"Error fetching synopsis: {e}"

# Function to analyze synopsis for LGBTQ content
def analyze_lgbtq_content(text):
    if not text:
        return False
    tokens = nltk.word_tokenize(text.lower())
    for keyword in LGBTQ_KEYWORDS:
        if keyword.lower() in tokens:
            return True
    return False

# Streamlit app UI
st.title("LGBTQ Book Identifier")
st.markdown(
    """
    Upload a CSV file containing book titles (with a column named 'Title').
    The app will analyze each title to identify LGBTQ themes or characters.
    """
)

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
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
                    synopsis = fetch_synopsis(title)
                    has_lgbtq_content = analyze_lgbtq_content(synopsis)
                    results.append(
                        {"Title": title, "Synopsis": synopsis, "LGBTQ Content": has_lgbtq_content}
                    )

            # Convert results to DataFrame
            result_df = pd.DataFrame(results)
            st.write("Analysis Complete!")
            st.dataframe(result_df)

            # Download results as CSV
            st.download_button(
                label="Download Results as CSV",
                data=result_df.to_csv(index=False),
                file_name="lgbtq_books_results.csv",
                mime="text/csv",
            )
