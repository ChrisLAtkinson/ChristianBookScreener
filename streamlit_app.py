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

# LGBTQ Keywords for analysis
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

def search_qbd_online(title):
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

def search_scholastic_online(title):
    try:
        url = "https://clubs.scholastic.com/search"
        response = requests.get(url, params={"q": title})  # Use the search parameter
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            results = soup.find_all("div", class_="product-tile")
            for result in results:
                if title.lower() in result.text.lower():
                    return True
        return False
    except Exception as e:
        st.warning(f"Error searching Scholastic Clubs Database: {e}")
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
        # Step 1: Check QBD
        if search_qbd_online(title):
            results.append({
                "Title": title,
                "Synopsis": "Identified via Queer Books Database",
                "Review": "Identified via Queer Books Database",
                "LGBTQ Content": True,
                "Confidence Level": "High (Verified by QBD)"
            })
            continue

        # Step 2: Check Scholastic
        if search_scholastic_online(title):
            results.append({
                "Title": title,
                "Synopsis": "Identified via Scholastic Clubs Database",
                "Review": "Identified via Scholastic Clubs Database",
                "LGBTQ Content": False,  # Scholastic may not flag LGBTQ content explicitly
                "Confidence Level": "Moderate (Verified by Scholastic)"
            })
            continue

        # Step 3: Analyze with GPT
        synopsis = fetch_synopsis_with_gpt(title)
        review = fetch_reviews_with_gpt(title)
        combined_text = f"{synopsis} {review}"
        has_lgbtq_content = analyze_lgbtq_content(combined_text)
        confidence = "Moderate (GPT and keyword analysis)" if has_lgbtq_content else "Low (No strong evidence)"
        
        results.append({
            "Title": title,
            "Synopsis": synopsis,
            "Review": review,
            "LGBTQ Content": has_lgbtq_content,
            "Confidence Level": confidence
        })
    return results

# Streamlit app
st.title("LGBTQ Book Identifier")

# Initialize session state
if "cumulative_results" not in st.session_state:
    st.session_state.cumulative_results = pd.DataFrame(columns=["Title", "Synopsis", "Review", "LGBTQ Content", "Confidence Level"])

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    books = pd.read_csv(uploaded_file)
    if "Title" not in books.columns:
        st.error("The uploaded CSV must contain a column named 'Title'.")
    else:
        st.success("File uploaded successfully!")
        titles = books["Title"].dropna().tolist()
        st.write(f"Found {len(titles)} book titles.")

        # Process titles in batches
        batch_size = 100
        batches = [titles[i:i + batch_size] for i in range(0, len(titles), batch_size)]

        for batch_number, batch in enumerate(batches):
            st.write(f"Processing batch {batch_number + 1} of {len(batches)}...")
            progress = st.progress(0)
            batch_results = []

            for i, title in enumerate(batch):
                batch_results.extend(process_batch([title]))
                progress.progress((i + 1) / len(batch))

            batch_df = pd.DataFrame(batch_results)
            st.session_state.cumulative_results = pd.concat(
                [st.session_state.cumulative_results, batch_df], ignore_index=True
            )

            # Display batch results
            st.write(f"Batch {batch_number + 1} results:")
            st.dataframe(batch_df)

            # Batch-specific CSV download
            csv_batch = batch_df.to_csv(index=False)
            st.download_button(
                label=f"Download Batch {batch_number + 1} Results as CSV",
                data=csv_batch,
                file_name=f"batch_{batch_number + 1}_results.csv",
                mime="text/csv",
            )
            st.markdown("---")

        # Cumulative results download
        st.write("All batches processed! Here's the complete result:")
        st.dataframe(st.session_state.cumulative_results)

        cumulative_csv = st.session_state.cumulative_results.to_csv(index=False)
        st.download_button(
            label="Download Complete Results as CSV",
            data=cumulative_csv,
            file_name="cumulative_lgbtq_analysis_results.csv",
            mime="text/csv",
        )
else:
    st.info("Please upload a CSV file to proceed.")
