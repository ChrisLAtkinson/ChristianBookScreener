import streamlit as st
import pandas as pd
import openai
import time

# Ensure OpenAI API key is loaded from Streamlit secrets
try:
    openai.api_key = st.secrets["openai"]["api_key"]
except KeyError:
    st.error(
        "OpenAI API key not found in Streamlit secrets. "
        "Please add the key to `.streamlit/secrets.toml` (local) or the Secrets Manager (Streamlit Cloud)."
    )
    st.stop()

# LGBTQ Keywords for analysis
LGBTQ_KEYWORDS = ["LGBTQ", "gay", "lesbian", "transgender", "queer", "nonbinary", "bisexual", "LGBT"]

# Function to fetch synopsis using OpenAI GPT
def fetch_synopsis_with_gpt(book_title):
    """
    Fetch a book synopsis using OpenAI GPT API.
    """
    try:
        prompt = f"Provide a short synopsis for the book titled '{book_title}'."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        return response["choices"][0]["message"]["content"].strip()
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
def process_batch(titles_batch, batch_progress_bar):
    """
    Process a batch of titles by fetching synopses and analyzing LGBTQ content.
    """
    results = []
    for i, title in enumerate(titles_batch):
        synopsis = fetch_synopsis_with_gpt(title)
        has_lgbtq_content = analyze_lgbtq_content(synopsis)
        results.append({"Title": title, "Synopsis": synopsis, "LGBTQ Content": has_lgbtq_content})

        # Update the progress bar
        batch_progress_bar.progress((i + 1) / len(titles_batch))
        time.sleep(0.5)  # Avoid overwhelming the API
    return results

# Streamlit app UI
st.title("LGBTQ Book Identifier with OpenAI GPT")
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
