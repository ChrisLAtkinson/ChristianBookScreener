# LGBTQ Book Identifier App

The **LGBTQ Book Identifier** is a Streamlit application that processes a CSV file of book titles to identify books featuring LGBTQ characters or themes. It uses online searches and natural language processing to analyze book synopses for LGBTQ-related content.

---

## Features

- **Upload CSV File**: Users can upload a CSV file containing book titles (must include a column named `Title`).
- **Automated Search**: The app fetches book synopses from the web using Google search.
- **LGBTQ Theme Detection**: Analyzes synopses for LGBTQ-related keywords and themes using Natural Language Processing (NLP).
- **Interactive Table of Results**: Displays the analysis results, including the book title, fetched synopsis, and whether LGBTQ content was identified.
- **Downloadable Results**: Allows users to download the analysis results as a CSV file.

---

## How It Works

1. **Upload CSV File**: 
   - The user uploads a CSV file containing book titles.
2. **Automated Web Search**:
   - The app performs Google searches for each book title to retrieve its synopsis.
3. **Keyword Analysis**:
   - Synopses are scanned for LGBTQ-related keywords (e.g., "LGBTQ," "gay," "lesbian," "queer").
4. **Results Display**:
   - Titles with detected LGBTQ themes are highlighted, and the full dataset is displayed interactively.
5. **Download CSV**:
   - Users can download the results for further review.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Required libraries (specified in `requirements.txt`)

### Steps to Install

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/lgbtq-book-identifier.git
   cd lgbtq-book-identifier
