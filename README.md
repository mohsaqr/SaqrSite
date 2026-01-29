# SaqrSite

A Streamlit app for syncing Google Scholar publications with your BibTeX bibliography, with automatic DOI lookup and verification.

## Features

- **Google Scholar Integration**: Fetch publications from any Google Scholar profile
- **Multiple Input Methods**: Paste HTML, BibTeX, or text lists
- **AI-Powered Extraction**: Uses LM Studio for intelligent publication parsing
- **DOI Lookup**: Automatic DOI discovery via CrossRef API
- **DOI Verification**: Validates DOIs exist before including them
- **BibTeX Generation**: Produces properly formatted BibTeX with DOIs
- **Missing Paper Detection**: Finds papers not yet in your bibliography

## Installation

```bash
# Clone the repository
git clone https://github.com/mohsaqr/SaqrSite.git
cd SaqrSite

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

The app will open at http://localhost:8501

### Requirements

- Python 3.9+
- LM Studio running locally (for AI extraction)

## Configuration

In the sidebar:
- **Scholar Input Method**: Choose how to input publications
- **LM Studio URL**: API endpoint (default: http://localhost:1234/v1)
- **papers.bib URL**: Your existing bibliography file
- **DOI Lookup**: Toggle CrossRef DOI lookup
- **CrossRef Email**: Optional email for faster API access

## License

MIT
