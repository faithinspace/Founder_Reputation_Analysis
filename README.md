## Features

- Easy-to-use web interface.
- Custom field specification for data extraction.
- Dynamic data processing with Python and Streamlit.
- Direct download capabilities for extracted data in various formats.
- Attended mode

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.6 or higher
- Pip for managing Python packages

## Installation

Follow these steps to get your development environment running:

```bash
# It's recommended to create a virtual environment
python -m venv venv
# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On MacOS/Linux
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

## Launching the Application

To run, navigate to the project directory and run the following command:

```bash
streamlit run streamlit_app.py
```


## Usage
After launching the application, open your web browser to the indicated address (typically http://localhost:8501). Use the sidebar to input the URL and fields you wish to scrape, then click the "Scrape" button to see results.