import os
import random
import time
import re
import json
from collections import OrderedDict
from datetime import datetime
from typing import List, Type

import pandas as pd
from bs4 import BeautifulSoup
from pydantic import BaseModel, create_model
import html2text
import tiktoken

from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


from openai import OpenAI
from ollama import Client

from api_management import get_api_key
from assets import (
    PRICING,
    HEADLESS_OPTIONS,
    SYSTEM_MESSAGE,
    USER_MESSAGE,
    PERSON_MESSAGE,
    LLAMA_MODEL_FULLNAME,
    HEADLESS_OPTIONS_DOCKER,
)

load_dotenv()


# Set up the Chrome WebDriver options


def is_running_in_docker():
    """
    Detect if the app is running inside a Docker container.
    This checks if the '/proc/1/cgroup' file contains 'docker'.
    """
    try:
        with open("/proc/1/cgroup", "rt") as file:
            return "docker" in file.read()
    except Exception:
        return False


def setup_selenium(attended_mode=False):
    options = Options()
    service = Service(ChromeDriverManager().install())

    # Apply headless options based on whether the code is running in Docker
    if is_running_in_docker():
        # Running inside Docker, use Docker-specific headless options
        for option in HEADLESS_OPTIONS_DOCKER:
            options.add_argument(option)
    else:
        # Not running inside Docker, use the normal headless options
        for option in HEADLESS_OPTIONS:
            options.add_argument(option)

    # Initialize the WebDriver
    driver = webdriver.Chrome(options=options)
    return driver


def fetch_html_selenium(url, attended_mode=False, driver=None):
    if driver is None:
        driver = setup_selenium(attended_mode)
        should_quit = True
        if not attended_mode:
            driver.get(url)
    else:
        should_quit = False
        # Do not navigate to the URL if in attended mode and driver is already initialized
        if not attended_mode:
            driver.get(url)

    try:
        if not attended_mode:
            # Add more realistic actions like scrolling
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(random.uniform(1.1, 1.8))
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/1.2);")
            time.sleep(random.uniform(1.1, 1.8))
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/1);")
            time.sleep(random.uniform(1.1, 1.8))
        # Get the page source from the current page
        html = driver.page_source
        return html
    finally:
        if should_quit:
            driver.quit()


def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove headers and footers based on common HTML tags or classes
    for element in soup.find_all(["header", "footer"]):
        element.decompose()  # Remove these tags and their content

    return str(soup)


def html_to_markdown_with_readability(html_content):
    cleaned_html = clean_html(html_content)

    # Convert to markdown
    markdown_converter = html2text.HTML2Text()
    markdown_converter.ignore_links = False
    markdown_content = markdown_converter.handle(cleaned_html)

    return markdown_content


def save_raw_data(raw_data: str, output_folder: str, file_name: str):
    """Save raw markdown data to the specified output folder."""
    os.makedirs(output_folder, exist_ok=True)
    raw_output_path = os.path.join(output_folder, file_name)
    with open(raw_output_path, "w", encoding="utf-8") as f:
        f.write(raw_data)
    print(f"Raw data saved to {raw_output_path}")
    return raw_output_path


def create_dynamic_listing_model(field_names: List[str]) -> Type[BaseModel]:
    """
    Dynamically creates a Pydantic model based on provided fields.
    field_name is a list of names of the fields to extract from the markdown.
    """
    # Initialize an OrderedDict to control field order
    field_definitions = OrderedDict()

    # Add other fields from field_names
    for idx, field in enumerate(field_names):
        field_definitions[field] = (str, ...)
    # Dynamically create the model with all field
    return create_model("DynamicListingModel", **field_definitions)


def create_listings_container_model(listing_model: Type[BaseModel]) -> Type[BaseModel]:
    """
    Create a container model that holds a list of the given listing model.
    """
    return create_model("DynamicListingsContainer", listings=(List[listing_model], ...))


def trim_to_token_limit(text, model, max_tokens=120000):
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    if len(tokens) > max_tokens:
        trimmed_text = encoder.decode(tokens[:max_tokens])
        return trimmed_text
    return text


def generate_system_message(listing_model: BaseModel) -> str:
    """
    Dynamically generate a system message based on the fields in the provided listing model.
    """
    # Use the model_json_schema() method to introspect the Pydantic model
    schema_info = listing_model.model_json_schema()

    # Extract field descriptions from the schema
    field_descriptions = []
    for field_name, field_info in schema_info["properties"].items():
        # Get the field type from the schema info
        field_type = field_info["type"]
        field_descriptions.append(f'"{field_name}": "{field_type}"')

    # Create the JSON schema structure for the listings
    schema_structure = ",\n".join(field_descriptions)

    # Generate the system message dynamically
    system_message = f"""
    You are an intelligent text extraction and conversion assistant. Your task is to extract structured information 
                        from the given text and convert it into a pure JSON format. The JSON should contain only the structured data extracted from the text, 
                        with no additional commentary, explanations, politeness, assistance suggestions, or extraneous information. 
                        You could encounter cases where you can't find the data of the fields you have to extract or the data will be in a foreign language.
                        Please process the following text and provide the output in pure JSON format with no words before or after the JSON:
    Please ensure the output strictly follows this schema:

    {{
        "article": [
            {{
                {schema_structure}
            }}
        ]
    }} """

    return system_message


def format_data(person, data, DynamicListingsContainer, DynamicListingModel, selected_model):
    token_counts = {}
    if selected_model in ["gpt-4o-mini", "gpt-4o-2024-08-06"]:
        # Use OpenAI API
        client = OpenAI(api_key=get_api_key('OPENAI_API_KEY'))
        completion = client.beta.chat.completions.parse(
            model=selected_model,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE + PERSON_MESSAGE + person},
                {"role": "user", "content": USER_MESSAGE + data},
            ],
            response_format=DynamicListingsContainer,
        )
        # Calculate tokens using tiktoken
        encoder = tiktoken.encoding_for_model(selected_model)
        input_token_count = len(encoder.encode(USER_MESSAGE + data))
        output_token_count = len(
            encoder.encode(json.dumps(completion.choices[0].message.parsed.dict()))
        )
        token_counts = {
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
        }
        return completion.choices[0].message.parsed, token_counts

    elif selected_model == "llama3.1:8b-text-fp16":
        # Dynamically generate the system message based on the schema
        sys_message = generate_system_message(DynamicListingModel)
        # print(SYSTEM_MESSAGE)
        # Point to the local server
        client = Client(host='http://localhost:11434')
        completion = client.chat(
            model=LLAMA_MODEL_FULLNAME,
            messages=[
                {"role": "system", "content": sys_message},
                {"role": "user", "content": USER_MESSAGE + data},
            ],
            stream=False
        )

        # Extract the content from the response
        response_content = completion.get("message").get("content")
        print("Summary of article:\n" + response_content)
        # Convert the content from JSON string to a Python dictionary
        parsed_response = json.loads(response_content)

        # Extract token usage
        token_counts = {
            "input_tokens": completion.get("prompt_eval_count"),
            "output_tokens": completion.get("eval_count")
        }

        return parsed_response, token_counts
    else:
        raise ValueError(f"Unsupported model: {selected_model}")


def save_formatted_data(
    formatted_data, output_folder: str, json_file_name: str, excel_file_name: str
):
    """Save formatted data as JSON and Excel in the specified output folder."""
    os.makedirs(output_folder, exist_ok=True)

    # Parse the formatted data if it's a JSON string (from Gemini API)
    if isinstance(formatted_data, str):
        try:
            formatted_data_dict = json.loads(formatted_data)
        except json.JSONDecodeError:
            raise ValueError(
                "The provided formatted data is a string but not valid JSON."
            )
    else:
        # Handle data from OpenAI or other sources
        formatted_data_dict = (
            formatted_data.dict() if hasattr(formatted_data, "dict") else formatted_data
        )

    # Save the formatted data as JSON
    json_output_path = os.path.join(output_folder, json_file_name)
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data_dict, f, indent=4)
    print(f"Formatted data saved to JSON at {json_output_path}")

    # Prepare data for DataFrame
    if isinstance(formatted_data_dict, dict):
        # If the data is a dictionary containing lists, assume these lists are records
        data_for_df = (
            next(iter(formatted_data_dict.values()))
            if len(formatted_data_dict) == 1
            else formatted_data_dict
        )
    elif isinstance(formatted_data_dict, list):
        data_for_df = formatted_data_dict
    else:
        raise ValueError(
            "Formatted data is neither a dictionary nor a list, cannot convert to DataFrame"
        )

    # Create DataFrame
    try:
        df = pd.DataFrame(data_for_df)
        print("DataFrame created successfully.")

        # Save the DataFrame to an Excel file
        excel_output_path = os.path.join(output_folder, excel_file_name)
        df.to_excel(excel_output_path, index=False)
        print(f"Formatted data saved to Excel at {excel_output_path}")

        return df
    except Exception as e:
        print(f"Error creating DataFrame or saving Excel: {str(e)}")
        return None


def calculate_price(token_counts, model):
    input_token_count = token_counts.get("input_tokens", 0)
    output_token_count = token_counts.get("output_tokens", 0)

    # Calculate the costs
    input_cost = input_token_count * PRICING[model]["input"]
    output_cost = output_token_count * PRICING[model]["output"]
    total_cost = input_cost + output_cost

    return input_token_count, output_token_count, total_cost


def generate_unique_folder_name(person):
    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    return f"{person}_{timestamp}"


def scrape_url(
    url: str,
    fields: List[str],
    selected_model: str,
    output_folder: str,
    file_number: int,
    markdown: str,
    person: str,
):
    """Scrape a single URL and save the results."""
    try:
        # Save raw data
        save_raw_data(markdown, output_folder, f"rawData_{file_number}.md")

        # Create the dynamic listing model
        DynamicListingModel = create_dynamic_listing_model(fields)

        # Create the container model that holds a list of the dynamic listing models
        DynamicListingsContainer = create_listings_container_model(DynamicListingModel)

        # Format data
        formatted_data, token_counts = format_data(
            person, markdown, DynamicListingsContainer, DynamicListingModel, selected_model
        )

        # Save formatted data
        save_formatted_data(
            formatted_data,
            output_folder,
            f"sorted_data_{file_number}.json",
            f"sorted_data_{file_number}.xlsx",
        )

        # Calculate and return token usage and cost
        input_tokens, output_tokens, total_cost = calculate_price(
            token_counts, selected_model
        )
        return input_tokens, output_tokens, total_cost, formatted_data

    except Exception as e:
        print(f"An error occurred while processing {url}: {e}")
        return 0, 0, 0, None


# Remove the main execution block if it's not needed for testing purposes
