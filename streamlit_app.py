# streamlit_app.py
from collections import OrderedDict

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import List

from scraper import (
    fetch_html_selenium,
    save_raw_data,
    format_data,
    save_formatted_data,
    calculate_price,
    html_to_markdown_with_readability,
    create_dynamic_listing_model,
    create_listings_container_model,
    setup_selenium,
    generate_unique_folder_name,
)
import re
from urllib.parse import urlparse
from assets import PRICING
import os

# Initialize Streamlit app
st.set_page_config(page_title="Starship Ventures: Founder Reputation Analysis", page_icon="🚀")
st.title("Founder Reputation Analysis 🚀")
st.text("Welcome! This is an app that helps investors analyze the online reputation of founders in lesser time than manual searches. It finds industry and market-specific mentions, summarizing the article's sentiment of the founder and translating it into English, saving you time from reading it.")
st.text("It costs about $0.0004 per query, so you would need an OpenAI key, which you can trial here for free: https://community.openai.com/t/openai-api-keys-in-free-account/348972 . After each query, you can see on the sidebar how many tokens you used and the precise cost per search.")
st.link_button("Click here for a short video tutorial on how it works", "https://www.canva.com/design/DAGWGtY-qiA/_1dEHBDEM_dsq29ncTZ6Tg/watch?utm_content=DAGWGtY-qiA&utm_campaign=designshare&utm_medium=link&utm_source=editor")

# Initialize session state variables
if "scraping_state" not in st.session_state:
    st.session_state["scraping_state"] = (
        "idle"  # Possible states: 'idle', 'waiting', 'scraping', 'completed'
    )
if "results" not in st.session_state:
    st.session_state["results"] = None
if "driver" not in st.session_state:
    st.session_state["driver"] = None

# Sidebar components
st.sidebar.title("Web Scraper Settings")

# API Keys
with st.sidebar.expander("API Keys", expanded=False):
    st.session_state["openai_api_key"] = st.text_input(
        "OpenAI API Key", type="password"
    )

# Model selection
model_selection = st.sidebar.selectbox(
    "Select Model", options=list(PRICING.keys()), index=0
)

# URL input
url_input = st.sidebar.text_input("Do not enter anything here")
# Process URLs
urls = list([

    "https://www.nac-zaken.nl/nieuws/greneer-gaat-voor-een-versnelde-energietransitie-met-financieringsuitbreiding-bij-beequip-",
    "https://mtsprout.nl/ranglijst/challenger50-van-2021/greener-power-solutions",
    "https://www.deondernemer.nl/innovatie/scale-up-greener-power-solutions-breidt-uit-naar-het-verenigd-koninkrijk~1146101"
    "https://www.f6s.com/member/dietercastelein",
    ]
)
# url_input = st.sidebar.text_input("Enter URL(s) separated by whitespace")
# # Process URLs
# urls = url_input.strip().split()

num_urls = len(urls)

person = st.sidebar.text_input("Enter the person you're looking up")

# Fields to extract
fields = list(["Article title", "Short LLM Description", "Positive", "Neutral", "Negative", "Overall sentiment"])

st.sidebar.markdown("---")

# Conditionally display Attended Mode options
if num_urls <= 1:
    # Attended mode toggle
    attended_mode = st.sidebar.toggle("Enable Attended Mode")
else:
    # Multiple URLs entered; disable Attended Mode
    attended_mode = False
    # Inform the user
    st.sidebar.info(
        "Attended Mode is disabled when multiple URLs are entered."
    )

st.sidebar.markdown("---")


# Main action button
if st.sidebar.button("LAUNCH SCRAPER", type="primary"):
    if person == "":
        st.error("Please enter the name of the person you're looking up.")
    else:
        # Set up scraping parameters in session state
        st.session_state["urls"] = urls
        st.session_state["fields"] = fields
        st.session_state["person"] = person
        st.session_state["model_selection"] = model_selection
        st.session_state["attended_mode"] = attended_mode
        st.session_state["scraping_state"] = "waiting" if attended_mode else "scraping"

# Scraping logic
if st.session_state["scraping_state"] == "waiting":
    # Attended mode: set up driver and wait for user interaction
    if st.session_state["driver"] is None:
        st.session_state["driver"] = setup_selenium(attended_mode=True)
        st.session_state["driver"].get(st.session_state["urls"][0])
        st.write("Perform any required actions in the browser window that opened.")
        st.write("Navigate to the page you want to scrape.")
        st.write("When ready, click the 'Resume Scraping' button.")
    else:
        st.write(
            "Browser window is already open. Perform your actions and click 'Resume Scraping'."
        )

    if st.button("Resume Scraping"):
        st.session_state["scraping_state"] = "scraping"
        st.rerun()

elif st.session_state["scraping_state"] == "scraping":
    with st.spinner("Scraping in progress..."):
        # Perform scraping
        output_folder = os.path.join(
            "output", generate_unique_folder_name(person)
        )
        os.makedirs(output_folder, exist_ok=True)

        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0
        all_data = []

        driver = st.session_state.get("driver", None)
        if st.session_state["attended_mode"] and driver is not None:
            # Attended mode: scrape the current page without navigating
            # Fetch HTML from the current page
            raw_html = fetch_html_selenium(
                st.session_state["urls"][0], attended_mode=True, driver=driver
            )
            markdown = html_to_markdown_with_readability(raw_html)
            save_raw_data(markdown, output_folder, "rawData_1.md")

            current_url = (
                driver.current_url
            )  # Use the current URL for logging and saving purposes

            # Scrape data if fields are specified
            if fields:
                # Create dynamic models
                DynamicListingModel = create_dynamic_listing_model(
                    st.session_state["fields"]
                )
                DynamicListingsContainer = create_listings_container_model(
                    DynamicListingModel
                )
                # Format data
                formatted_data, token_counts = format_data(
                    person,
                    markdown,
                    DynamicListingsContainer,
                    DynamicListingModel,
                    st.session_state["model_selection"],
                )
                input_tokens, output_tokens, cost = calculate_price(
                    token_counts, st.session_state["model_selection"]
                )
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_cost += cost
                # Save formatted data
                df = save_formatted_data(
                    formatted_data,
                    output_folder,
                    "sorted_data_1.json",
                    "sorted_data_1.xlsx",
                )
                all_data.append(formatted_data)
        else:
            # Non-attended mode or driver not available
            for i, url in enumerate(st.session_state["urls"], start=1):
                # Fetch HTML
                raw_html = fetch_html_selenium(url, attended_mode=False)
                markdown = html_to_markdown_with_readability(raw_html)
                save_raw_data(markdown, output_folder, f"rawData_{i}.md")

                # Scrape data if fields are specified
                if fields:
                    # Create dynamic models
                    DynamicListingModel = create_dynamic_listing_model(
                        st.session_state["fields"]
                    )
                    DynamicListingsContainer = create_listings_container_model(
                        DynamicListingModel
                    )
                    # Format data
                    formatted_data, token_counts = format_data(
                        person,
                        markdown,
                        DynamicListingsContainer,
                        DynamicListingModel,
                        st.session_state["model_selection"],
                    )
                    input_tokens, output_tokens, cost = calculate_price(
                        token_counts, st.session_state["model_selection"]
                    )
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    total_cost += cost
                    # Save formatted data
                    df = save_formatted_data(
                        formatted_data,
                        output_folder,
                        f"sorted_data_{i}.json",
                        f"sorted_data_{i}.xlsx",
                    )
                    all_data.append(formatted_data)

        # Clean up driver if used
        if driver:
            driver.quit()
            st.session_state["driver"] = None

        # Save results
        st.session_state["results"] = {
            "data": all_data,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_cost": total_cost,
            "output_folder": output_folder,
        }
        st.session_state["scraping_state"] = "completed"

# Display results of scraping costs
if st.session_state["scraping_state"] == "completed" and st.session_state["results"]:
    results = st.session_state["results"]
    all_data = results["data"]
    total_input_tokens = results["input_tokens"]
    total_output_tokens = results["output_tokens"]
    total_cost = results["total_cost"]
    output_folder = results["output_folder"]

    # Display scraping details
    if fields:
        st.subheader("Scraping Results")
        for i, data in enumerate(all_data, start=1):
            st.write(f"Data from URL {i}: \n" + st.session_state["urls"][i-1])

            # Handle string data (convert to dict if it's JSON)
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    st.error(f"Failed to parse data as JSON for URL {i}")
                    continue

            if isinstance(data, dict):
                if "listings" in data and isinstance(data["listings"], list):
                    df = pd.DataFrame(data["listings"])
                else:
                    # If 'listings' is not in the dict or not a list, use the entire dict
                    df = pd.DataFrame([data])
            elif hasattr(data, "listings") and isinstance(data.listings, list):
                # Handle the case where data is a Pydantic model
                listings = [item.dict() for item in data.listings]
                df = pd.DataFrame(listings)
            else:
                st.error(f"Unexpected data format for URL {i}")
                continue
            # Display the dataframe
            st.dataframe(df, use_container_width=True)

        # Display token usage and cost
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Scraping Details")
        st.sidebar.markdown("#### Token Usage")
        st.sidebar.markdown(f"*Input Tokens:* {total_input_tokens}")
        st.sidebar.markdown(f"*Output Tokens:* {total_output_tokens}")
        st.sidebar.markdown(f"**Total Cost:** :green-background[**${total_cost:.4f}**]")

        # Download options
        st.subheader("Download Extracted Data")
        col1, col2 = st.columns(2)
        with col1:
            json_data = json.dumps(
                all_data,
                default=lambda o: o.dict() if hasattr(o, "dict") else str(o),
                indent=4,
            )
            st.download_button(
                "Download JSON", data=json_data, file_name="scraped_data.json"
            )
        with col2:
            # Convert all data to a single DataFrame
            all_listings = []
            for data in all_data:
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                if isinstance(data, dict) and "listings" in data:
                    all_listings.extend(data["listings"])
                elif hasattr(data, "listings"):
                    all_listings.extend([item.dict() for item in data.listings])
                else:
                    all_listings.append(data)

            combined_df = pd.DataFrame(all_listings)
            st.download_button(
                "Download CSV",
                data=combined_df.to_csv(index=False),
                file_name="scraped_data.csv",
            )

        st.success(f"Scraping completed. Results saved in {output_folder}")

    # Reset scraping state
    if st.sidebar.button("Clear Results"):
        st.session_state["scraping_state"] = "idle"
        st.session_state["results"] = None

# Helper function to generate unique folder names
def generate_unique_folder_name(person):
    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    return f"{person}_{timestamp}"
