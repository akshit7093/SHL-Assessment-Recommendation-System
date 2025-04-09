import os
import json
import csv
import time
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Configuration ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# --- Crawler Configuration ---
START_URL = "https://www.shl.com/solutions/products/product-catalog/"
BASE_CATALOG_URL_PREFIX = "https://www.shl.com/solutions/products/product-catalog/"
# Define allowed URL patterns based on user input
ALLOWED_PATTERNS = [
    re.compile(r"^" + re.escape(BASE_CATALOG_URL_PREFIX) + r"$"), # Exact base URL
    re.compile(r"^" + re.escape(BASE_CATALOG_URL_PREFIX) + r"\?start=\d+"), # Pagination URLs
    re.compile(r"^" + re.escape(BASE_CATALOG_URL_PREFIX) + r"view/"), # Detail View URLs
]
MAX_PAGES_TO_CRAWL = 200 # Safety limit (adjust as needed)
RAW_DATA_FILENAME = "shl_raw_scraped_data_specific.jsonl" # New file for specific crawl

# --- Processing & Output Configuration ---
PROCESSED_JSON_FILENAME = "shl_processed_analysis_specific.json" # New file
PROCESSED_CSV_FILENAME = "shl_processed_analysis_specific.csv" # New file
POLITE_DELAY_SECONDS = 1
REQUEST_TIMEOUT = 30
MAX_LLM_CONTENT_CHARS = 15000
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

# --- LLM Setup ---
# Re-initialize LLM components (same as before)
llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", google_api_key=google_api_key)
prompt_template = ChatPromptTemplate.from_template(
    """
    Based *only* on the following text content scraped from an SHL web page, please provide:
    1. A concise summary of the page's main topic (2-4 sentences).
    2. If it describes an assessment or product, list its key features, benefits, or what it measures (up to 5 bullet points). Otherwise, state "Not applicable".

    Do not add any information not present in the text. If the text is insufficient or irrelevant, state that.

    Scraped Content:
    {context}

    Analysis:
    """
)
output_parser = StrOutputParser()
llm_chain = {"context": RunnablePassthrough()} | prompt_template | llm | output_parser

# --- Helper Functions ---

def get_soup(url):
    """Fetches URL content and returns a BeautifulSoup object or None on error."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            print(f"  Skipping URL {url}: Non-HTML content type ({content_type})")
            return None, None
        html_content = response.content.decode('utf-8', errors='ignore')
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup, response.text
    except requests.exceptions.Timeout:
        print(f"  Timeout error fetching {url}")
        return None, None
    except requests.exceptions.RequestException as e:
        print(f"  Request error fetching {url}: {e}")
        return None, None
    except Exception as e:
        print(f"  Error processing {url}: {e}")
        return None, None

def is_allowed_shl_url(url):
    """Checks if the URL matches one of the defined allowed patterns."""
    # Simple check first
    if not url or not url.startswith(BASE_CATALOG_URL_PREFIX):
        return False
    # Check against regex patterns
    for pattern in ALLOWED_PATTERNS:
        if pattern.match(url):
            return True
    # print(f"    Debug: URL rejected by patterns: {url}") # Optional debug
    return False

def extract_text_from_html(html_content):
    """Extracts and cleans text from raw HTML string."""
    if not html_content: return None
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']): # Added form removal
            element.decompose()
        body = soup.body
        text = body.get_text(separator=' ', strip=True) if body else soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"  Error extracting text from HTML snippet: {e}")
        return None

# --- Phase 1: Crawl & Scrape Raw HTML (Specific URLs) ---

def crawl_and_scrape_raw_specific(start_url, max_pages, output_filename):
    print(f"--- Starting Phase 1: Crawling Specific SHL URLs (Max: {max_pages} pages) ---")
    queue = deque([start_url])
    visited_urls = {start_url}
    pages_scraped = 0

    with open(output_filename, 'w') as f: # Clear/prepare output file
        f.write("")
    print(f"Cleared/Prepared raw data file: {output_filename}")

    while queue and pages_scraped < max_pages:
        current_url = queue.popleft()
        print(f"\nProcessing ({pages_scraped + 1}/{max_pages}): {current_url}")

        # Check BEFORE fetching if URL is allowed (it should be if it came from queue, but good practice)
        if not is_allowed_shl_url(current_url):
             print(f"  Skipping non-allowed URL from queue (should not happen): {current_url}")
             continue

        soup, raw_html = get_soup(current_url)

        if raw_html:
            try:
                with open(output_filename, 'a', encoding='utf-8') as f_out:
                    json.dump({"url": current_url, "raw_html": raw_html}, f_out)
                    f_out.write('\n')
                pages_scraped += 1
                print(f"  Successfully scraped and saved raw HTML ({len(raw_html)} bytes).")
            except Exception as e:
                print(f"  Error saving raw data for {current_url}: {e}")

        if soup:
            links_found_on_page = 0
            new_links_added = 0
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(current_url, href)
                parsed_url = urlparse(absolute_url)
                normalized_url = parsed_url._replace(fragment="").geturl() # Remove fragment
                links_found_on_page += 1

                # *** Crucial Change: Check against specific allowed patterns ***
                if is_allowed_shl_url(normalized_url) and normalized_url not in visited_urls:
                    visited_urls.add(normalized_url)
                    queue.append(normalized_url)
                    new_links_added += 1
            print(f"  Inspected {links_found_on_page} links, added {new_links_added} new valid URLs to queue.")

        print(f"  Politely waiting for {POLITE_DELAY_SECONDS} second(s)...")
        time.sleep(POLITE_DELAY_SECONDS)

    print(f"\n--- Phase 1 Complete: Scraped {pages_scraped} pages matching allowed patterns. ---")
    print(f"Raw data saved incrementally to {output_filename}")


# --- Phase 2: Process Raw Data & AI Analysis (No Changes Needed) ---

def process_raw_data_with_ai(raw_data_input_filename):
    print(f"\n--- Starting Phase 2: Processing Raw Data from {raw_data_input_filename} & AI Analysis ---")
    processed_results = []
    processed_count = 0
    error_count = 0

    try:
        with open(raw_data_input_filename, 'r', encoding='utf-8') as f_in:
            for line_num, line in enumerate(f_in, 1):
                processed_count += 1
                url = f"Unknown (Line {line_num})" # Default if parsing fails
                try:
                    data = json.loads(line)
                    url = data.get("url", f"Unknown (Line {line_num})")
                    raw_html = data.get("raw_html")
                    print(f"\nProcessing item {processed_count}: {url}")

                    if not url or not raw_html:
                        print("  Skipping: Missing URL or raw HTML in record.")
                        error_count += 1
                        processed_results.append({
                            "url": url, "extracted_text": None, "ai_analysis": None,
                            "processing_status": "Error: Invalid Raw Data Record" })
                        continue

                    print("  Extracting text from raw HTML...")
                    extracted_text = extract_text_from_html(raw_html)
                    ai_analysis = None
                    if not extracted_text:
                        print("  Failed to extract text.")
                        status = "Error: Text Extraction Failed"
                    else:
                        print(f"  Extracted ~{len(extracted_text)} characters. Sending to AI...")
                        try:
                            truncated_text = extracted_text
                            if len(extracted_text) > MAX_LLM_CONTENT_CHARS:
                                truncated_text = extracted_text[:MAX_LLM_CONTENT_CHARS] + "... (truncated)"
                                print(f"    Text truncated to {MAX_LLM_CONTENT_CHARS} chars for LLM.")

                            ai_analysis = llm_chain.invoke(truncated_text)
                            print("  AI analysis received.")
                            status = "Success: Analyzed"
                            print(f"  Politely waiting for {POLITE_DELAY_SECONDS} second(s)...")
                            time.sleep(POLITE_DELAY_SECONDS)

                        except Exception as e:
                            print(f"  Error during AI analysis: {e}")
                            ai_analysis = f"Error during AI analysis: {e}"
                            status = f"Error: AI Failed ({type(e).__name__})"
                            error_count += 1

                    processed_results.append({
                        "url": url, "extracted_text": extracted_text, "ai_analysis": ai_analysis,
                        "processing_status": status })

                except json.JSONDecodeError as e:
                    print(f"  Skipping invalid JSON line {line_num}: {e}")
                    error_count += 1
                    processed_results.append({ # Add error record
                         "url": f"Unknown (Line {line_num})", "extracted_text": None, "ai_analysis": None,
                         "processing_status": "Error: Invalid JSON in Raw Data" })
                    continue
                except Exception as e:
                     print(f"  Unexpected error processing line {line_num} ({url}): {e}")
                     error_count += 1
                     processed_results.append({
                        "url": url, "extracted_text": None, "ai_analysis": None,
                        "processing_status": f"Error: Unexpected Processing Failure ({type(e).__name__})" })
                     continue

    except FileNotFoundError:
        print(f"Error: Raw data file '{raw_data_input_filename}' not found. Cannot proceed.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading raw data file: {e}")
        return processed_results

    print(f"\n--- Phase 2 Complete: Processed {processed_count} records with {error_count} errors. ---")
    return processed_results

# --- Phase 3: Save Processed Results (No Changes Needed) ---

def save_processed_results(final_data, json_filename, csv_filename):
    """Saves the final processed data to JSON and CSV files."""
    print("\n--- Starting Phase 3: Saving Processed Results ---")
    if not final_data:
        print("No processed data to save.")
        return

    all_keys = set()
    for item in final_data: all_keys.update(item.keys())
    fieldnames = sorted(list(all_keys))

    # Save to JSON
    try:
        with open(json_filename, 'w', encoding='utf-8') as f_json:
            json.dump(final_data, f_json, ensure_ascii=False, indent=4)
        print(f"Successfully saved processed JSON results to: {json_filename}")
    except Exception as e:
        print(f"Error saving processed data to JSON file ({json_filename}): {e}")

    # Save to CSV
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(final_data)
        print(f"Successfully saved processed CSV results to: {csv_filename}")
    except Exception as e:
        print(f"Error saving processed data to CSV file ({csv_filename}): {e}")

    print("\n--- Phase 3 Complete ---")


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Specific SHL URL Crawler and Analyzer...")

    # # --- Phase 1 ---
    # crawl_and_scrape_raw_specific(
    #     start_url=START_URL,
    #     max_pages=MAX_PAGES_TO_CRAWL,
    #     output_filename=RAW_DATA_FILENAME
    # )

    # --- Phase 2 ---
    processed_data = process_raw_data_with_ai(RAW_DATA_FILENAME)

    # --- Phase 3 ---
    save_processed_results(
        final_data=processed_data,
        json_filename=PROCESSED_JSON_FILENAME,
        csv_filename=PROCESSED_CSV_FILENAME
    )

    print("\nScript finished.")