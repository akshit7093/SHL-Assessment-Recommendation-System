import requests
from bs4 import BeautifulSoup
import re
import logging
import time
from random import uniform
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_url(text):
    """Check if the input text is a URL."""
    url_pattern = re.compile(
        r'^(?:http|https)://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'
        r'(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(url_pattern.match(text))

def create_session_with_retry():
    """Create a requests session with retry capabilities."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,  # Maximum number of retries
        backoff_factor=1,  # Time factor between retries
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def scrape_job_description(url, max_retries=3):
    """Scrape job description content from a URL with retry mechanism."""
    retry_count = 0
    last_error = None
    
    # Different user agents to rotate
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
    ]
    
    while retry_count < max_retries:
        try:
            # Add random delay between retries to avoid rate limiting
            if retry_count > 0:
                sleep_time = uniform(1, 3) * retry_count
                logging.info(f"Retry {retry_count}/{max_retries} for {url}, waiting {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            
            # Rotate user agents
            headers = {
                'User-Agent': user_agents[retry_count % len(user_agents)],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            # Use session with retry capabilities
            session = create_session_with_retry()
            response = session.get(url, headers=headers, timeout=15)  # Increased timeout
            response.raise_for_status()
        
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Extract text from common job description containers
            job_content = ""
            
            # Look for common job description containers
            potential_containers = soup.select(
                '.job-description, .description, .job-details, '
                '#job-description, #description, #job-details, '
                '[class*="job"][class*="description"], '
                '[class*="job"][class*="details"], '
                '[id*="job"][id*="description"], '
                '[id*="job"][id*="details"]'
            )
            
            if potential_containers:
                for container in potential_containers:
                    job_content += container.get_text(separator='\n', strip=True) + "\n\n"
            else:
                # If no specific containers found, get the main content
                main_content = soup.select('main, article, .content, #content')
                if main_content:
                    for content in main_content:
                        job_content += content.get_text(separator='\n', strip=True) + "\n\n"
                else:
                    # Fallback to body content
                    job_content = soup.body.get_text(separator='\n', strip=True)
            
            # Clean up the text
            job_content = re.sub(r'\n+', '\n', job_content).strip()
            
            if not job_content:
                logging.warning(f"No content extracted from {url}")
                # Return a fallback message that won't cause issues with embedding
                return "No job description content could be extracted from the provided URL."
                
            logging.info(f"Successfully scraped content from {url}")
            return job_content
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Error scraping {url} (attempt {retry_count+1}/{max_retries}): {str(e)}")
            last_error = e
            retry_count += 1
        except Exception as e:
            logging.error(f"Unexpected error scraping {url} (attempt {retry_count+1}/{max_retries}): {str(e)}")
            last_error = e
            retry_count += 1
    
    # If all retries failed, return a fallback message
    error_message = f"Failed to scrape job description after {max_retries} attempts: {str(last_error)}"
    logging.error(error_message)
    return "Unable to access the job description at this time. Please try again later or provide the job description text directly."

def extract_job_details(text):
    """Extract structured job details from scraped text."""
    # This function can be expanded to extract specific job details
    # like job title, required skills, experience level, etc.
    # For now, we'll just return the cleaned text
    return text.strip()