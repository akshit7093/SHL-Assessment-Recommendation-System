import requests
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import logging
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
# Import the scraper module
from scraper import scrape_job_description, is_url

# Initialize models and caches as module-level singletons
_sentence_transformer = None
_llm = None
_llm_chain = None
_embedding_cache = {}

def get_sentence_transformer():
    global _sentence_transformer
    if _sentence_transformer is None:
        logging.info("Initializing SentenceTransformer model")
        _sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    return _sentence_transformer

def get_llm():
    global _llm, _llm_chain
    if _llm is None:
        logging.info("Initializing Gemma model")
        _llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", google_api_key=os.getenv("GOOGLE_API_KEY"))
        prompt_template = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant designed to analyze job descriptions and extract key information.
            Reply like you are the website and guiding users like a first person perspective.
            Based *only* on the following text content, please provide:
            1. A concise description of the particular assessment (2-4 sentences).
            2. Key features, benefits, or what it measures (up to 5 bullet points).
            Scraped Content:
            {context}
            Analysis:
            """
        )
        output_parser = StrOutputParser()
        _llm_chain = {"context": RunnablePassthrough()} | prompt_template | _llm | output_parser
    return _llm_chain

def generate_embedding(text):
    # Use cache to avoid regenerating embeddings for identical text
    cache_key = hash(text)
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]
    
    # Generate new embedding
    model = get_sentence_transformer()
    embedding = model.encode([text])
    
    # Cache the result
    _embedding_cache[cache_key] = embedding
    return embedding

# Function to process query and generate embedding
def process_query(input_data):
    try:
        # Check if input is a URL
        if is_url(input_data):
            # Scrape job description from URL
            text = scrape_job_description(input_data)
            
            # Check if scraping returned an error message
            if text.startswith("Unable to access") or text.startswith("No job description"):
                logging.warning(f"Scraping failed for URL: {input_data}")
                # Still try to process the error message to avoid breaking the flow
                processed_text = f"Query: {input_data}\n\nNote: {text}"
            else:
                try:
                    # Process the scraped content with Gemma to understand job requirements
                    llm_chain = get_llm()
                    job_analysis = llm_chain.invoke(text)
                    # Combine the original text with the analysis for better embedding
                    processed_text = f"Job Description: {text}\n\nAnalysis: {job_analysis}"
                except Exception as e:
                    logging.error(f"Error analyzing job description with LLM: {str(e)}")
                    # Fallback to just using the scraped text
                    processed_text = f"Job Description: {text}"
        else:
            # If not a URL, use the input text directly
            processed_text = input_data
            
        # Generate embedding from the processed text
        embedding = generate_embedding(processed_text)
        return embedding
    except Exception as e:
        logging.error(f"Error in process_query: {str(e)}")
        # Return a default embedding for the error message to avoid breaking the flow
        error_text = f"Error processing query: {str(e)}"
        return generate_embedding(error_text)

# Function to perform vector search
def vector_search(query_embedding):
    try:
        # Load the vector index
        index = faiss.read_index('shl_vector_index.idx')
        # Perform similarity search
        distances, indices = index.search(query_embedding, k=10)
        return distances, indices
    except Exception as e:
        logging.error(f"Error in vector search: {str(e)}")
        # Return empty results that won't break the flow
        # Create empty arrays with the right shape
        empty_indices = np.zeros((1, 10), dtype=np.int64)
        empty_distances = np.ones((1, 10), dtype=np.float32) * 999  # Large distance = low similarity
        return empty_distances, empty_indices

# Function to extract attributes from top results using Gemma
def extract_attributes(distances, indices):
    try:
        # Load and cache the processed data
        if not hasattr(extract_attributes, 'processed_data'):
            try:
                with open('shl_processed_analysis_specific.json', 'r', encoding='utf-8') as f:
                    extract_attributes.processed_data = json.load(f)
            except Exception as e:
                logging.error(f"Error loading processed data: {str(e)}")
                # Return empty results if data can't be loaded
                return [{
                    'Assessment Name': 'Error',
                    'URL': 'N/A',
                    'description': f"Error loading assessment data: {str(e)}",
                    'Key Features': [],
                    'Duration': '',
                    'Remote Testing': False,
                    'Raw Analysis': '',
                    'Similarity Score': 0
                }]
        processed_data = extract_attributes.processed_data
        results = []
        
        for i, idx in enumerate(indices[0]):
            try:
                # Handle index out of bounds
                if idx >= len(processed_data):
                    logging.warning(f"Index {idx} out of bounds for processed_data with length {len(processed_data)}")
                    continue
                    
                item = processed_data[idx]
                similarity_score = 1 / (0.5 + distances[0][i])  # Adjusted formula to boost similarity scores
                
                # Filter to only include assessment-specific URLs containing '/view/'
                if '/view/' not in item.get('url', ''):
                    continue
                extracted_text = item.get('extracted_text', '')
                
                if not extracted_text:
                    logging.warning(f"Empty extracted text for index {idx}")
                    continue
                
                try:
                    # Use Gemma to analyze the assessment details with a structured prompt
                    llm_chain = get_llm()
                    
                    # Extract assessment-specific content from the vector DB
                    assessment_content = extracted_text.split('\n\n')[0]  # Get first section which contains assessment-specific info
                    
                    analysis = llm_chain.invoke(
                        f"""Assessment Data:
                        {assessment_content}
                        
                        Please analyze this specific assessment and provide a focused, assessment-specific output with these exact section headers. Avoid general company information.
                        
                        ## description:
                        [Provide a 1 sentence description that specifically describes what this assessment measures, its primary purpose, and its target audience. Focus only on this specific assessment's unique characteristics.]
                        
                        ## Key Features:
                        - [List 3-5 specific features or capabilities of this assessment]
                        - [Focus on what skills/abilities it measures]
                        - [Include technical aspects like adaptive testing if applicable]
                        
                        ## Duration:
                        [Specify exact duration in minutes if available, or provide estimated time range]
                        
                        ## Remote Testing:
                        [Yes/No - Include any specific remote proctoring details if available]
                        
                        ## Target Role/Level:
                        [Specify the job roles, levels, or industries this assessment is designed for]
                        """
                    )
                except Exception as e:
                    logging.error(f"Error analyzing assessment with LLM: {str(e)}")
                    # Use a placeholder analysis if LLM fails
                    analysis = f"Assessment information. Unable to analyze details: {str(e)}"
                
                # Process the structured analysis output
                analysis_lines = analysis.split('\n')
                description = ''
                features = []
                assessment_name = item.get('title', '') or 'SHL Assessment'
                duration = ''
                remote_testing = False
                
                # Parse the structured response sections
                current_section = ''
                for line in analysis_lines:
                    line = line.strip()
                    if line.startswith('##'):
                        current_section = line.replace('#', '').strip().lower()
                    elif line and current_section == 'description:':
                        description = line.strip('[]')
                    elif line.startswith('-') and current_section == 'key features:':
                        feature = line.strip('- []')
                        if feature:
                            features.append(feature)
                    elif current_section == 'duration:':
                        if line and not line.startswith('['):
                            duration = line.strip('[]')
                    elif current_section == 'remote testing:':
                        remote_testing = 'yes' in line.lower() or 'available' in line.lower() or 'supported' in line.lower()
                # Parse the structured response sections
                current_section = ''
                for line in analysis_lines:
                    line = line.strip()
                    if line.startswith('##'):
                        current_section = line.replace('#', '').strip().lower()
                    elif line and current_section == 'description:':
                        # Extract clean description without brackets
                        if '[' in line and ']' in line:
                            description = line[line.find('[')+1:line.find(']')]
                        else:
                            description = line
                    elif line.startswith('-') and current_section == 'key features:':
                        feature = line.strip('- []')
                        if feature:
                            features.append(feature)
                    elif current_section == 'duration:':
                        if line and not line.startswith('['):
                            duration = line.strip('[]')
                    elif current_section == 'remote testing:':
                        remote_testing = 'yes' in line.lower() or 'available' in line.lower() or 'supported' in line.lower()
                
                # Clean up and validate the description
                if not description or len(description.strip()) < 10:
                    # Fallback to a basic description if the LLM output is insufficient
                    description = f"Assessment measuring key competencies and skills for {assessment_name}."
                
                # Ensure features list is not empty
                if not features:
                    features = ["Measures relevant job competencies", "Provides standardized assessment"]
                
                # Clean up duration string
                if duration:
                    # Extract numbers from duration string
                    duration_numbers = re.findall(r'\d+', duration)
                    if duration_numbers:
                        duration = duration_numbers[0]  # Take the first number found
                        
                # Fallback duration extraction if not found in analysis
                if not duration and 'approximate completion time' in extracted_text.lower():
                    time_match = re.search(r'Approximate Completion Time in minutes = (\d+)', extracted_text, re.IGNORECASE)
                    if time_match:
                        duration = f"{time_match.group(1)} minutes"

                result = {
                    'Assessment_Name': assessment_name,
                    'URL': item.get('url', 'N/A'),
                    'description': description,
                    'Key_Features': features,
                    'Duration': duration,
                    'Remote_Testing': remote_testing,
                    'Raw_Analysis': analysis,
                    'Similarity_Score': similarity_score
                }
                results.append(result)
            
            except Exception as e:
                logging.error(f"Error processing result at index {i}: {str(e)}")
                # Add an error result instead of failing completely
                results.append({
                    'Assessment_Name': 'Error',
                    'URL': 'N/A',
                    'description': f"Error processing assessment: {str(e)}",
                    'Key_Features': [],
                    'Duration': '',
                    'Remote_Testing': False,
                    'Raw_Analysis': '',
                    'Similarity_Score': 0
                })
        
        # If no results were found or all processing failed, return a helpful message
        if not results:
            results.append({
                'Assessment_Name': 'No Results',
                'URL': 'N/A',
                'description': "No matching assessments found for your query.",
                'Key_Features': ["Try a different search term", "Be more specific about the job role or skills"],
                'Duration': '',
                'Remote_Testing': False,
                'Raw_Analysis': '',
                'Similarity_Score': 0
            })
            
        return results
    except Exception as e:
        logging.error(f"Unexpected error in extract_attributes: {str(e)}")
        # Return a single error result
        return [{
            'Assessment Name': 'Error',
            'URL': 'N/A',
            'description': f"An unexpected error occurred: {str(e)}",
            'Key Features': ["Please try again later"],
            'Duration': '',
            'Remote Testing': False,
            'Raw Analysis': '',
            'Similarity Score': 0
        }]

# Example usage
def calculate_metrics(results, relevant_assessments, k=3):
    """Calculate Mean Recall@K and MAP@K metrics.
    
    Args:
        results: List of retrieved assessment results
        relevant_assessments: List of relevant assessment IDs/names
        k: Number of top results to consider (default: 3)
    
    Returns:
        tuple: (recall@k, map@k)
    """
    if not results or not relevant_assessments:
        return 0.0, 0.0
    
    # Get top K results
    top_k = results[:k]
    retrieved_assessments = [r['Assessment_Name'] for r in top_k]
    
    # Calculate Recall@K
    relevant_retrieved = sum(1 for r in retrieved_assessments if r in relevant_assessments)
    recall_k = relevant_retrieved / len(relevant_assessments) if relevant_assessments else 0.0
    
    # Calculate MAP@K
    precision_sum = 0.0
    relevant_count = 0
    
    for i, assessment in enumerate(retrieved_assessments, 1):
        if assessment in relevant_assessments:
            relevant_count += 1
            precision_at_i = relevant_count / i
            precision_sum += precision_at_i
    
    map_k = precision_sum / min(k, len(relevant_assessments)) if relevant_assessments else 0.0
    
    return recall_k, map_k

def main():
    try:
        input_query = "Your input query or URL here"
        query_embedding = process_query(input_query)
        distances, indices = vector_search(query_embedding)
        # Reshape indices and distances to match expected format
        if len(indices.shape) == 1:
            indices = indices.reshape(1, -1)
            distances = distances.reshape(1, -1)
        results = extract_attributes(distances=distances, indices=indices)
        
        # Example usage of metrics calculation
        # In a real scenario, relevant_assessments would come from ground truth data
        relevant_assessments = ["Example Assessment 1", "Example Assessment 2"]
        recall_k, map_k = calculate_metrics(results, relevant_assessments)
        logging.info(f"Mean Recall@3: {recall_k:.3f}")
        logging.info(f"MAP@3: {map_k:.3f}")
        
        return results
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        return [{
            'Assessment Name': 'Error',
            'URL': 'N/A',
            'description': f"An error occurred while processing your query: {str(e)}",
            'Key Features': ["Please try again later"],
            'Duration': '',
            'Remote Testing': False,
            'Raw Analysis': '',
            'Similarity Score': 0
        }]

if __name__ == "__main__":
    results = main()
    print(results)