

# SHL Assessment Recommendation System - Complete Documentation

## 1. Project Overview
### Problem Statement
Automate the process of recommending SHL assessments based on job descriptions or other relevant text inputs.

### Solution
A recommendation system that:
1. Processes input text (job descriptions or queries)
2. Generates semantic embeddings
3. Matches against a database of SHL assessments
4. Returns the most relevant assessments with detailed analysis

## 2. System Architecture
### Components
1. **Input Processing Module**
   - URL scraping for job descriptions
   - Text preprocessing and cleaning
   - LLM-based content analysis

2. **Embedding Generation**
   - Sentence Transformer models for semantic embeddings
   - Embedding caching for efficiency

3. **Vector Database**
   - FAISS index for fast similarity search
   - Assessment metadata storage

4. **Recommendation Engine**
   - Similarity scoring
   - LLM-based attribute extraction
   - Result ranking

### Data Flow
1. User provides input (text or URL)
2. System processes input and generates embedding
3. Vector search finds similar assessments
4. Results are analyzed and formatted
5. Recommendations returned to user

## 3. Implementation Details
### Key Technologies
- Python 3.9+
- SentenceTransformers for embeddings
- FAISS for vector search
- LangChain for LLM integration
- Google's Gemma model for analysis

### Core Functions
1. `process_query()` - Handles input processing
2. `vector_search()` - Performs similarity matching
3. `extract_attributes()` - Analyzes and formats results
4. `calculate_metrics()` - Evaluates system performance

## 4. API Specifications
### Input
- Text string or URL

### Output
JSON array of assessment objects with:
- Assessment name
- URL
- Summary
- Key features
- Duration
- Remote testing availability
- Similarity score

## 5. Data Sources
1. SHL assessment descriptions
2. Pre-processed assessment metadata
3. Vector embeddings database

## 6. Evaluation Metrics
1. **Recall@K** - Proportion of relevant assessments in top K results
2. **MAP@K** - Mean Average Precision at K

## 7. Usage Examples
```python
from query_processing import process_query, vector_search, extract_attributes

# Process a job description URL
results = process_query("https://example.com/job-description")
print(results)
```

## 8. Deployment
### Requirements
- Python environment
- Google API key for Gemma
- FAISS index file

### Installation
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables
4. Run: `python query_processing.py`

## 9. Future Enhancements
1. User feedback integration
2. Assessment popularity weighting
3. Multi-language support
4. Batch processing capability