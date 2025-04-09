import os
import logging
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from query_processing import process_query, vector_search, extract_attributes
from scraper import is_url, scrape_job_description

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI()

# Pydantic models for request/response validation
class Query(BaseModel):
    query: str

class Assessment(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendationResponse(BaseModel):
    recommended_assessments: List[Assessment]

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(query: Query):
    """Endpoint to get assessment recommendations based on job description or query."""
    try:
        query_text = query.query.strip()
        if not query_text:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Check if input is a URL and scrape if needed
        if is_url(query_text):
            try:
                query_text = await asyncio.to_thread(scrape_job_description, query_text)
                if not query_text:
                    raise HTTPException(status_code=400, detail="Could not extract job description from URL")
            except Exception as e:
                logging.error(f"Error scraping URL {query_text}: {str(e)}")
                raise HTTPException(status_code=400, detail="Failed to scrape job description from URL")
        
        # Process the query asynchronously
        query_embedding = await asyncio.to_thread(process_query, query_text)
        
        # Perform vector search asynchronously
        distances, indices = await asyncio.to_thread(vector_search, query_embedding)
        
        # Reshape indices and distances if needed
        if len(indices.shape) == 1:
            indices = indices.reshape(1, -1)
            distances = distances.reshape(1, -1)
            
        # Extract attributes from search results asynchronously
        raw_results = await asyncio.to_thread(extract_attributes, distances=distances, indices=indices)
        
        # Transform results to match required response format
        recommended_assessments = []
        for idx, result in enumerate(raw_results[:10]):  # Limit to 10 assessments
            # Calculate similarity score (normalized between 0 and 1)
            similarity_score = 1 / (1 + distances[0][idx]) if idx < len(distances[0]) else 0.0
            
            assessment = Assessment(
                url=result['URL'],
                description=result.get('description', '').strip(),
                duration=int(''.join(filter(str.isdigit, result.get('Duration', '60')))) if result.get('Duration') and any(c.isdigit() for c in result.get('Duration')) else 60,
                remote_support="Yes" if result.get('Remote_Testing', False) else "No",
                adaptive_support="Yes" if any("adaptive" in feature.lower() for feature in result.get('Key_Features', [])) else "No",
                test_type=[feature.strip() for feature in result.get('Key_Features', []) if feature.strip()]
            )
            recommended_assessments.append(assessment)
        
        return RecommendationResponse(recommended_assessments=recommended_assessments)
        
    except Exception as e:
        logging.error(f"Error in recommendation pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info"
    )