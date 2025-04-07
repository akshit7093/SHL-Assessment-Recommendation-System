import os
import logging
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from query_processing import process_query, vector_search, extract_attributes
from scraper import is_url, scrape_job_description
import uvicorn

# Configure logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

class Query(BaseModel):
    query: str

class RecommendationResult(BaseModel):
    Assessment_Name: str
    URL: str
    Summary: str
    Key_Features: List[str]
    Duration: str
    Remote_Testing: bool
    Raw_Analysis: str
    Similarity_Score: float

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationResult]

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(query: Query):
    try:
        if not query.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Log the incoming query for debugging
        logging.info(f"Processing query: {query.query[:100]}{'...' if len(query.query) > 100 else ''}")
        
        # Process the query asynchronously
        query_embedding = await asyncio.to_thread(process_query, query.query)
        
        # Perform vector search asynchronously
        distances, indices = await asyncio.to_thread(vector_search, query_embedding)
        
        # Reshape indices and distances to match expected format
        if len(indices.shape) == 1:
            indices = indices.reshape(1, -1)
            distances = distances.reshape(1, -1)
            
        # Extract attributes from search results asynchronously
        results = await asyncio.to_thread(extract_attributes, distances=distances, indices=indices)
        
        # Log successful processing
        logging.info(f"Successfully processed query and found {len(results)} results")
        
        return {"recommendations": results}
        
    except Exception as e:
        logging.error(f"Error in recommendation pipeline: {str(e)}")
        # Return a user-friendly error message
        return JSONResponse(
            status_code=200,  # Return 200 with error content instead of 500
            content={
                "recommendations": [{
                    "Assessment_Name": "Error",
                    "URL": "N/A",
                    "Summary": "An error occurred while processing your request.",
                    "Key_Features": ["Please try again later or with a different query."],
                    "Duration": "",
                    "Remote_Testing": False,
                    "Raw_Analysis": "",
                    "Similarity_Score": 0
                }]
            }
        )

if __name__ == "__main__":
    # Configure server with single worker for better resource management
    # Disable TensorFlow logging for cleaner output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=5000,
        workers=1,  # Single worker to prevent resource conflicts
        log_level="info",
        limit_concurrency=10  # Limit concurrent requests
    )