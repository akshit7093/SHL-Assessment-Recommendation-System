# SHL Assessment Recommendation System

## Overview
This project provides an intelligent assistant that helps analyze SHL assessment data and generate personalized recommendations for hiring decisions. It extracts, processes, and analyzes SHL-provided candidate assessments, enabling talent acquisition teams to make data-driven decisions efficiently.

## Features
- Extracts data from SHL exported datasets
- Cleans and processes raw assessment data
- Converts data into vector embeddings stored with FAISS for similarity search
- Utilizes LLMs (Google Generative AI via LangChain) for smart recommendation generation
- Offers FastAPI backend with endpoints for analysis and recommendation
- Includes web scraping tools for SHL portal (if needed)
- Modular design enabling extension and integration

## System Architecture
- **Backend:** FastAPI
- **Data Processing:** numpy, pandas (if used), faiss-cpu, BeautifulSoup (for scraping SHL portal if automated data retrieval required)
- **LLM Interface:** LangChain, google-generativeai, langchain-google-genai
- **Vector Storage:** FAISS
- **Environment Management:** python-dotenv

## Data Flow
1. **Data Extraction:** Scrape/download SHL assessment reports exported in CSV or JSON
2. **Preprocessing:** Clean, normalize, filter relevant features
3. **Embedding:** Convert candidate data into vector representations
4. **Storage:** Store vectors in FAISS index for efficient retrieval
5. **Query:** On a search/query, retrieve similar profiles and generate recommendations using large language models
6. **API Response:** Return recommendations via FastAPI endpoints

## Tech Stack & Libraries
- fastapi
- uvicorn
- requests
- beautifulsoup4
- python-dotenv
- pydantic
- langchain
- google-generativeai
- langchain-google-genai
- numpy
- faiss-cpu

## API Usage
- **Endpoint:** `/recommend` (example)
- **Method:** POST
- **Request Body:**
```json
{
  "candidate_profile": "<String with candidate data or query>"
}
```
- **Response:**
```json
{
  "recommendation": "<String with tailored recommendation>"
}
```

## Setup Instructions
1. **Clone Repository:**
```bash
git clone <repo-url>
cd SHL-hiring-assist
```

2. **Create Virtual Environment (Optional but recommended):**
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

4. **Add Environment Variables:**
- Create a `.env` file with your API keys and configurations:
```
API_KEY=your_google_api_key
OTHER_CONFIG=your_value
```

5. **Run Server:**
```bash
uvicorn api:app --reload
```

## Deployment
- Use `gunicorn` or similar for production (e.g., with uvicorn workers)
- Containerize using Docker (add Dockerfile)
- Setup CI/CD via GitHub Actions
- Host on platforms like Azure, AWS, or Heroku

## Future Improvements
- Enhanced UI frontend
- Support more SHL test types
- Add analytics dashboard
- Role-based access & authentication
- More detailed logging and monitoring

## License
This project is proprietary/confidential or specify appropriate license here.

## Maintainer
- Your Name
- Contact info/email

---
This documentation serves as both developer reference and GitHub project description. Please update with your specific details and credentials as needed.