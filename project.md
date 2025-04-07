

# SHL Assessment Recommendation System - Complete Documentation

## 1. Project Overview
### Problem Statement
Automate the process of recommending SHL assessments based on job descriptions or other relevant text inputs.

### Solution Approach
- Natural Language Processing to analyze input queries
- Vector similarity matching against SHL assessment database
- Web interface for user interaction
- API backend for processing requests

## 2. System Architecture
### Components
1. **Frontend**: Streamlit web application (frontend.py)
2. **Backend**: FastAPI service (api.py)
3. **Data Processing**:
   - Web scraper (scraper.py)
   - Query processor (query_processing.py)
   - Vector database converter (convert_to_vector_db.py)

### Data Flow
1. User submits query via frontend
2. Frontend sends query to backend API
3. Backend processes query and matches against assessment database
4. Results returned to frontend for display

## 3. Implementation Details
### Frontend
- Built with Streamlit
- Features:
  - Text input area
  - Recommendation display
  - Error handling

### Backend API
- Endpoints:
  - POST /recommend - Processes queries and returns recommendations
- Response format:
  ```json
  {
    "recommendations": [
      {
        "Assessment_Name": string,
        "URL": string,
        "Summary": string,
        "Key_Features": [string],
        "Duration": string,
        "Remote_Testing": boolean,
        "Similarity_Score": float
      }
    ]
  }
  ```

## 4. Data Sources
- SHL Product Catalog (https://www.shl.com/solutions/products/product-catalog/)
- Processed data stored in:
  - shl_raw_scraped_data_specific.jsonl
  - shl_processed_analysis_specific.json
  - shl_vector_index.idx

## 5. Usage Examples
### Running the System
1. Start backend API:
   ```bash
   uvicorn api:app --reload
   ```
2. Start frontend:
   ```bash
   streamlit run frontend.py
   ```

### Sample Queries
- "Need cognitive ability test for graduate hiring"
- "Looking for personality assessment for managerial roles"

## 6. Evaluation Metrics
- Recommendation relevance
- Response time
- Error rate

## 7. Future Enhancements
- User feedback system
- Additional filtering options
- Improved similarity algorithms

## 1. Project Overview

**Objective:**  
Design and develop an intelligent web application that simplifies the process for hiring managers to find the most appropriate SHL assessments for their open roles. The system must accept a natural language query, job description text, or URL as input and return a ranked list (up to 10 items) of the most relevant assessment tests from SHL’s product catalog.

**Business Context:**  
Hiring managers at SHL currently rely on keyword searches and manual filtering to select assessments—a process that is both time-consuming and inefficient. This project aims to leverage modern AI techniques, including emerging LLM stacks and evaluation/tracing methods, to deliver a more effective recommendation engine that is both accurate and user-friendly.

---

## 2. Detailed Problem Statement

### Input Specifications:
- **User Input:**  
  - A natural language query describing the role or required candidate skills.
  - Alternatively, a complete job description text or a URL containing the job description.
- **Processing Requirement:**  
  - Use a language model (e.g., via Google’s Gemini API or another LLM provider) to process and embed the input text.
  
### Output Specifications:
- **Recommendation List:**  
  - Return a minimum of 1 and a maximum of 10 SHL assessment tests in a tabular format.
- **Each Recommendation Must Include:**  
  - **Assessment Name:** The title of the assessment with a URL hyperlink to its SHL catalog page.
  - **Remote Testing Support:** Indicate “Yes” or “No.”
  - **Adaptive/IRT Support:** Indicate “Yes” or “No.”
  - **Duration:** The estimated time to complete the assessment.
  - **Test Type:** The category or type of assessment (e.g., Cognitive, Personality, etc.).

### Evaluation Metrics:
- **Mean Recall@3:**  
  - Measures the proportion of relevant assessments retrieved in the top 3 recommendations, averaged over all test queries.
- **Mean Average Precision@3 (MAP@3):**  
  - Evaluates both the relevance and ranking order of the top 3 retrieved assessments.
- **Target:**  
  - A higher Mean Recall@3 and MAP@3 indicate better performance.

### Example Queries for Benchmark Testing:
- “I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.”
- “Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript. Need an assessment package that can test all skills with a maximum duration of 60 minutes.”
- “Here is a JD text, can you recommend some assessments that can help me screen applications with a time limit of less than 30 minutes?”
- “I am hiring for an analyst and want applications to screen using Cognitive and personality tests. What options are available within 45 minutes?”

---

## 3. Detailed Project Planning & Approach

### Phase 1: Requirement Analysis and Objectives Definition
- **Understand SHL’s Business Domain:**  
  - Review the SHL product catalog ([SHL Product Catalog](https://www.shl.com/solutions/products/product-catalog/)) to grasp the types, features, and support details of the assessments.
- **Define Objectives & KPIs:**  
  - Primary goal: Accelerate the hiring process by matching roles with the most relevant assessments.
  - KPIs include Mean Recall@3, MAP@3, user satisfaction, and reduced time in selecting assessments.

### Phase 2: Data Acquisition & Preprocessing
- **Data Sources:**  
  - Obtain the assessment data from the SHL product catalog.
  - If possible, extract any additional metadata (e.g., test descriptions, support details, duration).
- **Data Cleaning:**  
  - Normalize text fields, manage missing data, and standardize categorical values (e.g., converting “yes/no” to Boolean flags).
- **Feature Extraction:**  
  - For text fields (assessment descriptions), apply vectorization methods such as TF-IDF or generate embeddings using an LLM.
  - Construct structured records for each assessment including all required attributes.

### Phase 3: Recommendation Engine Design
- **Input Processing:**  
  - Accept a natural language query, job description text, or URL.
  - Use an LLM (e.g., via Gemini API) to convert the input into an embedding vector.
- **Content-Based Filtering:**  
  - Represent each assessment as a feature vector using preprocessed metadata and text embeddings.
  - Compute cosine similarity between the query embedding and each assessment vector.
- **Ranking & Filtering:**  
  - Rank assessments by similarity score.
  - Apply additional business rules and filters (e.g., duration limits, remote testing flag) based on the user’s query.
  - Ensure that only 1–10 assessments are returned.
- **Hybrid Considerations (Optional):**  
  - If candidate or interaction data is available, integrate collaborative filtering signals to refine recommendations.
  - Leverage evals and tracing tools to explain the recommendation decisions.

### Phase 4: API & Application Development
- **Backend Development:**  
  - Use Python for the recommendation engine and develop the API using Flask, FastAPI, or a low-code framework like Streamlit or Gradio.
  - Develop endpoints that accept queries and return JSON-formatted recommendation results.
- **Frontend/Demo Application:**  
  - Build a simple user interface where users can input queries and view the recommendation table.
  - Use low-code frameworks to speed up development if front-end skills are limited.
- **Integration:**  
  - Ensure the API and demo are fully integrated, responsive, and scalable.

### Phase 5: Testing, Tuning, and Evaluation
- **Offline Evaluation:**  
  - Build a benchmark test set using the provided example queries.
  - Compute Mean Recall@3 and MAP@3 for performance assessment.
- **Parameter Tuning:**  
  - Experiment with different vectorization techniques (e.g., TF-IDF vs. LLM embeddings) and similarity thresholds.
  - Fine-tune business rules (e.g., filtering by duration and support features) to improve relevance.
- **Iterative Feedback:**  
  - Use eval tracing to log model decisions and refine recommendations.
  - Incorporate user or peer feedback to improve UI and algorithm accuracy.

### Phase 6: Deployment & Documentation
- **Deployment:**  
  - Containerize the application using Docker.
  - Deploy the web demo and API endpoint on a cloud service (e.g., Heroku, AWS Free Tier, or Streamlit Sharing) to obtain public URLs.
- **Documentation:**  
  - Publish the code on GitHub with clear README documentation.
  - Write a one-page document summarizing your approach, tools used (e.g., Gemini API, scikit-learn, Flask/FastAPI, etc.), and evaluation results.
- **Submission:**  
  - Provide three URLs:  
    1. A URL to the live demo.  
    2. An API endpoint URL (JSON response).  
    3. The GitHub repository URL with all source code.
  - Include the one-page document outlining your complete approach and planning.

---

## 4. Tools & Libraries

- **Programming Language:** Python  
- **Frameworks:** Flask or FastAPI for API, Streamlit/Gradio for demo UI  
- **Libraries:**  
  - Data processing: pandas, numpy  
  - Text processing: scikit-learn (TF-IDF), transformers/LLM API (e.g., Gemini API)  
  - Similarity computation: scikit-learn metrics, numpy  
  - Evaluation: Custom scripts for Mean Recall@3 and MAP@3  
- **Deployment:** Docker, cloud hosting (Heroku/AWS/Azure/Streamlit Sharing)

---

## 5. Final Considerations

- **Accuracy & Business Impact:**  
  - Focus on achieving high Mean Recall@3 and MAP@3 on the benchmark queries.
  - Ensure that the recommendations align with real-world hiring requirements.
- **Explainability:**  
  - Incorporate logging/tracing to explain why certain assessments were recommended.
- **Demo Quality:**  
  - Prioritize a clean, user-friendly demo that clearly shows input, processing, and results.
- **Iterative Improvement:**  
  - Be prepared to iterate on the design based on testing and feedback.

Below is a structured outline that lists every link provided in the assignment asset along with a description of its purpose and how it is used in the project:

---

### 1. SHL Product Catalog  
- **Link:** [https://www.shl.com/solutions/products/product-catalog/](https://www.shl.com/solutions/products/product-catalog/)  
- **Use:**  
  - **Primary Data Source:** This catalog contains the full list of SHL assessment tests along with detailed metadata such as assessment name, test type, duration, and support features (e.g., remote testing, adaptive/IRT support).  
  - **Data Extraction:** Your system will crawl or ingest this data to build the knowledge base for recommendations.

---

### 2. Gemini API Pricing  
- **Link:** [https://ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing)  
- **Use:**  
  - **Access to LLM Capabilities:** This link provides details on the free APIs (including pricing) for Google’s Gemini language model.  
  - **Query Processing:** You can leverage Gemini’s API to convert natural language queries or job description texts into embeddings that are used to match against assessment data.

---

### 3. Submission Form (Office Forms)  
- **Link:** [https://forms.office.com/r/Pq8dYPEGH4](https://forms.office.com/r/Pq8dYPEGH4)  
- **Use:**  
  - **Submission Portal:** This form is provided for you to submit your final project deliverables (demo URLs, API endpoint, GitHub repository, and one-page approach document).  
  - **Guidance Reference:** It also outlines the submission requirements for the assignment.

---

### 4. SHL Job Posting on LinkedIn  
- **Link:** [https://www.linkedin.com/jobs/view/research-engineer-ai-at-shl-4194768899/?originalSubdomain=in](https://www.linkedin.com/jobs/view/research-engineer-ai-at-shl-4194768899/?originalSubdomain=in)  
- **Use:**  
  - **Context & Positioning:** This posting provides context about the role you’re applying for (Research Intern – AI) at SHL.  
  - **Alignment with Business Needs:** It helps you understand the expectations and strategic focus of SHL, ensuring that your recommendation system addresses real-world hiring challenges.

---