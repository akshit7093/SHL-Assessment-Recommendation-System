import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load processed JSON data
json_file_path = 'shl_processed_analysis_specific.json'
with open(json_file_path, 'r', encoding='utf-8') as f:
    processed_data = json.load(f)

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract text data for embedding
texts = [item['extracted_text'] for item in processed_data if item['extracted_text']]

# Generate embeddings
embeddings = model.encode(texts)

# Convert embeddings to numpy array
embeddings_np = np.array(embeddings)

# Create a FAISS index
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(embeddings_np)

# Save the index to a file
faiss.write_index(index, 'shl_vector_index.idx')

print('Vector database created and saved successfully.')