import streamlit as st
import requests

# Set the title of the app
st.title("Assessment Recommendation System")

# Add a description for the user
st.markdown("""
This app recommends assessments based on your query. Enter a job description or any relevant text, and get tailored recommendations.
""")

# Input field for the user's query
query = st.text_area("Enter your query (e.g., job description):", height=150)

# Button to trigger the recommendation process
if st.button("Get Recommendations"):
    if not query.strip():
        st.error("Query cannot be empty. Please enter a valid query.")
    else:
        # Display a spinner while the request is being processed
        with st.spinner("Fetching recommendations..."):
            try:
                # Send the query to the FastAPI backend
                response = requests.post(
                    "http://127.0.0.1:5000/recommend",  # Replace with your API URL if hosted elsewhere
                    json={"query": query}
                )

                # Check if the request was successful
                if response.status_code == 200:
                    data = response.json()
                    recommendations = data.get("recommendations", [])

                    if recommendations and recommendations[0].get("Assessment_Name") == "Error":
                        # Handle error response from the backend
                        st.error("An error occurred while processing your request. Please try again later.")
                    elif not recommendations:
                        st.warning("No recommendations found for your query.")
                    else:
                        # Display the recommendations
                        st.success(f"Found {len(recommendations)} recommendations:")
                        for idx, rec in enumerate(recommendations, start=1):
                            st.subheader(f"Recommendation {idx}: {rec['Assessment_Name']}")
                            if '/view/' in rec['URL']:
                                st.markdown(f"**Assessment URL (Primary):** [{rec['URL']}]({rec['URL']})")
                            else:
                                st.markdown(f"**Catalog URL (Secondary):** [{rec['URL']}]({rec['URL']})")
                            st.markdown(f"**Summary:** {rec['Summary']}")
                            st.markdown(f"**Key Features:** {', '.join(rec['Key_Features'])}")
                            st.markdown(f"**Duration:** {rec.get('Duration', 'Not specified')}")
                            st.markdown(f"**Remote Testing Available:** {'Yes' if rec.get('Remote_Testing', False) else 'No'}")
                            st.markdown(f"**Adaptive/IRT Support:** {'Yes' if rec.get('Adaptive_IRT', False) else 'No'}")
                            st.markdown(f"**Test Type:** {rec.get('Test_Type', 'Not specified')}")
                            st.markdown(f"**Similarity Score:** {rec.get('Similarity_Score', 0):.2f}")
                            st.markdown("---")
                else:
                    # Handle unexpected HTTP errors
                    st.error(f"Unexpected error: {response.status_code} - {response.text}")

            except Exception as e:
                # Handle connection or other unexpected errors
                st.error(f"Failed to connect to the server. Error: {str(e)}")