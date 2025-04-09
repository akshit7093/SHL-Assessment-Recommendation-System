# streamlit_app.py
import streamlit as st
import subprocess
import requests
import time
import os



# Streamlit App
st.title("Assessment Recommendation System")
st.markdown("""
This app recommends assessments based on your query. Enter a job description or any relevant text, and get tailored recommendations.
""")

query = st.text_area("Enter your query (e.g., job description):", height=150)

if st.button("Get Recommendations"):
    if not query.strip():
        st.error("Query cannot be empty. Please enter a valid query.")
    else:
        with st.spinner("Fetching recommendations..."):
            try:
                response = requests.post(
                    "https://akshit7093-shl.hf.space/recommend",  # backend port
                    json={"query": query}
                )

                if response.status_code == 200:
                    data = response.json()
                    recommendations = data.get("recommended_assessments", [])

                    if not recommendations:
                        st.warning("No recommendations found for your query.")
                    else:
                        st.success(f"Found {len(recommendations)} recommendations:")
                        for idx, rec in enumerate(recommendations, start=1):
                            st.subheader(f"Recommendation {idx}")
                            st.markdown(f"**URL:** [{rec['url']}]({rec['url']})")
                            st.markdown(f"**Description:** {rec['description']}")
                            st.markdown(f"**Duration:** {rec['duration']} minutes")
                            st.markdown(f"**Remote Support:** {rec['remote_support']}")
                            st.markdown(f"**Adaptive Support:** {rec['adaptive_support']}")
                            st.markdown(f"**Test Types:** {', '.join(rec['test_type'])}")
                            st.markdown("---")
                else:
                    st.error(f"Unexpected error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to the server. Error: {str(e)}")
