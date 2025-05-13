import streamlit as st
import requests
import os
import time
import json
import base64
from datetime import datetime

# Configuration
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000")

# App title and styling
st.set_page_config(
    page_title="Finance Voice Assistant",
    page_icon="💹",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        font-size: 16px;
    }
    .info-box {
        background-color: #e7f3fe;
        border-left: 6px solid #2196F3;
        padding: 10px;
        margin-bottom: 15px;
    }
    .response-box {
        background-color: #e8f4f8;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stAudio {
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("🗣️ Finance Voice Assistant")

# Session state initialization
if 'responses' not in st.session_state:
    st.session_state.responses = []

# App header and description
st.markdown("""
<div class="info-box">
    <h3>Morning Market Brief</h3>
    <p>Ask about your portfolio's risk exposure, market updates, or specific stock information.</p>
    <p>Example: "What's our risk exposure in Asia tech stocks today, and highlight any earnings surprises?"</p>
</div>
""", unsafe_allow_html=True)

# Input section
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input("Enter your query:", value=st.session_state.get("query", ""), placeholder="Example: What's our risk exposure in Asia tech stocks today?")
    st.session_state.query = query

with col2:
    # Response format selection
    response_format = st.radio("Response format:", ("Text", "Voice"))

# Process query button
if st.button("Submit Query") and query:
    with st.spinner("Processing your request..."):
        try:
            # Call orchestrator service
            response = requests.post(
                f"{ORCHESTRATOR_URL}/process_query",
                json={
                    "query": query,
                    "response_format": response_format
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Add timestamp to response
                result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result["query"] = query
                
                # Add to session state
                st.session_state.responses.append(result)
                
                # Success message
                st.success("Query processed successfully!")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Failed to process query: {str(e)}")

# Display responses
if st.session_state.responses:
    st.markdown("### Recent Responses")
    
    for idx, resp in enumerate(reversed(st.session_state.responses)):
        with st.container():
            st.markdown(f"""
            <div class="response-box">
                <small>{resp.get('timestamp', 'N/A')}</small>
                <h4>Query: {resp.get('query', 'N/A')}</h4>
                <p>{resp.get('text_response', 'No response text')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display audio player if voice response
            if resp.get('format') == 'voice' and resp.get('audio_path'):
                audio_path = resp.get('audio_path')
                try:
                    # If the file is accessible directly
                    if os.path.exists(audio_path):
                        with open(audio_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/mp3')
                    else:
                        st.warning(f"Audio file not found: {audio_path}")
                except Exception as e:
                    st.error(f"Error playing audio: {str(e)}")
            
            st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px;">
    <p style="color: gray; font-size: 12px;">Finance Voice Assistant © 2023</p>
</div>
""", unsafe_allow_html=True)