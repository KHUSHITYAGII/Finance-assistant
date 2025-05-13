import streamlit as st
import requests
import json
import base64
from datetime import datetime
import pytz
import time
import os

# Service URLs - set these as environment variables in production
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")
VOICE_AGENT_URL = os.getenv("VOICE_AGENT_URL", "http://localhost:8006")

st.set_page_config(
    page_title="Financial Assistant",
    page_icon="💰",
    layout="wide",
)

# Custom CSS for a better looking app
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px 24px;
        border-radius: 12px;
        border: none;
    }
    .stTextInput>div>div>input {
        background-color: white;
        color: #333;
        border-radius: 12px;
        padding: 12px;
    }
    h1 {
        color: #2E6E9E;
    }
    .info-box {
        background-color: #e1f5fe;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .response-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("💰 Financial Assistant")

# Get current time in New York timezone (for market relevance)
ny_timezone = pytz.timezone('America/New_York')
current_time = datetime.now(ny_timezone)
st.markdown(f"<p>Current time in New York: {current_time.strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)

# Sample queries sidebar
st.sidebar.header("Sample Queries")
if st.sidebar.button("What's our risk exposure in Asia tech stocks today?"):
    st.session_state.query = "What's our risk exposure in Asia tech stocks today, and highlight any earnings surprises?"

# Input section
st.markdown("<div class='info-box'>Ask me about your financial portfolio, market updates, or specific stock information. I'm here to help!</div>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Enter your query:", value=st.session_state.get("query", ""), placeholder="Example: What's our risk exposure in Asia tech stocks today?")

# Response format selection
with col2:
    response_format = st.radio("Response format:", ("Text", "Voice"))

# Process query button
if st.button("Submit Query") and query:
    with st.spinner("Processing your request..."):
        try:
            # Call orchestrator service
            response = requests.post(
                f"{ORCHESTRATOR_URL}/process_query",
                json={"query": query, "response_format": response_format.lower()},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display the text response
                st.markdown("<div class='response-box'>", unsafe_allow_html=True)
                st.subheader("Response:")
                st.write(result.get("text", "No response generated."))
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Handle voice response if applicable
                if response_format == "Voice" and "voice_endpoint" in result:
                    st.markdown("<div class='response-box'>", unsafe_allow_html=True)
                    st.subheader("Voice Response:")
                    
                    # Get the TTS audio
                    voice_response = requests.post(
                        result["voice_endpoint"],
                        json=result["voice_payload"],
                        timeout=30
                    )
                    
                    if voice_response.status_code == 200:
                        # Create an audio player
                        audio_bytes = voice_response.content
                        st.audio(audio_bytes, format="audio/mp3")
                    else:
                        st.error("Failed to generate voice response.")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Display additional data if available
                if "analysis" in result:
                    st.markdown("<div class='response-box'>", unsafe_allow_html=True)
                    st.subheader("Analysis Details:")
                    st.json(result["analysis"])
                    st.markdown("</div>", unsafe_allow_html=True)
            
            else:
                st.error(f"Error: {response.text}")
        
        except Exception as e:
            st.error(f"Error processing request: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Financial Assistant - Powered by AI")