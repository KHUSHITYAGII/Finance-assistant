# Multi-Agent Finance Assistant

A sophisticated multi-agent system that delivers finance market briefs via text and voice interfaces. This project includes API integrations, web scraping, document retrieval, analysis, and voice processing capabilities.

## Architecture

![Architecture Diagram](https://via.placeholder.com/800x500?text=Finance+Assistant+Architecture)

The system consists of the following components:

### Agents
- **API Agent**: Fetches financial data from Yahoo Finance and other sources
- **Scraping Agent**: Extracts information from financial websites and filings
- **Retriever Agent**: Implements RAG (Retrieval Augmented Generation) with FAISS
- **Analysis Agent**: Processes financial data to extract insights
- **Language Agent**: Generates natural language responses using LLMs
- **Voice Agent**: Handles speech-to-text and text-to-speech conversions

### Orchestration
The system uses FastAPI microservices for each agent with a central orchestrator that routes requests and manages the workflow.

## Setup & Deployment

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- OpenAI API key

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/finance-assistant.git
cd finance-assistant