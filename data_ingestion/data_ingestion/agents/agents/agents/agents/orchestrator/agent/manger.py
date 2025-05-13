"""
Agent orchestrator that manages and routes requests between agents
"""

import os
from typing import Dict, Any, List, Optional
from loguru import logger
import asyncio
from datetime import datetime

from agents.api_agent import APIAgent
from agents.scraping_agent import ScrapingAgent
from agents.retriever_agent import RetrieverAgent
from agents.analysis_agent import AnalysisAgent
from agents.language_agent import LanguageAgent
from agents.voice_agent import VoiceAgent
from config.settings import settings
from config.prompts import Prompts


class Orchestrator:
    """Main orchestrator for managing and routing between agents"""
    
    def __init__(self):
        """Initialize the orchestrator and all agents"""
        logger.info("Initializing orchestrator...")
        
        # Initialize all agents
        self.api_agent = APIAgent()
        self.scraping_agent = ScrapingAgent()
        self.retriever_agent = RetrieverAgent()
        self.analysis_agent = AnalysisAgent()
        self.language_agent = LanguageAgent()
        self.voice_agent = VoiceAgent()
        
        # State management
        self.conversation_history = []
        self.initialized = True
        
        logger.info("Orchestrator initialized")
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and generate a response
        
        Args:
            query: User query string
            
        Returns:
            Dict containing response data
        """
        logger.info(f"Processing query: {query}")
        
        # Add query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Get relevant documents from retrieval agent
        retrieval_results = await self.retriever_agent.retrieve(query)
        
        # Check confidence level
        confidence = max([r.get("similarity", 0) for r in retrieval_results]) if retrieval_results else 0
        
        # If confidence is too low, try to extract key entities and retry
        if confidence < settings.CONFIDENCE_THRESHOLD and "asia tech" not in query.lower():
            # Extract entities and retry
            logger.info(f"Low confidence ({confidence}), refining query...")
            refined_query = f"{query} related to Asia tech stocks"
            retrieval_results = await self.retriever_agent.retrieve(refined_query)
            confidence = max([r.get("similarity", 0) for r in retrieval_results]) if retrieval_results else 0
        
        # Handle queries about Asia tech stocks specifically
        if "asia tech" in query.lower() or "asia" in query.lower() or "tech" in query.lower():
            # Get portfolio allocation
            portfolio_data = await self.api_agent.get_portfolio_allocation()
            
            # Get stock performance data
            stock_performance = await self.api_agent.get_stock_data()
            
            # Get earnings data
            earnings_data = await self.api_agent.get_earnings_data()
            
            # Get market news
            market_news = await self.scraping_agent.get_news()
            
            # Analyze earnings surprises
            earnings_analysis = await self.analysis_agent.analyze_earnings(earnings_data)
            
            # Analyze sentiment
            sentiment_analysis = await self.analysis_agent.analyze_sentiment(market_news)
            
            # Generate market brief
            brief = await self.language_agent.generate_market_brief(
                query,
                portfolio_data=portfolio_data,
                stock_performance=stock_performance,
                earnings_data=earnings_analysis,
                market_news=sentiment_analysis
            )
            
            # Add sources
            sources = [
                {"type": "portfolio", "content": portfolio_data},
                {"type": "market_data", "content": stock_performance},
                {"type": "earnings", "content": earnings_analysis},
                {"type": "news", "content": market_news[:3]}  # Include only the first 3 news items
            ]
            
            # Higher confidence for direct Asia tech queries
            if confidence < 0.8:
                confidence = 0.8
        else:
            # For other queries, use retrieval-based approach
            # Extract relevant information from retrieval results
            contexts = [r["text"] for r in retrieval_results]
            
            # Generate market brief based on retrieved information
            brief = await self.language_agent.generate_response(query, contexts)
            
            # Use retrieval results as sources
            sources = [
                {"type": "document", "content": r["text"], "metadata": r["metadata"]}
                for r in retrieval_results
            ]
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": brief})
        
        return {
            "brief": brief,
            "confidence": confidence,
            "sources": sources
        }


# Singleton instance
_orchestrator = None

def get_orchestrator() -> Orchestrator:
    """
    Get the singleton orchestrator instance
    
    Returns:
        Orchestrator instance
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator