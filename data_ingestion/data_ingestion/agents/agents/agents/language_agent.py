import os
import logging
import requests
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

class LanguageAgent:
    def __init__(self, openai_api_key=None):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model="gpt-3.5-turbo",
            temperature=0.1
        )
    
    def generate_market_brief(self, query, analysis_results):
        """
        Generate a concise market brief based on analysis results
        """
        # Extract relevant data from analysis results
        exposure_analysis = analysis_results.get("exposure_analysis", {})
        earnings_analysis = analysis_results.get("earnings_analysis", {})
        sentiment_analysis = analysis_results.get("sentiment_analysis", {})
        
        # Prepare context for the LLM
        context = {
            "query": query,
            "asia_tech_exposure": exposure_analysis.get("asia_tech_exposure", "N/A"),
            "asia_tech_change": exposure_analysis.get("asia_tech_change", "N/A"),
            "positive_surprises": [
                f"{item.get('symbol')}: beat by {item.get('surprisePercentage', 0):.1f}%"
                for item in earnings_analysis.get("positive_surprises", [])
            ],
            "negative_surprises": [
                f"{item.get('symbol')}: missed by {abs(item.get('surprisePercentage', 0)):.1f}%"
                for item in earnings_analysis.get("negative_surprises", [])
            ],
            "overall_sentiment": sentiment_analysis.get("overall_sentiment", "neutral"),
            "interest_rates": sentiment_analysis.get("sentiment_factors", {}).get("interest_rates", "neutral"),
            "key_trends": [
                trend.get("headline") for trend in sentiment_analysis.get("key_trends", [])[:3]
            ]
        }
        
        # Create prompt template
        prompt_template = """
        You are a professional financial analyst providing a concise morning market brief.
        Answer the following query clearly and directly: "{query}"
        
        Here's the relevant information:
        - Asia tech allocation: {asia_tech_exposure}% of AUM (change from yesterday: {asia_tech_change}%)
        - Earnings surprises (positive): {positive_surprises}
        - Earnings surprises (negative): {negative_surprises}
        - Market sentiment: {overall_sentiment}
        - Interest rate impact: {interest_rates}
        - Key trends: {key_trends}
        
        Provide a concise and professional brief (2-3 sentences) that directly answers the query.
        Focus on the numbers and key insights without unnecessary explanations.
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            result = chain.run(**context)
            return result.strip()
        except Exception as e:
            self.logger.error(f"Error generating market brief: {e}")
            return "Unable to generate market brief due to an error."
    
    def process_query(self, query, analysis_results):
        """Process a natural language query using a language model"""
        response_data = {}
        
        try:
            market_brief = self.generate_market_brief(query, analysis_results)
            response_data["market_brief"] = market_brief
        except Exception as e:
            self.logger.error(f"Error in language agent: {e}")
            response_data["error"] = str(e)
        
        return response_data

# Example usage
if __name__ == "__main__":
    agent = LanguageAgent()
    
    # Example analysis results
    analysis_results = {
        "exposure_analysis": {
            "asia_tech_exposure": 22.0,
            "asia_tech_change": 4.0
        },
        "earnings_analysis": {
            "positive_surprises": [
                {"symbol": "TSM", "surprisePercentage": 4.0}
            ],
            "negative_surprises": [
                {"symbol": "005930.KS", "surprisePercentage": -2.0}
            ]
        },
        "sentiment_analysis": {
            "overall_sentiment": "cautiously optimistic",
            "sentiment_factors": {
                "interest_rates": "negative"
            },
            "key_trends": [
                {"headline": "TSMC shares rise on strong earnings"},
                {"headline": "Samsung misses expectations"},
                {"headline": "Bond yields rise to 4.5%"}
            ]
        }
    }
    
    result = agent.process_query(
        "What's our risk exposure in Asia tech stocks today, and highlight any earnings surprises?",
        analysis_results
    )
    print(result)