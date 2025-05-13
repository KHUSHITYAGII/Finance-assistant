import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import os
from datetime import datetime
import json

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

logger = logging.getLogger(__name__)

class MarketBriefOutput(BaseModel):
    """Output schema for structured market briefs"""
    summary: str = Field(description="Brief summary of market conditions")
    risk_assessment: str = Field(description="Risk assessment for Asia tech stocks")
    earnings_highlights: str = Field(description="Key earnings surprises")
    sentiment: str = Field(description="Market sentiment analysis")
    recommendations: Optional[List[str]] = Field(description="Optional recommendations")

class LanguageAgent:
    """
    Language Agent responsible for LLM integration, prompt templating,
    response synthesis from multiple data sources, and narrative generation.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the Language Agent with configuration.
        
        Args:
            config: Configuration dictionary for the agent including API keys and model settings
        """
        self.config = config or {}
        self.model_name = self.config.get("model_name", "gpt-3.5-turbo")
        self.temperature = self.config.get("temperature", 0.2)
        self.max_tokens = self.config.get("max_tokens", 1000)
        
        # Initialize the LLM client based on provider
        self._initialize_llm()
        
        # Initialize retrievers and knowledge bases
        self.retrievers = self.config.get("retrievers", {})
        
        logger.info(f"Language Agent initialized with model: {self.model_name}")
        
    def _initialize_llm(self):
        """Initialize the LLM client based on provider configuration"""
        provider = self.config.get("provider", "openai").lower()
        
        try:
            if provider == "openai":
                self.llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=self.config.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
                )
                self.provider = "openai"
                
            elif provider == "anthropic":
                anthropic_key = self.config.get("anthropic_api_key", os.getenv("ANTHROPIC_API_KEY"))
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                self.provider = "anthropic"
                
            else:
                logger.warning(f"Unknown provider: {provider}, defaulting to OpenAI")
                self.llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=self.config.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
                )
                self.provider = "openai"
                
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            # Fallback to simple completion function for testing
            self.provider = "mock"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_response(self, prompt: str, system_prompt: str = None, 
                         output_parser: Any = None) -> Union[str, Dict]:
        """
        Generate a response from the LLM using the provided prompt.
        
        Args:
            prompt: User prompt or query
            system_prompt: Optional system prompt to guide model behavior
            output_parser: Optional parser for structured output
            
        Returns:
            String response from the LLM or parsed structured output
        """
        try:
            if self.provider == "openai":
                chat_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt or "You are a helpful financial assistant."),
                    ("human", prompt)
                ])
                
                if output_parser:
                    chain = chat_prompt | self.llm | output_parser
                else:
                    chain = chat_prompt | self.llm | StrOutputParser()
                
                return chain.invoke({})
                
            elif self.provider == "anthropic":
                message = self.anthropic_client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_prompt or "You are a helpful financial assistant.",
                    messages=[{"role": "user", "content": prompt}]
                )
                
                response = message.content[0].text
                
                # Handle output parsing if needed
                if output_parser:
                    try:
                        return output_parser.parse(response)
                    except Exception as e:
                        logger.error(f"Error parsing output: {str(e)}")
                        return response
                else:
                    return response
                
            elif self.provider == "mock":
                logger.warning("Using mock LLM provider")
                mock_response = f"Mock response to: {prompt[:30]}..."
                
                # Handle output parsing if needed
                if output_parser:
                    try:
                        # Create a mock structured response for JsonOutputParser
                        if isinstance(output_parser, JsonOutputParser):
                            return {
                                "summary": "Mock market summary",
                                "risk_assessment": "Mock risk assessment",
                                "earnings_highlights": "Mock earnings highlights",
                                "sentiment": "Mock sentiment analysis"
                            }
                        return mock_response
                    except Exception:
                        return mock_response
                return mock_response
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I'm having trouble generating a response right now."
    
    def retrieve_relevant_knowledge(self, query: str, sources: List[str] = None) -> List[Dict]:
        """
        Retrieve relevant knowledge from vector stores and knowledge bases.
        
        Args:
            query: User query or context for retrieval
            sources: Optional list of specific sources to query
            
        Returns:
            List of relevant document chunks with metadata
        """
        results = []
        
        try:
            # Use configured retrievers if available
            if not sources:
                sources = list(self.retrievers.keys())
                
            for source in sources:
                retriever = self.retrievers.get(source)
                if retriever:
                    # Execute retrieval and add results
                    source_results = retriever.get_relevant_documents(query)
                    for doc in source_results:
                        results.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "source": source
                        })
            
            # Sort by relevance if available in metadata
            results = sorted(results, 
                         key=lambda x: x.get("metadata", {}).get("relevance_score", 0), 
                         reverse=True)
                         
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {str(e)}")
            return []
    
    def create_market_brief(self, 
                          risk_data: Dict, 
                          earnings_data: Dict, 
                          sentiment_data: Dict,
                          portfolio_data: Dict = None,
                          additional_context: Dict = None,
                          query: str = None,
                          structured_output: bool = False) -> Union[str, Dict]:
        """
        Create a comprehensive market brief based on financial data.
        
        Args:
            risk_data: Risk exposure analysis
            earnings_data: Earnings analysis
            sentiment_data: Market sentiment evaluation
            portfolio_data: Optional portfolio allocation data
            additional_context: Any additional context from retrievers or other agents
            query: Original user query for context
            structured_output: Whether to return structured JSON output
            
        Returns:
            Natural language market brief or structured JSON output
        """
        logger.info("Creating market brief")
        
        try:
            # Build context from all available data
            context_parts = []
            
            # Add risk exposure data
            if risk_data:
                exposure = risk_data.get("exposure_percentage", 0)
                prev_exposure = risk_data.get("previous_exposure_percentage", 0)
                change = risk_data.get("change", 0)
                
                context_parts.append(
                    f"Risk Exposure Data:\n"
                    f"- Current exposure to Asia tech stocks: {exposure}% of AUM\n"
                    f"- Previous exposure: {prev_exposure}% (change: {change}%)\n"
                    f"- Top risk factors: {', '.join(risk_data.get('risk_factors', ['N/A']))}\n"
                )
            
            # Add earnings data
            if earnings_data:
                beats = earnings_data.get("top_beats", [])
                misses = earnings_data.get("top_misses", [])
                
                earnings_context = f"Earnings Data:\n"
                
                for company in beats[:3]:
                    ticker = company.get("ticker", "")
                    surprise = company.get("surprise_percent", 0)
                    earnings_context += f"- {ticker} beat estimates by {surprise}%\n"
                    
                for company in misses[:3]:
                    ticker = company.get("ticker", "")
                    surprise = company.get("surprise_percent", 0)
                    earnings_context += f"- {ticker} missed estimates by {abs(surprise)}%\n"
                    
                context_parts.append(earnings_context)
            
            # Add sentiment data
            if sentiment_data:
                sentiment = sentiment_data.get("sentiment_category", "neutral")
                tilt = sentiment_data.get("tilt", "neutral")
                factors = sentiment_data.get("driving_factors", [])
                
                sentiment_context = (
                    f"Market Sentiment Data:\n"
                    f"- Overall sentiment: {sentiment}\n"
                    f"- Sentiment tilt: {tilt}\n"
                )
                
                if factors:
                    sentiment_context += "- Key factors:\n"
                    for factor in factors:
                        sentiment_context += f"  * {factor}\n"
                        
                context_parts.append(sentiment_context)
            
            # Add portfolio data if available
            if portfolio_data:
                portfolio_context = "Portfolio Allocation:\n"
                
                region_sectors = portfolio_data.get("region_sector_allocation", {})
                for region_sector, allocation in region_sectors.items():
                    if "Asia-tech" in region_sector:
                        portfolio_context += f"- {region_sector}: {allocation}%\n"
                        
                context_parts.append(portfolio_context)
            
            # Add additional context from retrievers or other agents
            if additional_context:
                retrieved_documents = additional_context.get("retrieved_documents", [])
                if retrieved_documents:
                    relevant_info = "Additional Relevant Information:\n"
                    for i, doc in enumerate(retrieved_documents[:3]):
                        source = doc.get("source", "Unknown")
                        content = doc.get("content", "")
                        relevant_info += f"- From {source}: {content[:150]}...\n"
                    context_parts.append(relevant_info)
                
                news_items = additional_context.get("news_items", [])
                if news_items:
                    news_info = "Recent News:\n"
                    for i, news in enumerate(news_items[:3]):
                        title = news.get("title", "")
                        news_info += f"- {title}\n"
                    context_parts.append(news_info)
            
            # Combine all context parts
            full_context = "\n".join(context_parts)
            
            # Create the system prompt
            system_prompt = """
            You are a precise, clear financial analyst creating a morning market brief.
            Your task is to synthesize financial data into a concise, insightful brief.
            Focus on the key highlights without unnecessary explanations.
            Use a professional, authoritative tone suitable for financial professionals.
            Be direct and get straight to the point. Limit your response to 3-4 sentences maximum.
            """
            
            # Create the user prompt
            user_prompt = f"""
            Based on the following financial data, create a brief morning market update focused specifically 
            on Asia tech stocks exposure and earnings surprises:
            
            {full_context}
            
            The portfolio manager asked: "{query or 'What\'s our risk exposure in Asia tech stocks today, and highlight any earnings surprises?'}"
            
            Provide a concise response focusing on:
            1. Current Asia tech allocation as percentage of AUM and how it changed
            2. Any significant earnings surprises (positive or negative)
            3. Regional market sentiment with key driving factors
            
            Respond as if speaking directly to the portfolio manager in a professional, concise manner.
            """
            
            # Configure output format based on structured_output flag
            if structured_output:
                # Update system prompt for structured output
                system_prompt += """
                Format your response as a structured JSON object with the following fields:
                - summary: A concise 1-2 sentence overall market brief
                - risk_assessment: Brief assessment of Asia tech exposure
                - earnings_highlights: Key earnings surprises
                - sentiment: Summary of market sentiment
                - recommendations: Optional list of brief recommendations
                """
                
                user_prompt += """
                Please format your response as a JSON object with fields for summary, risk_assessment,
                earnings_highlights, sentiment, and optionally recommendations.
                """
                
                # Use JsonOutputParser for structured output
                json_parser = JsonOutputParser(pydantic_object=MarketBriefOutput)
                response = self.generate_response(user_prompt, system_prompt, json_parser)
                
                return response
            else:
                # Generate the market brief as plain text
                response = self.generate_response(user_prompt, system_prompt)
                
                return response
            
        except Exception as e:
            logger.error(f"Error creating market brief: {str(e)}")
            error_msg = "I apologize, but I'm unable to generate a market brief at this moment due to a technical issue."
            
            if structured_output:
                return {
                    "summary": error_msg,
                    "risk_assessment": "",
                    "earnings_highlights": "",
                    "sentiment": "",
                    "recommendations": []
                }
            return error_msg
    
    def personalize_response(self, market_brief: Union[str, Dict], 
                           user_preferences: Dict = None) -> Union[str, Dict]:
        """
        Personalize the market brief based on user preferences and history.
        
        Args:
            market_brief: Generated market brief (string or structured dict)
            user_preferences: User preferences and settings
            
        Returns:
            Personalized market brief
        """
        if not user_preferences:
            return market_brief
            
        try:
            # Handle structured vs. unstructured briefs
            is_structured = isinstance(market_brief, dict)
            
            # Extract user preferences
            focus_areas = user_preferences.get("focus_areas", [])
            communication_style = user_preferences.get("communication_style", "neutral")
            detail_level = user_preferences.get("detail_level", "medium")
            prioritized_companies = user_preferences.get("prioritized_companies", [])
            
            # Create personalization prompt
            system_prompt = """
            You are a personalization expert for financial briefings.
            Adapt the provided market brief to match the user's preferences without changing the core information.
            Maintain professionalism while adjusting the style and emphasis as requested.
            """
            
            if is_structured:
                # Convert structured brief to text for personalization
                brief_text = f"""
                Summary: {market_brief.get('summary', '')}
                Risk Assessment: {market_brief.get('risk_assessment', '')}
                Earnings Highlights: {market_brief.get('earnings_highlights', '')}
                Sentiment: {market_brief.get('sentiment', '')}
                """
                
                if 'recommendations' in market_brief and market_brief['recommendations']:
                    brief_text += f"Recommendations: {', '.join(market_brief['recommendations'])}"
            else:
                brief_text = market_brief
            
            user_prompt = f"""
            Original market brief:
            "{brief_text}"
            
            User preferences:
            - Focus areas: {', '.join(focus_areas) if focus_areas else 'None specified'}
            - Communication style: {communication_style}
            - Detail level: {detail_level}
            - Prioritized companies: {', '.join(prioritized_companies) if prioritized_companies else 'None specified'}
            
            Rewrite the market brief to match these preferences, keeping the same core information
            but adjusting emphasis, tone, and detail level accordingly. The response should remain
            brief and concise regardless of detail level. If specific companies are prioritized,
            highlight information about them when available.
            """
            
            # If structured output, ensure we return in the same format
            if is_structured:
                system_prompt += """
                Format your response as a structured JSON object with the following fields:
                - summary: A concise 1-2 sentence overall market brief
                - risk_assessment: Brief assessment of Asia tech exposure
                - earnings_highlights: Key earnings surprises
                - sentiment: Summary of market sentiment
                - recommendations: Optional list of brief recommendations
                """
                
                user_prompt += """
                Please format your response as a JSON object with fields for summary, risk_assessment,
                earnings_highlights, sentiment, and optionally recommendations.
                """
                
                # Use JsonOutputParser for structured output
                json_parser = JsonOutputParser(pydantic_object=MarketBriefOutput)
                personalized_brief = self.generate_response(user_prompt, system_prompt, json_parser)
            else:
                personalized_brief = self.generate_response(user_prompt, system_prompt)
            
            return personalized_brief
            
        except Exception as e:
            logger.error(f"Error personalizing response: {str(e)}")
            return market_brief  # Fall back to original brief if personalization fails
    
    def generate_followup_suggestions(self, query: str, market_brief: Union[str, Dict]) -> List[str]:
        """
        Generate relevant follow-up questions based on the market brief.
        
        Args:
            query: Original user query
            market_brief: Generated market brief (string or structured dict)
            
        Returns:
            List of follow-up question suggestions
        """
        try:
            # Handle structured vs. unstructured briefs
            if isinstance(market_brief, dict):
                # Convert structured brief to text for analysis
                brief_text = f"""
                Summary: {market_brief.get('summary', '')}
                Risk Assessment: {market_brief.get('risk_assessment', '')}
                Earnings Highlights: {market_brief.get('earnings_highlights', '')}
                Sentiment: {market_brief.get('sentiment', '')}
                """
                
                if 'recommendations' in market_brief and market_brief['recommendations']:
                    brief_text += f"Recommendations: {', '.join(market_brief['recommendations'])}"
            else:
                brief_text = market_brief
                
            system_prompt = """
            You are a financial advisor assistant helping to generate relevant follow-up questions.
            Based on the original query and market brief, suggest 3 helpful follow-up questions
            that the user might want to ask next. These should be natural extensions of the conversation.
            Provide just the questions without any additional text or explanations.
            Each question should be concise and focused.
            """
            
            user_prompt = f"""
            Original query: "{query}"
            
            Market brief: "{brief_text}"
            
            Generate 3 relevant follow-up questions that would naturally continue this conversation.
            Format each as a standalone question on its own line without numbering or bullets.
            Focus on questions that would help the portfolio manager make informed decisions about
            Asia tech stocks, earnings impacts, or risk adjustments.
            """
            
            response = self.generate_response(user_prompt, system_prompt)
            
            # Parse response into individual questions
            questions = [q.strip() for q in response.split('\n') if q.strip()]
            
            return questions[:3]  # Ensure we return at most 3 questions
            
        except Exception as e:
            logger.error(f"Error generating follow-up suggestions: {str(e)}")
            return [
                "Can you provide more details about specific tech stocks in our portfolio?",
                "What's the expected market volatility for today?",
                "Are there any market events today that might impact our positions?"
            ]  # Default questions if generation fails
            
    def convert_brief_to_speech_optimized(self, market_brief: Union[str, Dict]) -> str:
        """
        Convert market brief to speech-optimized format for TTS engines.
        
        Args:
            market_brief: Generated market brief (string or structured dict)
            
        Returns:
            Speech-optimized version of the market brief
        """
        try:
            # Handle structured vs. unstructured briefs
            if isinstance(market_brief, dict):
                # Convert structured brief to text format
                brief_text = market_brief.get('summary', '')
                
                # Add additional details if they exist
                if market_brief.get('risk_assessment'):
                    brief_text += " " + market_brief['risk_assessment']
                if market_brief.get('earnings_highlights'):
                    brief_text += " " + market_brief['earnings_highlights']
                if market_brief.get('sentiment'):
                    brief_text += " " + market_brief['sentiment']
            else:
                brief_text = market_brief
                
            system_prompt = """
            You are an expert in converting written financial briefs to spoken format.
            Your task is to adapt the written text to be more suitable for text-to-speech (TTS) delivery.
            Consider:
            1. Adding brief pauses where appropriate (<break> tags can be used)
            2. Spelling out acronyms or adding clarification where helpful
            3. Making numerical data more speech-friendly (percentages, currency values)
            4. Removing visual formatting elements
            Keep the content concise and complete, just optimize for speech delivery.
            """
            
            user_prompt = f"""
            Convert this financial market brief to a speech-optimized format for a text-to-speech engine:
            
            "{brief_text}"
            
            The output should maintain all key information but be easier to follow when heard rather than read.
            You can use SSML tags like <break strength="weak"/> for short pauses or <break strength="medium"/> 
            for longer pauses if needed to improve clarity.
            """
            
            speech_optimized = self.generate_response(user_prompt, system_prompt)
            
            return speech_optimized
            
        except Exception as e:
            logger.error(f"Error converting brief to speech format: {str(e)}")
            return market_brief if isinstance(market_brief, str) else json.dumps(market_brief)
    
    def analyze_confidence(self, market_brief: Union[str, Dict], 
                         source_data: Dict) -> Tuple[float, List[str]]:
        """
        Analyze the confidence level of the generated market brief based on source data.
        
        Args:
            market_brief: Generated market brief
            source_data: Source data used to generate the brief
            
        Returns:
            Tuple of (confidence_score, factors_affecting_confidence)
        """
        try:
            # Extract text from market brief if structured
            if isinstance(market_brief, dict):
                brief_text = f"""
                Summary: {market_brief.get('summary', '')}
                Risk Assessment: {market_brief.get('risk_assessment', '')}
                Earnings Highlights: {market_brief.get('earnings_highlights', '')}
                Sentiment: {market_brief.get('sentiment', '')}
                """
            else:
                brief_text = market_brief
                
            # Create evaluation prompt
            system_prompt = """
            You are an objective evaluator of financial information quality.
            Your task is to assess the confidence level in a generated market brief based on the source data.
            Analyze completeness, accuracy, and potential gaps or uncertainties.
            Score confidence on a scale from 0.0 to 1.0, where:
            - 0.0-0.3: Low confidence (significant gaps, contradictions, or unclear data)
            - 0.4-0.7: Medium confidence (some gaps or moderate uncertainty)
            - 0.8-1.0: High confidence (comprehensive, consistent data with minimal gaps)
            """
            
            # Prepare source data summary
            source_summary = []
            if 'risk_data' in source_data and source_data['risk_data']:
                source_summary.append(f"Risk data: {len(source_data['risk_data'])} data points")
            if 'earnings_data' in source_data and source_data['earnings_data']:
                earnings_count = len(source_data['earnings_data'].get('top_beats', [])) + \
                               len(source_data['earnings_data'].get('top_misses', []))
                source_summary.append(f"Earnings data: {earnings_count} companies")
            if 'sentiment_data' in source_data and source_data['sentiment_data']:
                source_summary.append(f"Sentiment data: {len(source_data['sentiment_data'].get('driving_factors', []))} factors")
                
            source_summary_text = "\n".join(source_summary)
            
            user_prompt = f"""
            Market brief to evaluate:
            "{brief_text}"
            
            Source data overview:
            {source_summary_text}
            
            Evaluate the confidence level of this market brief based on the source data.
            Return your response as a JSON object with these fields:
            - confidence_score: A float between 0.0 and 1.0
            - factors: A list of strings describing factors affecting confidence
            
            For example:
            {{
                "confidence_score": 0.85,
                "factors": ["Comprehensive earnings data", "Limited sentiment analysis sources"]
            }}
            """
            
            # Use JSON output parser
            class ConfidenceOutput(BaseModel):
                confidence_score: float = Field(description="Confidence score between 0.0 and 1.0")
                factors: List[str] = Field(description="Factors affecting confidence")
                
            json_parser = JsonOutputParser(pydantic_object=ConfidenceOutput)
            result = self.generate_response(user_prompt, system_prompt, json_parser)
            
            # Extract values from result
            confidence_score = result.get('confidence_score', 0.5)
            factors = result.get('factors', ["Insufficient data to assess confidence"])
            
            return confidence_score, factors
            
        except Exception as e:
            logger.error(f"Error analyzing confidence: {str(e)}")
            return 0.5, ["Error in confidence analysis"]