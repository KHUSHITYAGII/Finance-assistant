"""
LLM prompt templates for various agents
"""

class Prompts:
    """Collection of prompt templates for different agents"""
    
    # System prompts
    SYSTEM_MARKET_BRIEF = """You are a financial analyst assistant that provides concise, accurate
market briefings based on real-time data. Focus on clarity, precision, and actionable insights.
Stick to facts from reliable sources and avoid speculation. Format your response to be clear
and easily digestible by busy financial professionals."""

    # Language Agent prompts
    MARKET_BRIEF_TEMPLATE = """Based on the following information, generate a concise market brief
that addresses the question: "{question}".

DATA SOURCES:
- Portfolio Allocation: {portfolio_data}
- Stock Performance: {stock_performance}
- Earnings Reports: {earnings_data}
- Market News: {market_news}

Your brief should:
1. Directly answer the question
2. Highlight key metrics and changes
3. Mention notable earnings surprises
4. Provide a brief sentiment assessment
5. Be conversational but professional
6. Be concise (100-150 words max)

Market Brief:"""

    EARNINGS_ANALYSIS_TEMPLATE = """Analyze the following earnings data and identify surprises 
(beats or misses) of 2% or more:

{earnings_data}

Format the response as:
- Company X: [beat/missed] by Y%
- Company Z: [beat/missed] by W%

Only include companies with surprises of 2% or more."""

    RISK_EXPOSURE_TEMPLATE = """Based on the following portfolio allocation data, analyze the 
risk exposure for {sector} stocks in {region}:

{portfolio_data}

Include:
1. Current allocation percentage
2. Change from previous day
3. Major concentration risks
4. Brief volatility assessment

Be quantitative and precise."""

    # Retriever Agent prompts
    QUERY_TRANSFORMATION_TEMPLATE = """Transform this user question into a search query optimized 
for retrieving relevant financial information:

User Question: {question}

The query should:
1. Focus on key entities (companies, regions, sectors)
2. Include relevant financial terms
3. Be concise yet specific
4. Remove conversational elements

Transformed Query:"""

    # Analysis Agent prompts
    SENTIMENT_ANALYSIS_TEMPLATE = """Analyze the sentiment in these market news snippets related to {region} {sector}:

{news_snippets}

Provide a single sentiment assessment on a scale:
- Strongly Bearish
- Bearish
- Slightly Bearish
- Neutral
- Slightly Bullish
- Bullish
- Strongly Bullish

Include a one-sentence rationale based on the dominant themes in the news."""

    # Fallback prompts
    CLARIFICATION_TEMPLATE = """I need more specific information to provide an accurate market brief. 
Could you please clarify:

{clarification_question}

This will help me give you more relevant information."""