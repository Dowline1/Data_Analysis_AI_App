"""
Stock Market Recommendations Tool - External Web Search Integration

Uses Tavily Search API to find and recommend growth stocks based on current
market analysis and expert insights. This demonstrates external API integration.
"""

import os
from typing import Optional
from tavily import TavilyClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class StockRecommender:
    """
    External API tool for researching and recommending growth stocks.
    
    Uses Tavily Search API to find current market analysis, expert recommendations,
    and growth stock opportunities with detailed justifications.
    """
    
    def __init__(self):
        """Initialize the Tavily search client."""
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            logger.warning("TAVILY_API_KEY not found in environment. Tool will not function.")
            self.client = None
        else:
            self.client = TavilyClient(api_key=api_key)
            logger.info("Tavily search client initialized for stock recommendations")
    
    def get_growth_stocks(
        self, 
        market: str = "US",
        sector: Optional[str] = None
    ) -> str:
        """
        Search for top growth stock recommendations based on current market analysis.
        
        Args:
            market: Target market (e.g., 'US', 'Europe', 'Global')
            sector: Optional sector focus (e.g., 'Technology', 'Healthcare')
            
        Returns:
            Formatted string with top 10 growth stock recommendations and justifications
        """
        if not self.client:
            return "Tavily API key not configured. Cannot search for stock recommendations."
        
        try:
            # Build specific query for individual company stocks (not sector overviews)
            if sector:
                query = f"best individual {sector} stocks 2026 {market} company ticker buy recommendation"
            else:
                query = f"top individual growth stocks 2026 {market} company ticker symbol buy recommendation"
            
            logger.info(f"Searching for growth stocks: {query}")
            
            # Get more results initially, we'll filter to valid stocks only
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=10,  # Get more to filter down to 5 valid ones
                topic="finance",
                include_answer=True
            )
            
            # Format results
            if not response or "results" not in response:
                return "No stock recommendations found. Please try again."
            
            results = response["results"]
            
            # Filter results to only include those with identifiable stock tickers
            valid_stocks = []
            
            for result in results:
                title = result.get("title", "")
                content = result.get("content", "")
                
                import re
                
                # Try to extract a valid ticker (2-5 uppercase letters)
                ticker = None
                
                # Method 1: Parentheses format (AAPL)
                ticker_match = re.search(r'\(([A-Z]{2,5})\)', title)
                if ticker_match:
                    ticker = ticker_match.group(1)
                
                # Method 2: Colon format AAPL:
                if not ticker:
                    ticker_match = re.search(r'\b([A-Z]{2,5}):', title)
                    if ticker_match:
                        ticker = ticker_match.group(1)
                
                # Method 3: Dash format AAPL -
                if not ticker:
                    ticker_match = re.search(r'\b([A-Z]{2,5})\s*[-–—]', title)
                    if ticker_match:
                        ticker = ticker_match.group(1)
                
                # Method 4: Look in content for ticker patterns
                if not ticker:
                    content_match = re.search(r'(?:ticker|symbol|stock).*?([A-Z]{2,5})\b', content[:300], re.IGNORECASE)
                    if content_match:
                        potential = content_match.group(1)
                        exclude = ['US', 'UK', 'AI', 'ETF', 'CEO', 'IPO', 'NYSE', 'NASDAQ', 'THE', 'AND']
                        if potential not in exclude:
                            ticker = potential
                
                # Only include if we found a ticker (skip sector overview articles)
                if ticker:
                    valid_stocks.append(result)
                    logger.info(f"Valid stock found with ticker: {ticker}")
                else:
                    logger.info(f"Skipping result without ticker: {title[:50]}")
                
                # Stop once we have 5 valid stocks
                if len(valid_stocks) >= 5:
                    break
            
            if not valid_stocks:
                return f"Could not find specific company stock recommendations for {market} market. Please try a different search."
            
            output = f"📈 **Top Growth Stock Recommendations - {market} Market**\n\n"
            
            # Add AI summary if available
            if "answer" in response and response["answer"]:
                output += f"**Market Overview:**\n{response['answer']}\n\n"
                output += "---\n\n"
            
            output += "**Recommended Growth Stocks:**\n\n"
            
            for i, result in enumerate(valid_stocks, 1):
                title = result.get("title", "Unknown Stock")
                url = result.get("url", "")
                content = result.get("content", "No analysis available")
                
                # Extract stock information with validation
                import re
                
                # Method 1: Look for explicit ticker patterns like (AAPL) or AAPL:
                ticker_pattern1 = re.search(r'\(([A-Z]{1,5})\)', title)
                ticker_pattern2 = re.search(r'\b([A-Z]{2,5}):', title)
                ticker_pattern3 = re.search(r'\b([A-Z]{2,5})\s*[-–—]', title)
                
                ticker = None
                if ticker_pattern1:
                    ticker = ticker_pattern1.group(1)
                elif ticker_pattern2:
                    ticker = ticker_pattern2.group(1)
                elif ticker_pattern3:
                    ticker = ticker_pattern3.group(1)
                else:
                    # Method 2: Find standalone capital letter sequences
                    potential_tickers = re.findall(r'\b([A-Z]{2,5})\b', title)
                    # Filter out common non-ticker words
                    exclude = ['US', 'UK', 'AI', 'ETF', 'CEO', 'IPO', 'NYSE', 'NASDAQ', 'TOP', 'NEW', 'BEST', 'THE']
                    tickers = [t for t in potential_tickers if t not in exclude]
                    ticker = tickers[0] if tickers else None
                
                # Extract company name (text before ticker or first 50 chars)
                if ticker:
                    # Get text before the ticker
                    company_match = re.match(rf'^(.*?)[\(\[]?{ticker}', title)
                    company = company_match.group(1).strip(' :-–—') if company_match else ""
                else:
                    # Try to get company name from start of title
                    company_match = re.match(r'^([A-Z][a-zA-Z\s&\.]+?)(?:\s*[-:–—(]|$)', title)
                    company = company_match.group(1).strip() if company_match else ""
                
                # Validation: Ensure we have at least ticker OR company name
                if not ticker and not company:
                    # Fallback: extract from content
                    content_ticker = re.search(r'\b([A-Z]{2,5})\b', content[:200])
                    if content_ticker:
                        exclude = ['US', 'UK', 'AI', 'ETF', 'CEO', 'IPO', 'NYSE', 'NASDAQ', 'THE', 'AND', 'FOR', 'WITH']
                        potential = content_ticker.group(1)
                        if potential not in exclude:
                            ticker = potential
                
                # Build final stock identifier with validation
                if ticker and company:
                    stock_name = f"{company} ({ticker})"
                elif ticker:
                    stock_name = ticker
                elif company:
                    stock_name = company
                else:
                    # Last resort: use first meaningful part of title
                    clean_title = re.sub(r'[\[\(].*?[\]\)]', '', title).strip()
                    stock_name = clean_title[:60]
                
                logger.info(f"Extracted stock: {stock_name} from title: {title}")
                
                # Create focused summary emphasizing investment rationale
                investment_keywords = ['growth', 'revenue', 'earnings', 'profit', 'margin', 'market', 
                                      'forecast', 'outlook', 'expected', 'increase', 'expansion', 
                                      'innovation', 'leader', 'competitive', 'strong', 'performance']
                
                # Find sentences with investment keywords
                sentences = content.split('.')
                relevant_sentences = []
                for sentence in sentences[:10]:
                    if any(keyword in sentence.lower() for keyword in investment_keywords):
                        relevant_sentences.append(sentence.strip())
                        if len(' '.join(relevant_sentences)) > 250:
                            break
                
                if relevant_sentences:
                    summary = '. '.join(relevant_sentences[:3]) + '.'
                else:
                    summary = content[:250].strip() + "..."
                
                # Format output with validated stock name
                output += f"### {i}. 📊 {stock_name}\n\n"
                output += f"**Why This Stock:** {summary}\n\n"
                
                if url:
                    output += f"**🔗 [Read Complete Analysis]({url})**\n\n"
                
                output += "---\n\n"
            
            logger.info(f"Found {len(valid_stocks)} valid growth stock recommendations with tickers")
            return output
            
        except Exception as e:
            logger.error(f"Failed to search for growth stocks: {e}")
            return f"Error searching for growth stocks: {str(e)}"


def recommend_growth_stocks_tool(market: str = "US") -> str:
    """
    LangChain tool function for growth stock recommendations.
    
    This is the tool function that will be exposed to the ReAct agent.
    
    Args:
        market: Target market for recommendations
        
    Returns:
        Formatted recommendations from web search with expert analysis
    """
    recommender = StockRecommender()
    return recommender.get_growth_stocks(market)
