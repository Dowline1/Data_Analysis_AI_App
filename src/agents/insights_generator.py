"""
Insights Generator Agent - Uses LLM for natural language insights.
This is appropriate LLM use: generating human-readable text from structured data.
"""

from typing import Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class InsightsGeneratorAgent:
    """
    Agent that uses LLM to generate natural language insights from financial data.
    This is appropriate LLM usage: converting structured metrics into readable insights.
    """
    
    def __init__(self):
        """Initialize with Gemini LLM."""
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=0.7  # Some creativity for natural language
        )
        
        logger.info(f"Initialized InsightsGeneratorAgent with {Config.GEMINI_MODEL}")
    
    def generate_insights(
        self,
        metrics: Dict,
        category_summary: Dict,
        subscriptions: Dict
    ) -> Dict[str, List[str]]:
        """
        Use LLM to generate natural language insights from structured data.
        
        Args:
            metrics: Financial metrics from MetricsCalculatorAgent
            category_summary: Category breakdown from TransactionCategorizerAgent
            subscriptions: Subscription data from SubscriptionDetectorAgent
            
        Returns:
            Dict with categorized insights
        """
        logger.info("Generating insights from financial data")
        
        # Build context for LLM
        context = self._build_context(metrics, category_summary, subscriptions)
        
        # Generate different types of insights
        insights = {
            'spending_insights': self._generate_spending_insights(context),
            'income_insights': self._generate_income_insights(context),
            'savings_insights': self._generate_savings_insights(context),
            'subscription_insights': self._generate_subscription_insights(context),
            'recommendations': self._generate_recommendations(context),
            'alerts': self._generate_alerts(context)
        }
        
        logger.info(f"Generated {sum(len(v) for v in insights.values())} total insights")
        
        return insights
    
    def _build_context(
        self,
        metrics: Dict,
        category_summary: Dict,
        subscriptions: Dict
    ) -> Dict:
        """
        Build structured context for LLM from all analysis data.
        
        Args:
            metrics: Metrics dictionary
            category_summary: Category summary
            subscriptions: Subscription data
            
        Returns:
            Context dict
        """
        return {
            'basic': metrics.get('basic', {}),
            'spending': metrics.get('spending', {}),
            'income': metrics.get('income', {}),
            'trends': metrics.get('trends', {}),
            'health': metrics.get('health', {}),
            'categories': category_summary,
            'subscriptions': subscriptions
        }
    
    def _generate_spending_insights(self, context: Dict) -> List[str]:
        """
        Generate insights about spending patterns using LLM.
        
        Args:
            context: Financial data context
            
        Returns:
            List of spending insights
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial advisor analyzing spending patterns.
Generate 3-5 clear, actionable insights about spending behavior.
Focus on:
- Category breakdown and top spending areas
- Unusual or notable spending patterns
- Trends (increasing/decreasing)
- Comparisons to typical behavior

Format: Return a JSON array of insight strings.
Example: ["You spent €500 on Dining, which is 25% of total expenses", "Shopping expenses increased by 30% recently"]

Keep insights concise, specific, and data-driven."""),
            ("user", """Analyze this spending data:

Total Expenses: €{total_expenses}
Top Categories: {top_categories}
Largest Expense: €{largest_expense}
Spending Trend: {spending_trend}
Daily Average: €{daily_avg}

Generate 3-5 spending insights.""")
        ])
        
        spending = context.get('spending', {})
        basic = context.get('basic', {})
        trends = context.get('trends', {})
        
        # Get top categories
        category_spending = spending.get('spending_by_category', {})
        top_categories = ', '.join([f"{k}: €{v:.2f}" for k, v in list(category_spending.items())[:3]])
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                'total_expenses': basic.get('total_expenses', 0),
                'top_categories': top_categories or "None",
                'largest_expense': abs(basic.get('largest_expense', 0)),
                'spending_trend': trends.get('spending_trend', 'stable'),
                'daily_avg': trends.get('daily_spending_avg', 0)
            })
            
            # Extract insights from response
            insights = self._parse_insights_response(response.content)
            
            return insights[:5]  # Max 5 insights
            
        except Exception as e:
            logger.error(f"Error generating spending insights: {e}")
            return [f"Total expenses: €{basic.get('total_expenses', 0):.2f}"]
    
    def _generate_income_insights(self, context: Dict) -> List[str]:
        """
        Generate insights about income using LLM.
        
        Args:
            context: Financial data context
            
        Returns:
            List of income insights
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial advisor analyzing income patterns.
Generate 2-4 clear insights about income.
Focus on:
- Total income and sources
- Income consistency
- Income vs. expenses comparison

Format: Return a JSON array of insight strings.
Keep insights concise and data-driven."""),
            ("user", """Analyze this income data:

Total Income: €{total_income}
Income Sources: {income_sources}
Average Income Transaction: €{avg_income}
Total Expenses: €{total_expenses}
Net Balance: €{net_balance}

Generate 2-4 income insights.""")
        ])
        
        income = context.get('income', {})
        basic = context.get('basic', {})
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                'total_income': income.get('total_income', 0),
                'income_sources': income.get('income_sources', 0),
                'avg_income': income.get('average_income', 0),
                'total_expenses': basic.get('total_expenses', 0),
                'net_balance': basic.get('net_balance', 0)
            })
            
            insights = self._parse_insights_response(response.content)
            return insights[:4]
            
        except Exception as e:
            logger.error(f"Error generating income insights: {e}")
            return [f"Total income: €{income.get('total_income', 0):.2f}"]
    
    def _generate_savings_insights(self, context: Dict) -> List[str]:
        """
        Generate insights about savings and financial health using LLM.
        
        Args:
            context: Financial data context
            
        Returns:
            List of savings insights
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial advisor analyzing savings and financial health.
Generate 2-4 clear insights about savings capacity and financial health.
Focus on:
- Savings rate and net savings
- Financial health score interpretation
- Income/expense ratio

Format: Return a JSON array of insight strings.
Keep insights actionable and encouraging where appropriate."""),
            ("user", """Analyze this savings data:

Health Score: {health_score}/100 ({rating})
Net Savings: €{net_savings}
Savings Percentage: {savings_pct}%
Income/Expense Ratio: {income_expense_ratio}

Generate 2-4 savings insights.""")
        ])
        
        health = context.get('health', {})
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                'health_score': health.get('health_score', 0),
                'rating': health.get('rating', 'Unknown'),
                'net_savings': health.get('net_savings', 0),
                'savings_pct': health.get('savings_percentage', 0),
                'income_expense_ratio': health.get('factors', {}).get('income_expense_ratio', 0)
            })
            
            insights = self._parse_insights_response(response.content)
            return insights[:4]
            
        except Exception as e:
            logger.error(f"Error generating savings insights: {e}")
            return [f"Financial health score: {health.get('health_score', 0)}/100"]
    
    def _generate_subscription_insights(self, context: Dict) -> List[str]:
        """
        Generate insights about subscriptions using LLM.
        
        Args:
            context: Financial data context
            
        Returns:
            List of subscription insights
        """
        subscriptions = context.get('subscriptions', {})
        sub_list = subscriptions.get('subscriptions', [])
        
        if not sub_list:
            return ["No recurring subscriptions detected"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial advisor analyzing subscription spending.
Generate 2-3 clear insights about recurring subscriptions.
Focus on:
- Total subscription cost impact
- Most expensive subscriptions
- Opportunities to save

Format: Return a JSON array of insight strings.
Be specific and actionable."""),
            ("user", """Analyze this subscription data:

Total Monthly Cost: €{total_monthly}
Total Annual Cost: €{total_annual}
Number of Subscriptions: {count}
Most Expensive: {most_expensive}

Generate 2-3 subscription insights.""")
        ])
        
        try:
            most_exp = subscriptions.get('total_subscription_cost', 0) / subscriptions.get('count', 1)
            
            chain = prompt | self.llm
            response = chain.invoke({
                'total_monthly': subscriptions.get('total_subscription_cost', 0),
                'total_annual': subscriptions.get('total_subscription_cost', 0) * 12,
                'count': subscriptions.get('count', 0),
                'most_expensive': f"€{most_exp:.2f}/month"
            })
            
            insights = self._parse_insights_response(response.content)
            return insights[:3]
            
        except Exception as e:
            logger.error(f"Error generating subscription insights: {e}")
            return [f"Total subscription cost: €{subscriptions.get('total_subscription_cost', 0):.2f}/month"]
    
    def _generate_recommendations(self, context: Dict) -> List[str]:
        """
        Generate actionable recommendations using LLM.
        
        Args:
            context: Financial data context
            
        Returns:
            List of recommendations
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial advisor providing actionable recommendations.
Generate 3-5 specific, actionable recommendations to improve financial health.
Focus on:
- Reducing unnecessary spending
- Increasing savings
- Managing subscriptions
- Improving financial habits

Format: Return a JSON array of recommendation strings.
Start each with an action verb (e.g., "Review", "Consider", "Set aside")."""),
            ("user", """Based on this financial summary:

Health Score: {health_score}/100
Savings Rate: {savings_pct}%
Total Subscriptions: €{subscriptions}/month
Spending Volatility: {volatility}

Generate 3-5 actionable recommendations.""")
        ])
        
        health = context.get('health', {})
        subs = context.get('subscriptions', {})
        trends = context.get('trends', {})
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                'health_score': health.get('health_score', 0),
                'savings_pct': health.get('savings_percentage', 0),
                'subscriptions': subs.get('total_subscription_cost', 0),
                'volatility': 'high' if trends.get('spending_volatility', 0) > 50 else 'moderate'
            })
            
            recommendations = self._parse_insights_response(response.content)
            return recommendations[:5]
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Review your spending patterns regularly"]
    
    def _generate_alerts(self, context: Dict) -> List[str]:
        """
        Generate alerts for unusual patterns or concerns.
        
        Args:
            context: Financial data context
            
        Returns:
            List of alerts
        """
        alerts = []
        
        health = context.get('health', {})
        basic = context.get('basic', {})
        trends = context.get('trends', {})
        
        # Check financial health
        health_score = health.get('health_score', 0)
        if health_score < 40:
            alerts.append("⚠️ Low financial health score - urgent attention needed")
        elif health_score < 60:
            alerts.append("⚠️ Financial health could be improved")
        
        # Check negative balance
        if basic.get('net_balance', 0) < 0:
            alerts.append("⚠️ Negative net balance - expenses exceed income")
        
        # Check high volatility
        if trends.get('spending_volatility', 0) > 100:
            alerts.append("⚠️ High spending volatility detected - consider budgeting")
        
        # Check increasing spending trend
        if trends.get('spending_trend') == 'increasing':
            alerts.append("📈 Spending is trending upward")
        
        return alerts
    
    def _parse_insights_response(self, content: str) -> List[str]:
        """
        Parse LLM response to extract insights array.
        
        Args:
            content: LLM response content
            
        Returns:
            List of insight strings
        """
        try:
            # Try to parse as JSON array
            import json
            
            # Remove markdown code blocks if present
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            # Parse JSON
            insights = json.loads(content)
            
            if isinstance(insights, list):
                return [str(insight) for insight in insights]
            else:
                return [str(insights)]
                
        except Exception as e:
            logger.warning(f"Could not parse JSON insights, using raw text: {e}")
            
            # Fallback: split by lines
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            # Remove common prefixes
            cleaned = []
            for line in lines:
                # Remove bullet points, numbers, etc.
                line = line.lstrip('*-•►> ').lstrip('0123456789.') .strip()
                if line and len(line) > 10:  # Ignore very short lines
                    cleaned.append(line)
            
            return cleaned if cleaned else [content]
