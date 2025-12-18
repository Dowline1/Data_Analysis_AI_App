"""
Analysis Subgraph - Orchestrates financial analysis agents.
Uses CodeAct agents for calculations and LLM for natural language insights.
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.state.app_state import AnalysisState
from src.agents.transaction_categorizer import TransactionCategorizerAgent
from src.agents.metrics_calculator import MetricsCalculatorAgent
from src.agents.subscription_detector import SubscriptionDetectorAgent
from src.agents.insights_generator import InsightsGeneratorAgent
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AnalysisGraph:
    """
    LangGraph subgraph for financial analysis.
    
    Flow:
    1. categorize_transactions (CodeAct - keyword matching)
    2. calculate_metrics (CodeAct - pandas operations)
    3. detect_subscriptions (CodeAct - pattern analysis)
    4. generate_insights (LLM - natural language)
    5. finalize_analysis
    
    Demonstrates:
    - CodeAct pattern: Python execution for calculations
    - LLM usage: Natural language generation from structured data
    - Subgraph orchestration with LangGraph
    """
    
    def __init__(self):
        """Initialize analysis agents and build graph."""
        # Initialize agents
        self.categorizer = TransactionCategorizerAgent()
        self.metrics_calculator = MetricsCalculatorAgent()
        self.subscription_detector = SubscriptionDetectorAgent()
        self.insights_generator = InsightsGeneratorAgent()
        
        # Build graph
        self.graph = self._build_graph()
        
        logger.info("Initialized AnalysisGraph with 4 agents")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the analysis subgraph.
        
        Returns:
            Compiled StateGraph
        """
        # Create graph
        workflow = StateGraph(AnalysisState)
        
        # Add nodes
        workflow.add_node("categorize_transactions", self._categorize_transactions)
        workflow.add_node("calculate_metrics", self._calculate_metrics)
        workflow.add_node("detect_subscriptions", self._detect_subscriptions)
        workflow.add_node("generate_insights", self._generate_insights)
        workflow.add_node("finalize_analysis", self._finalize_analysis)
        
        # Add edges (linear flow)
        workflow.set_entry_point("categorize_transactions")
        workflow.add_edge("categorize_transactions", "calculate_metrics")
        workflow.add_edge("calculate_metrics", "detect_subscriptions")
        workflow.add_edge("detect_subscriptions", "generate_insights")
        workflow.add_edge("generate_insights", "finalize_analysis")
        workflow.add_edge("finalize_analysis", END)
        
        # Compile with memory
        memory = MemorySaver()
        compiled = workflow.compile(checkpointer=memory)
        
        logger.info("Built AnalysisGraph with 5 nodes")
        
        return compiled
    
    def _categorize_transactions(self, state: AnalysisState) -> Dict[str, Any]:
        """
        Node: Categorize transactions using CodeAct pattern.
        
        Args:
            state: Current analysis state
            
        Returns:
            State updates
        """
        logger.info("Node: categorize_transactions")
        
        transactions = state.get('transactions', [])
        
        if not transactions:
            logger.warning("No transactions to categorize")
            return {
                'errors': ["No transactions provided for analysis"]
            }
        
        try:
            # Execute Python categorization (CodeAct)
            categorized = self.categorizer.categorize_transactions(transactions)
            
            # Get category summary
            category_summary = self.categorizer.get_category_summary(categorized)
            spending_by_category = self.categorizer.get_spending_by_category(categorized)
            
            logger.info(f"Categorized {len(categorized)} transactions into {len(category_summary)} categories")
            
            return {
                'transactions': categorized,
                'category_summary': category_summary,
                'spending_by_category': spending_by_category
            }
            
        except Exception as e:
            logger.error(f"Error categorizing transactions: {e}")
            return {
                'errors': [f"Categorization failed: {str(e)}"]
            }
    
    def _calculate_metrics(self, state: AnalysisState) -> Dict[str, Any]:
        """
        Node: Calculate financial metrics using CodeAct pattern.
        
        Args:
            state: Current analysis state
            
        Returns:
            State updates
        """
        logger.info("Node: calculate_metrics")
        
        transactions = state.get('transactions', [])
        
        try:
            # Execute pandas calculations (CodeAct)
            metrics = self.metrics_calculator.calculate_all_metrics(transactions)
            
            logger.info(f"Calculated metrics: Health score {metrics['health']['health_score']}/100")
            
            return {
                'basic_metrics': metrics['basic'],
                'spending_metrics': metrics['spending'],
                'income_metrics': metrics['income'],
                'trend_metrics': metrics['trends'],
                'financial_health_score': metrics['health']['health_score'],
                'financial_health_rating': metrics['health']['rating'],
                'financial_health_details': metrics['health']
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                'errors': [f"Metrics calculation failed: {str(e)}"]
            }
    
    def _detect_subscriptions(self, state: AnalysisState) -> Dict[str, Any]:
        """
        Node: Detect recurring subscriptions using CodeAct pattern.
        
        Args:
            state: Current analysis state
            
        Returns:
            State updates
        """
        logger.info("Node: detect_subscriptions")
        
        transactions = state.get('transactions', [])
        
        try:
            # Execute pattern detection (CodeAct)
            subscription_data = self.subscription_detector.detect_subscriptions(transactions)
            
            subscriptions = subscription_data['subscriptions']
            
            logger.info(f"Detected {len(subscriptions)} subscriptions, total: €{subscription_data['total_subscription_cost']:.2f}/month")
            
            return {
                'subscriptions': subscriptions,
                'subscription_total_cost': subscription_data['total_subscription_cost']
            }
            
        except Exception as e:
            logger.error(f"Error detecting subscriptions: {e}")
            return {
                'subscriptions': [],
                'subscription_total_cost': 0.0,
                'errors': [f"Subscription detection failed: {str(e)}"]
            }
    
    def _generate_insights(self, state: AnalysisState) -> Dict[str, Any]:
        """
        Node: Generate natural language insights using LLM.
        
        This is appropriate LLM usage: converting structured data to human-readable text.
        
        Args:
            state: Current analysis state
            
        Returns:
            State updates
        """
        logger.info("Node: generate_insights")
        
        try:
            # Build metrics dict for LLM
            metrics = {
                'basic': state.get('basic_metrics', {}),
                'spending': state.get('spending_metrics', {}),
                'income': state.get('income_metrics', {}),
                'trends': state.get('trend_metrics', {}),
                'health': state.get('financial_health_details', {})
            }
            
            category_summary = state.get('category_summary', {})
            
            subscriptions = {
                'subscriptions': state.get('subscriptions', []),
                'total_subscription_cost': state.get('subscription_total_cost', 0),
                'count': len(state.get('subscriptions', []))
            }
            
            # Generate insights using LLM
            insights = self.insights_generator.generate_insights(
                metrics=metrics,
                category_summary=category_summary,
                subscriptions=subscriptions
            )
            
            # Flatten insights into single list for recommendations
            all_insights = []
            for category, insight_list in insights.items():
                if category != 'alerts' and category != 'recommendations':
                    all_insights.extend(insight_list)
            
            recommendations = insights.get('recommendations', [])
            alerts = insights.get('alerts', [])
            
            logger.info(f"Generated {len(all_insights)} insights, {len(recommendations)} recommendations, {len(alerts)} alerts")
            
            return {
                'insights': all_insights,
                'recommendations': recommendations,
                'alerts': alerts
            }
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {
                'insights': ["Analysis complete - view metrics for details"],
                'recommendations': ["Review your financial data regularly"],
                'alerts': [],
                'errors': [f"Insight generation failed: {str(e)}"]
            }
    
    def _finalize_analysis(self, state: AnalysisState) -> Dict[str, Any]:
        """
        Node: Finalize analysis and log summary.
        
        Args:
            state: Current analysis state
            
        Returns:
            State updates
        """
        logger.info("Node: finalize_analysis")
        
        # Log summary
        health_score = state.get('financial_health_score', 0)
        health_rating = state.get('financial_health_rating', 'Unknown')
        num_transactions = len(state.get('transactions', []))
        num_subscriptions = len(state.get('subscriptions', []))
        num_insights = len(state.get('insights', []))
        
        logger.info(f"""
Analysis Complete:
- Transactions: {num_transactions}
- Health Score: {health_score}/100 ({health_rating})
- Subscriptions: {num_subscriptions}
- Insights: {num_insights}
        """)
        
        return {
            'analysis_complete': True
        }
    
    def invoke(self, transactions: list, thread_id: str = "default") -> Dict[str, Any]:
        """
        Run analysis on transactions.
        
        Args:
            transactions: List of validated transactions from ingestion
            thread_id: Thread ID for checkpointing
            
        Returns:
            Final state dict
        """
        logger.info(f"Starting analysis on {len(transactions)} transactions")
        
        initial_state = {
            'transactions': transactions
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        result = self.graph.invoke(initial_state, config)
        
        logger.info("Analysis complete")
        
        return result
    
    def stream(self, transactions: list, thread_id: str = "default"):
        """
        Stream analysis execution.
        
        Args:
            transactions: List of validated transactions
            thread_id: Thread ID for checkpointing
            
        Yields:
            State updates
        """
        logger.info(f"Streaming analysis on {len(transactions)} transactions")
        
        initial_state = {
            'transactions': transactions
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        for update in self.graph.stream(initial_state, config):
            yield update
