"""
Data Ingestion Subgraph - LangGraph Implementation
Orchestrates file parsing, schema mapping, validation, and HITL review.
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.state.app_state import DataIngestionState
from src.tools.file_parser import FileParser
from src.agents.schema_mapper import SchemaMapperAgent
from src.agents.data_validator import DataValidatorAgent
from src.utils.guardrails import GuardrailManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataIngestionGraph:
    """
    Data Ingestion Subgraph implementing:
    - File parsing (external tool)
    - Schema mapping (ReAct agent)
    - Data validation (ReAct agent with reflection)
    - HITL checkpoint for user review
    - Reflection loop for poor quality data
    """
    
    def __init__(self):
        self.file_parser = FileParser()
        self.schema_mapper = SchemaMapperAgent()
        self.validator = DataValidatorAgent()
        self.guardrails = GuardrailManager(enable_pii_masking=True)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph for data ingestion."""
        
        # Create graph with DataIngestionState
        workflow = StateGraph(DataIngestionState)
        
        # Add nodes
        workflow.add_node("parse_file", self._parse_file_node)
        workflow.add_node("extract_transactions", self._extract_transactions_node)
        workflow.add_node("validate_data", self._validate_data_node)
        workflow.add_node("request_human_review", self._request_human_review_node)
        workflow.add_node("finalize_ingestion", self._finalize_ingestion_node)
        
        # Define edges
        workflow.set_entry_point("parse_file")
        
        # Parse → Extract
        workflow.add_edge("parse_file", "extract_transactions")
        
        # Extract → Validate
        workflow.add_edge("extract_transactions", "validate_data")
        
        # Validate → Decision (reflection loop or proceed)
        workflow.add_conditional_edges(
            "validate_data",
            self._should_reflect_or_proceed,
            {
                "reflect": "extract_transactions",  # Loop back for reflection
                "proceed": "request_human_review"
            }
        )
        
        # Human review → Decision (approved or rejected)
        workflow.add_conditional_edges(
            "request_human_review",
            self._check_human_approval,
            {
                "approved": "finalize_ingestion",
                "rejected": "extract_transactions",  # Re-extract with feedback
                "waiting": END  # Interrupt for human input
            }
        )
        
        # Finalize → End
        workflow.add_edge("finalize_ingestion", END)
        
        # Compile with memory for HITL interrupts
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory, interrupt_before=["request_human_review"])
    
    # === Node Functions ===
    
    def _parse_file_node(self, state: DataIngestionState) -> DataIngestionState:
        """Node: Parse the uploaded file."""
        logger.info(f"Parsing file: {state['file_path']}")
        # GUARDRAIL #1: Validate file before processing
        is_valid, issues = self.guardrails.validate_file(state['file_path'])
        
        if not is_valid:
            logger.error(f"File validation failed: {issues}")
            state['errors'].extend(issues)
            return state
        
        
        try:
            result = self.file_parser.parse_file(state['file_path'])
            
            state['file_type'] = result['file_type']
            state['extracted_columns'] = result['columns']
            
            # Store either raw text OR structured data (prefer structured)
            df = result['dataframe']
            if df is not None and len(df) > 0:
                # We have structured table data - convert to JSON for state storage
                logger.info(f"Storing structured table data: {len(df)} rows, {len(df.columns)} columns")
                state['raw_data'] = df.to_json(orient='records', date_format='iso')
            else:
                # No tables found, use raw text
                logger.info(f"Storing raw text data: {len(result['raw_text'])} chars")
                state['raw_data'] = result['raw_text']
            
            logger.info(f"Successfully parsed {result['file_type']} file")
            
        except Exception as e:
            logger.error(f"File parsing failed: {e}")
            state['errors'].append(f"Parsing error: {str(e)}")
        
        return state
    
    def _extract_transactions_node(self, state: DataIngestionState) -> DataIngestionState:
        """Node: Extract transactions using Schema Mapper Agent."""
        logger.info("Extracting transactions with Schema Mapper Agent")
        
        # Check if we have reflection feedback
        reflection_context = None
        if state.get('reflection_notes'):
            reflection_context = state['reflection_notes'][-1]  # Get latest reflection
            logger.info(f"Applying reflection feedback: {reflection_context}")
        
        try:
            # Choose extraction method based on data format
            raw_data = state['raw_data']
            
            # Check if raw_data is JSON (structured table)
            if raw_data and raw_data.strip().startswith('['):
                # It's JSON - convert back to DataFrame
                import json
                import pandas as pd
                try:
                    data_list = json.loads(raw_data)
                    df = pd.DataFrame(data_list)
                    logger.info(f"Using structured DataFrame extraction ({len(df)} rows, {len(df.columns)} columns)")
                    transactions = self.schema_mapper.extract_from_dataframe(
                        df, 
                        state['file_type']
                    )
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON, falling back to text extraction")
                    transactions = self.schema_mapper.extract_from_text(
                        raw_data, 
                        state['file_type']
                    )
            else:
                # It's raw text
                logger.info(f"Using text extraction ({len(raw_data)} chars)")
                transactions = self.schema_mapper.extract_from_text(
                    raw_data, 
                    state['file_type']
                )
            
            state['transactions'] = transactions
            state['retry_count'] = state.get('retry_count', 0) + 1
            
            logger.info(f"Extracted {len(transactions)} transactions (attempt #{state['retry_count']})")
            
        except Exception as e:
            logger.error(f"Transaction extraction failed: {e}")
            state['errors'].append(f"Extraction error: {str(e)}")
            state['retry_count'] = state.get('retry_count', 0) + 1
        
        return state
    
    def _validate_data_node(self, state: DataIngestionState) -> DataIngestionState:
        """Node: Validate transaction data quality."""
        logger.info("Validating transaction data quality")
        
        try:
            validation_result = self.validator.validate(state['transactions'])
            
            state['data_quality_score'] = validation_result['quality_score']
            state['validation_summary'] = {
                'quality_score': validation_result['quality_score'],
                'is_valid': validation_result['is_valid'],
                'issues_found': validation_result['issues'],
                'suggestions': validation_result['suggestions']
            }
            
            # Store validation feedback for potential reflection
            if not validation_result['is_valid']:
                reflection = f"Quality score: {validation_result['quality_score']}/100. "
                reflection += f"Issues: {', '.join(validation_result['issues'])}. "
                reflection += f"Suggestions: {', '.join(validation_result['suggestions'])}"
                state['reflection_notes'].append(reflection)
            
            logger.info(f"Validation complete - Score: {validation_result['quality_score']}/100")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            state['errors'].append(f"Validation error: {str(e)}")
            state['data_quality_score'] = 0
        
        return state
    
    def _request_human_review_node(self, state: DataIngestionState) -> DataIngestionState:
        """Node: Prepare data for human review (HITL checkpoint)."""
        logger.info("Requesting human review of extracted transactions")
        
        # Calculate monthly totals for human review
        import pandas as pd
        from datetime import datetime
        
        transactions = state['transactions']
        if transactions:
            df = pd.DataFrame(transactions)
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.to_period('M')
            
            # Calculate monthly totals
            monthly_summary = df.groupby('month').agg({
                'amount': ['sum', 'count']
            }).round(2)
            
            # Convert to dict for storage
            monthly_totals = {}
            for month, row in monthly_summary.iterrows():
                month_str = str(month)
                monthly_totals[month_str] = {
                    'total': float(row[('amount', 'sum')]),
                    'income': float(df[(df['month'] == month) & (df['amount'] > 0)]['amount'].sum()),
                    'expenses': float(abs(df[(df['month'] == month) & (df['amount'] < 0)]['amount'].sum())),
                    'net': float(df[df['month'] == month]['amount'].sum()),
                    'transaction_count': int(row[('amount', 'count')])
                }
            
            # Store in validation_summary for human review
            if state['validation_summary'] is None:
                state['validation_summary'] = {}
            
            state['validation_summary']['monthly_totals'] = monthly_totals
            state['validation_summary']['total_transactions'] = len(transactions)
            state['validation_summary']['date_range'] = {
                'start': str(df['date'].min().date()),
                'end': str(df['date'].max().date())
            }
            
            logger.info(f"Prepared monthly summary for {len(monthly_totals)} months")
            for month, data in monthly_totals.items():
                logger.info(f"  {month}: €{data['net']:.2f} net ({data['transaction_count']} transactions)")
        
        return state
    
    def _finalize_ingestion_node(self, state: DataIngestionState) -> DataIngestionState:
        """Node: Finalize ingestion after human approval."""
        logger.info("Finalizing data ingestion")
        
        state['ingestion_complete'] = True
        logger.info(f"Data ingestion complete: {len(state['transactions'])} transactions approved")
        
        return state
    
    # === Conditional Edge Functions ===
    
    def _should_reflect_or_proceed(self, state: DataIngestionState) -> Literal["reflect", "proceed"]:
        """
        Decide whether to reflect (loop back) or proceed to human review.
        
        Reflection loop triggers if:
        1. Data quality is poor (below threshold)
        2. Haven't exceeded max reflection iterations
        3. Have transactions to work with
        """
        max_iterations = 3
        current_iteration = state.get('retry_count', 0)
        quality_threshold = 70
        
        # If validation passed (quality score >= threshold), proceed to human review
        quality_score = state.get('data_quality_score', 0)
        if quality_score >= quality_threshold:
            logger.info(f"Validation passed (score {quality_score} >= {quality_threshold}) - proceeding to human review")
            return "proceed"
        
        # If no transactions extracted, don't loop (extraction won't improve)
        if len(state.get('transactions', [])) == 0:
            logger.warning("No transactions extracted - proceeding to human review")
            return "proceed"
        
        # If max iterations reached, proceed anyway (let human decide)
        if current_iteration >= max_iterations:
            logger.warning(f"Max reflection iterations ({max_iterations}) reached - proceeding to human review")
            return "proceed"
        
        # Reflection loop: try again with feedback
        logger.info(f"Data quality below threshold (score {quality_score} < {quality_threshold}) - initiating reflection loop (iteration {current_iteration + 1})")
        return "reflect"
    
    def _check_human_approval(self, state: DataIngestionState) -> Literal["approved", "rejected", "waiting"]:
        """
        Check if human has approved the data.
        
        This function checks the human_approved field:
        - None: Still waiting for review (interrupt)
        - True: Approved, proceed
        - False: Rejected, re-extract
        """
        approval = state.get('human_approved')
        
        if approval is None:
            logger.info("Waiting for human approval")
            return "waiting"
        elif approval is True:
            logger.info("Human approved data - finalizing")
            return "approved"
        else:
            logger.info("Human rejected data - re-extracting with feedback")
            # Add human feedback to reflection notes if provided
            if state.get('human_feedback'):
                state['reflection_notes'].append(f"Human feedback: {state['human_feedback']}")
            return "rejected"
    
    # === Public Methods ===
    
    def invoke(self, file_path: str, thread_id: str = "default"):
        """
        Run the data ingestion pipeline.
        
        Args:
            file_path: Path to the file to ingest
            thread_id: Thread ID for checkpointing
            
        Returns:
            Final state after ingestion
        """
        initial_state = DataIngestionState(
            file_path=file_path,
            file_type=None,
            raw_data=None,
            extracted_columns=[],
            transactions=[],
            data_quality_score=0.0,
            human_approved=None,
            human_feedback=None,
            validation_summary=None,
            errors=[],
            reflection_notes=[],
            retry_count=0
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        return self.graph.invoke(initial_state, config)
    
    def stream(self, file_path: str, thread_id: str = "default"):
        """
        Stream the data ingestion pipeline (for real-time updates in UI).
        
        Args:
            file_path: Path to the file to ingest
            thread_id: Thread ID for checkpointing
            
        Yields:
            State updates as they happen
        """
        initial_state = DataIngestionState(
            file_path=file_path,
            file_type=None,
            raw_data=None,
            extracted_columns=[],
            transactions=[],
            data_quality_score=0.0,
            human_approved=None,
            human_feedback=None,
            validation_summary=None,
            errors=[],
            reflection_notes=[],
            retry_count=0
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        for state in self.graph.stream(initial_state, config):
            yield state
    
    def update_human_approval(self, thread_id: str, approved: bool, feedback: str = None):
        """
        Update the graph state with human approval/rejection.
        
        This is called after the graph interrupts at the HITL checkpoint.
        
        Args:
            thread_id: Thread ID (must match original invoke/stream)
            approved: True if approved, False if rejected
            feedback: Optional feedback from human
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        # Update state and resume
        update_state = {
            "human_approved": approved,
            "human_feedback": feedback
        }
        
        return self.graph.update_state(config, update_state)
