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
            state['raw_data'] = result['raw_text']  # Store raw text
            state['extracted_columns'] = result['columns']
            
            # Temporarily store DataFrame for extraction (not in state schema)
            state['_temp_dataframe'] = result['dataframe']
            
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
            # Choose extraction method based on available data
            if state.get('_temp_dataframe') is not None and len(state['_temp_dataframe']) > 0:
                transactions = self.schema_mapper.extract_from_dataframe(
                    state['_temp_dataframe'], 
                    state['file_type']
                )
            else:
                transactions = self.schema_mapper.extract_from_text(
                    state['raw_data'], 
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
            state['validation_passed'] = validation_result['is_valid']  # Temporary field
            
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
            state['validation_passed'] = False
        
        return state
    
    def _request_human_review_node(self, state: DataIngestionState) -> DataIngestionState:
        """Node: Prepare data for human review (HITL checkpoint)."""
        logger.info("Requesting human review of extracted transactions")
        
        # This node prepares the state for human review
        # The graph will interrupt here, allowing the UI to display data
        # and collect user feedback
        # GUARDRAIL #2: Apply PII masking to approved transactions
        state['transactions'] = self.guardrails.mask_pii(state['transactions'])
        
        pii_summary = self.guardrails.get_pii_masking_summary()
        if pii_summary:
            logger.info(f"PII masking applied: {pii_summary}")
        
        
        state['human_review_requested'] = True
        
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
        
        # If validation passed, proceed to human review
        if state.get('validation_passed', False):
            logger.info("Validation passed - proceeding to human review")
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
        logger.info(f"Data quality below threshold - initiating reflection loop (iteration {current_iteration + 1})")
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
    
    def invoke(self, file_path: str, config: dict = None):
        """
        Run the data ingestion pipeline.
        
        Args:
            file_path: Path to the file to ingest
            config: Optional config with thread_id for checkpointing
            
        Returns:
            Final state after ingestion
        """
        initial_state = DataIngestionState(
            file_path=file_path,
            file_type=None,
            parsed_data=None,
            raw_text=None,
            columns_found=[],
            transactions=[],
            validation_score=0.0,
            validation_passed=False,
            extraction_attempts=0,
            reflection_notes=[],
            errors=[],
            human_review_requested=False,
            human_approved=None,
            human_feedback=None,
            ingestion_complete=False
        )
        
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        return self.graph.invoke(initial_state, config)
    
    def stream(self, file_path: str, config: dict = None):
        """
        Stream the data ingestion pipeline (for real-time updates in UI).
        
        Args:
            file_path: Path to the file to ingest
            config: Optional config with thread_id for checkpointing
            
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
        
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        for state in self.graph.stream(initial_state, config):
            yield state
    
    def update_human_approval(self, approved: bool, feedback: str = None, config: dict = None):
        """
        Update the graph state with human approval/rejection.
        
        This is called after the graph interrupts at the HITL checkpoint.
        
        Args:
            approved: True if approved, False if rejected
            feedback: Optional feedback from human
            config: Config with thread_id to resume correct execution
        """
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        # Update state and resume
        update_state = {
            "human_approved": approved,
            "human_feedback": feedback
        }
        
        return self.graph.invoke(update_state, config)
