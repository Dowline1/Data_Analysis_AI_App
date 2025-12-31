# LangGraph Architecture Migration

## Overview

This document describes the migration from direct agent calls to a LangGraph StateGraph architecture to meet assignment requirements.

## Architecture Changes

### Before: Linear Pipeline

The original implementation used direct function calls in a linear sequence:

```
File Upload → Schema Detection → Transaction Extraction → 
Categorization → Subscription Detection → Metrics Calculation → 
Expert Analysis → Results Display
```

HITL checkpoints were implemented in Streamlit session state.

### After: LangGraph StateGraph with Subgraphs

The new architecture uses LangGraph's StateGraph with four hierarchical subgraphs:

```
Main Graph
├── Schema Detection Subgraph
│   ├── parse_file_node
│   ├── validate_schema_node (reflection)
│   ├── schema_guardrails_node
│   ├── await_confirmation_node (HITL)
│   └── apply_overrides_node
│
├── Data Processing Subgraph
│   ├── extract_transactions_node
│   ├── categorize_transactions_node
│   ├── validate_categorization_node (reflection)
│   ├── reflection_loop_node (self-correction)
│   ├── amount_guardrails_node
│   └── injection_guardrails_node
│
├── Subscription Detection Subgraph
│   ├── detect_subscriptions_node
│   ├── validate_subscriptions_node (reflection)
│   ├── await_confirmation_node (HITL)
│   └── apply_confirmations_node
│
└── Metrics & Analysis Subgraph
    ├── calculate_metrics_node
    ├── calculate_health_score_node
    ├── expert_analysis_node
    ├── validate_analysis_node (reflection)
    └── output_guardrails_node
```

## Key Features Implemented

### 1. StateGraph Architecture

- **State Schema** (`src/graph/state.py`): TypedDict defining the complete state that flows through the graph
- **Main Workflow** (`src/graph/workflow.py`): Orchestrates four subgraphs with conditional edges
- **Nodes** (`src/graph/nodes/*.py`): Individual processing units that read and write state

### 2. Subgraphs

Four hierarchical subgraphs, each compiled independently and added as nodes to the main graph:

1. **Schema Detection**: Parses file, validates schema, awaits user confirmation
2. **Data Processing**: Extracts transactions, categorizes with reflection loop
3. **Subscription Detection**: Detects subscriptions, awaits user confirmation
4. **Metrics & Analysis**: Calculates metrics, health scores, and generates expert insights

### 3. Reflection Loops

Self-correction mechanisms that validate outputs and attempt corrections:

- **Schema Validation**: Checks for missing columns and unknown account types
- **Categorization Validation**: Identifies uncategorized transactions, triggers re-categorization
- **Subscription Validation**: Checks for suspicious patterns (e.g., one-time payments)
- **Analysis Validation**: Ensures all stages completed successfully

Max 2 reflection iterations per stage to prevent infinite loops.

### 4. Guardrails System

Safety and quality checks at multiple stages:

- **Schema Guardrails**: Validates file paths, column names, account types
- **Amount Guardrails**: Checks for reasonable transaction amounts, valid dates
- **Prompt Injection Guardrails**: Detects malicious patterns in descriptions/feedback
- **Output Guardrails**: Validates expert report length, checks for PII leakage

Uses `guardrails-ai` library for structured validation.

### 5. HITL Checkpoints

LangGraph's built-in interrupt/resume mechanism for human-in-the-loop:

- **Schema Confirmation**: User reviews and overrides account type detection
- **Subscription Confirmation**: User selects legitimate subscriptions, adds notes

Implemented via `MemorySaver` checkpointer with thread IDs for state persistence.

### 6. MCP-Compliant Tools

Custom tools following Model Context Protocol (`src/tools/mcp_tools.py`):

- **FileParserTool**: Parse CSV/Excel files, extract structure
- **TransactionFilterTool**: Filter transactions by criteria (amount, category, date)
- **AggregationTool**: Group and aggregate transaction data
- **TrendAnalysisTool**: Analyze spending/income trends over time

Each tool has defined input/output schemas using Pydantic.

### 7. Evaluation Framework

Testing framework for quality assurance (`src/evals/evaluators.py`):

- **CategorizationEvaluator**: Accuracy, precision/recall per category
- **SubscriptionDetectionEvaluator**: Detection rate, false positive rate
- **HealthScoreEvaluator**: Score range validity, component consistency
- **EndToEndEvaluator**: Complete pipeline validation

## State Management

### State Schema

The `AnalysisState` TypedDict tracks:

- **Input**: File path, file content
- **Schema Detection**: Schema info, confirmations, overrides
- **Data Processing**: Transactions, raw data, categorization status
- **Subscriptions**: Detected subscriptions, user selections/notes
- **Metrics**: Account metrics, health score
- **Expert Analysis**: Insights, report
- **Reflection**: Validation errors, reflection notes, iteration count
- **Guardrails**: Violations, pass/fail status
- **HITL**: Awaiting input flags, user feedback

### Annotated Fields

Uses `Annotated` with `operator.add` for accumulative fields:
- `transactions`: Accumulates across multiple extraction calls
- `validation_errors`: Collects errors from all validation nodes
- `reflection_notes`: Tracks all reflection observations
- `guardrail_violations`: Accumulates violations across guardrail nodes

## Workflow Execution

### Initialization

```python
from src.graph.workflow import create_analysis_workflow, create_initial_state

# Create workflow (cached)
workflow = create_analysis_workflow()

# Create initial state
state = create_initial_state(file_path="/path/to/statement.csv")

# Config with thread ID for checkpointing
config = {"configurable": {"thread_id": "unique-thread-id"}}
```

### Streaming Execution

```python
# Stream events until interrupt or completion
for event in workflow.stream(state, config, stream_mode="values"):
    current_state = event
    
    # Check for HITL interrupt
    if event.get("awaiting_user_input"):
        break  # Show UI for user input
```

### Resuming After HITL

```python
# After user provides input
updated_state = dict(current_state)
updated_state["schema_overrides"] = user_overrides
updated_state["awaiting_user_input"] = False

# Continue from checkpoint
for event in workflow.stream(updated_state, config, stream_mode="values"):
    # Process next stages...
```

## File Structure

```
src/
├── graph/
│   ├── __init__.py
│   ├── state.py                 # State schema (TypedDict)
│   ├── workflow.py              # Main graph and subgraphs
│   └── nodes/
│       ├── schema_detection.py  # Schema detection nodes
│       ├── data_processing.py   # Transaction processing nodes
│       ├── subscription_detection.py  # Subscription nodes
│       ├── metrics_analysis.py  # Metrics & expert analysis nodes
│       └── guardrails.py        # Guardrail validation nodes
│
├── tools/
│   ├── __init__.py
│   └── mcp_tools.py             # MCP-compliant custom tools
│
├── evals/
│   ├── __init__.py
│   └── evaluators.py            # Evaluation framework
│
└── agents/                      # Existing agents (wrapped in nodes)
    ├── schema_mapper.py
    ├── transaction_categorizer.py
    ├── subscription_detector.py
    ├── metrics_calculator.py
    └── expert_insights_agent.py

app/
├── app.py                       # Original Streamlit app (preserved)
└── app_langgraph.py            # New LangGraph-based app
```

## Agent Integration

Existing agents are wrapped in LangGraph nodes:

```python
def categorize_transactions_node(state: AnalysisState) -> Dict[str, Any]:
    """Node that wraps TransactionCategorizerAgent."""
    transactions = state.get("transactions", [])
    
    # Use existing agent
    categorizer = TransactionCategorizerAgent()
    categorized = []
    
    for tx in transactions:
        category = categorizer.categorize(
            description=tx["description"],
            amount=tx["amount"],
            account_type=tx["account_type"]
        )
        tx["category"] = category
        categorized.append(tx)
    
    # Return state updates
    return {
        "transactions": categorized,
        "categorization_complete": True,
        "reflection_notes": [f"Categorized {len(categorized)} transactions"]
    }
```

## Conditional Edges

Edges with logic to route based on state:

```python
def should_reflect(state: AnalysisState) -> Literal["reflect", "continue"]:
    """Check if reflection loop is needed."""
    return "reflect" if state.get("needs_reflection") else "continue"

# In subgraph
subgraph.add_conditional_edges(
    "validate_categorization",
    should_reflect,
    {
        "reflect": "reflection_loop",
        "continue": "amount_guardrails"
    }
)
```

## Benefits of LangGraph Architecture

1. **Modularity**: Each subgraph is independent and testable
2. **Composability**: Subgraphs can be reused in different workflows
3. **Observability**: Built-in streaming provides visibility into execution
4. **Persistence**: Checkpointing enables pause/resume for HITL
5. **Error Handling**: State tracking makes debugging easier
6. **Scalability**: Can distribute subgraphs across services
7. **Testability**: Each node can be unit tested independently

## Running the Application

### Original App (Direct Calls)
```bash
streamlit run app/app.py
```

### New LangGraph App
```bash
streamlit run app/app_langgraph.py
```

Both apps have identical functionality from the user's perspective, but the LangGraph version uses the StateGraph architecture internally.

## Testing

### Unit Tests for Nodes

```python
from src.graph.nodes.data_processing import categorize_transactions_node
from src.graph.state import AnalysisState

def test_categorization_node():
    state = AnalysisState(
        transactions=[
            {"description": "Netflix", "amount": -15.99, "account_type": "checking"}
        ],
        ...
    )
    
    result = categorize_transactions_node(state)
    
    assert result["categorization_complete"] == True
    assert len(result["transactions"]) == 1
    assert result["transactions"][0]["category"] is not None
```

### Evaluation Tests

```python
from src.evals.evaluators import CategorizationEvaluator

evaluator = CategorizationEvaluator()
metrics = evaluator.evaluate(predicted, ground_truth)

assert metrics["overall_accuracy"] >= 0.8
```

## Future Enhancements

1. **Distributed Execution**: Deploy subgraphs as separate services
2. **Advanced Guardrails**: Integrate more sophisticated validation rules
3. **External API Tools**: Add tools for real-time market data, exchange rates
4. **Multi-Agent Collaboration**: Parallel expert agents with consensus mechanism
5. **Streaming Responses**: Real-time updates during long-running analysis
6. **Persistent Storage**: Database integration for historical analysis

## Dependencies

Key new dependencies:
- `langgraph>=1.0.0`: StateGraph, subgraphs, checkpointing
- `langgraph-checkpoint>=3.0.0`: Memory saver for HITL
- `guardrails-ai>=0.7.0`: Validation framework
- `langchain>=1.2.0`, `langchain-core>=1.2.0`: Updated for compatibility

See `requirements.txt` for complete list.

## Migration Checklist

- [x] Install LangGraph dependencies
- [x] Create state schema (TypedDict)
- [x] Build main StateGraph with 4 subgraphs
- [x] Implement reflection loops for self-correction
- [x] Add guardrails system for safety
- [x] Implement MCP-compliant tools
- [x] Create evaluation framework
- [x] Update main app to use LangGraph
- [x] Test HITL checkpoints
- [x] Document architecture changes

## Conclusion

The migration to LangGraph provides a robust, scalable architecture that meets all assignment requirements while preserving the excellent features of the original implementation (multi-account support, account-aware analysis, HITL validation, etc.).
