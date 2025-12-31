# LangSmith Tracing Setup Guide

## Overview

✅ **LangSmith tracing is already configured** in this project!

The LangGraph workflow, all agents, and nodes are automatically traced when LangSmith environment variables are set. No additional code is needed - LangChain handles tracing transparently.

## Current Configuration

### Environment Variables

The `.env.example` file includes LangSmith configuration:

```dotenv
# LangSmith Tracing (Optional but recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=bank-account-analyzer
```

### What Gets Traced

When LangSmith is enabled, **all of the following are automatically traced**:

#### 1. LangGraph Workflow Execution
- **Main StateGraph**: Entry/exit, state transitions
- **4 Subgraphs**: Individual subgraph execution
- **Node Executions**: Each node (parse_file, categorize, etc.)
- **Conditional Edges**: Decision points (should_reflect, check_guardrails)
- **Checkpoints**: HITL interrupts and resumptions

#### 2. Agent LLM Calls
All agents using `ChatGoogleGenerativeAI`:
- `SchemaMapperAgent` - Column detection, account classification
- `TransactionCategorizerAgent` - Transaction categorization
- `SubscriptionDetectorAgent` - Subscription validation
- `ExpertInsightsAgent` - Expert analysis generation
- `ChatWithDataAgent` - Q&A interactions

#### 3. Chain Executions
- Prompt templates
- Output parsers (JSON, String)
- Retry logic
- Error handling

#### 4. State Updates
- State mutations between nodes
- Accumulated fields (transactions, errors, notes)
- Reflection loop iterations

## Setup Instructions

### 1. Get LangSmith API Key

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Sign up or log in
3. Navigate to **Settings** → **API Keys**
4. Create a new API key
5. Copy the key

### 2. Configure Environment

Create a `.env` file in the project root (if not exists):

```bash
cp .env.example .env
```

Edit `.env` and add your LangSmith API key:

```dotenv
# LangSmith Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls_your_actual_api_key_here
LANGCHAIN_PROJECT=bank-account-analyzer
```

**Important**: `.env` is in `.gitignore` - your API key won't be committed.

### 3. Verify Setup

Run this to confirm tracing is working:

```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('LANGCHAIN_TRACING_V2:', os.getenv('LANGCHAIN_TRACING_V2')); print('LANGCHAIN_API_KEY:', 'SET' if os.getenv('LANGCHAIN_API_KEY') else 'NOT SET'); print('LANGCHAIN_PROJECT:', os.getenv('LANGCHAIN_PROJECT'))"
```

Expected output:
```
LANGCHAIN_TRACING_V2: true
LANGCHAIN_API_KEY: SET
LANGCHAIN_PROJECT: bank-account-analyzer
```

## What You'll See in LangSmith

### Trace View

Each workflow execution creates a trace showing:

```
bank-account-analyzer Run
├── Main Graph
│   ├── Schema Detection Subgraph
│   │   ├── parse_file_node [LLM Call]
│   │   ├── validate_schema_node
│   │   ├── schema_guardrails_node
│   │   ├── [INTERRUPT - Awaiting User Input]
│   │   └── apply_overrides_node
│   │
│   ├── Data Processing Subgraph
│   │   ├── extract_transactions_node [LLM Call]
│   │   ├── categorize_transactions_node
│   │   │   └── [Multiple LLM Calls for each transaction]
│   │   ├── validate_categorization_node
│   │   ├── reflection_loop_node (if needed)
│   │   ├── amount_guardrails_node
│   │   └── injection_guardrails_node
│   │
│   ├── Subscription Detection Subgraph
│   │   ├── detect_subscriptions_node [LLM Call]
│   │   ├── validate_subscriptions_node
│   │   ├── [INTERRUPT - Awaiting User Input]
│   │   └── apply_confirmations_node
│   │
│   └── Metrics & Analysis Subgraph
│       ├── calculate_metrics_node
│       ├── calculate_health_score_node
│       ├── expert_analysis_node [LLM Call]
│       ├── validate_analysis_node
│       └── output_guardrails_node
│
└── [Trace Complete]
```

### Metrics Tracked

- **Latency**: Time spent in each node/subgraph
- **Token Usage**: Input/output tokens per LLM call
- **Cost**: Estimated cost per run (if model pricing configured)
- **Errors**: Any exceptions or validation failures
- **Retries**: Reflection loop iterations

### Agent-Specific Traces

Each agent's LLM calls are traced with:

**SchemaMapperAgent**:
- Prompt: Column identification instructions
- Input: Sample data rows
- Output: Column mappings, account types
- Tokens: ~500-1000 per call

**TransactionCategorizerAgent**:
- Prompt: Category classification rules
- Input: Transaction description + amount
- Output: Category name
- Tokens: ~200-400 per transaction
- Note: Can be many calls for large datasets

**SubscriptionDetectorAgent**:
- Prompt: Subscription validation criteria
- Input: Merchant patterns, frequencies
- Output: Validation result (is_subscription: true/false)
- Tokens: ~300-500 per validation

**ExpertInsightsAgent**:
- Prompt: Expert analysis guidelines
- Input: Complete transaction ledger, metrics, subscriptions
- Output: Multi-section analysis report
- Tokens: ~2000-4000 (largest call)

## Advanced Features

### Custom Metadata

Add custom metadata to traces by setting environment variables:

```dotenv
LANGCHAIN_METADATA={"user_id": "test_user", "environment": "development"}
```

Or programmatically in nodes:

```python
from langsmith import trace

@trace(name="custom_node", metadata={"stage": "processing"})
def my_custom_node(state: AnalysisState) -> Dict[str, Any]:
    # Node implementation
    return {"status": "complete"}
```

### Filtering Traces

In LangSmith dashboard:
- Filter by project: `bank-account-analyzer`
- Filter by metadata: `environment:development`
- Filter by status: `success`, `error`
- Filter by latency: `> 5s`
- Filter by tokens: `> 1000`

### Sharing Traces

Share specific traces with your team:
1. Open trace in LangSmith
2. Click **Share** button
3. Copy share link
4. Link is public (but requires LangSmith account)

### Comparing Runs

Compare multiple runs to track improvements:
1. Select 2+ runs in dashboard
2. Click **Compare**
3. View side-by-side metrics
4. Identify bottlenecks

## Performance Insights

### Expected Latencies (with Gemini 1.5 Flash)

| Component | Typical Latency | Notes |
|-----------|----------------|-------|
| Schema Detection | 2-4s | Single LLM call |
| Transaction Extraction | 1-2s | Data processing, minimal LLM |
| Categorization | 0.5s per tx | Hybrid (keyword + LLM) |
| Subscription Detection | 3-5s | LLM validation of patterns |
| Metrics Calculation | <1s | Pure computation, no LLM |
| Expert Analysis | 5-10s | Large context, detailed output |
| **Total (100 transactions)** | **60-90s** | Varies with file size |

### Optimization Opportunities

LangSmith traces will reveal:
1. **Slow nodes**: Identify bottlenecks (usually Expert Analysis)
2. **Unnecessary reflections**: Check if reflection loops trigger too often
3. **Token waste**: Find prompts that can be shortened
4. **Parallel opportunities**: Identify independent LLM calls that can run concurrently

## Troubleshooting

### Traces Not Appearing

**Check 1: Environment variables loaded**
```python
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv("LANGCHAIN_TRACING_V2"))  # Should print "true"
```

**Check 2: API key valid**
- Go to smith.langchain.com
- Verify key exists and is active
- Check project name matches

**Check 3: Network connectivity**
```bash
curl -H "x-api-key: $LANGCHAIN_API_KEY" https://api.smith.langchain.com/
```

### Partial Traces

If only some components are traced:
- Ensure all agents use LangChain components (ChatGoogleGenerativeAI, prompts)
- Check for direct API calls that bypass LangChain
- Verify subgraphs are properly compiled

### High Token Usage

If costs are high:
1. Review prompts - remove verbose examples
2. Use Gemini Flash instead of Pro (already configured)
3. Implement caching for repeated categorizations
4. Batch similar transactions before LLM calls

## Best Practices

### 1. Name Your Runs

Add run names for easier tracking:

```python
from langsmith import traceable

@traceable(name="analyze_bank_statement", run_type="chain")
def analyze_statement(file_path: str):
    workflow = create_analysis_workflow()
    state = create_initial_state(file_path)
    return workflow.invoke(state)
```

### 2. Tag Important Runs

Tag runs with metadata:

```dotenv
LANGCHAIN_METADATA={"customer_id": "ABC123", "file_type": "credit_card"}
```

### 3. Monitor Errors

Set up alerts in LangSmith for:
- Error rate > 5%
- Latency > 120s
- Token usage > 10k per run

### 4. Review Weekly

Check LangSmith dashboard weekly for:
- Performance degradation
- New error patterns
- Cost trends

## Integration with Testing

Use LangSmith with the evaluation framework:

```python
from src.evals.evaluators import EndToEndEvaluator
from langsmith import Client

client = Client()
evaluator = EndToEndEvaluator()

# Run evaluation
results = evaluator.evaluate_pipeline(test_case, actual_results)

# Log to LangSmith
client.create_feedback(
    run_id=run_id,
    key="accuracy",
    score=results["categorization"]["overall_accuracy"]
)
```

## Cost Estimation

With Gemini 1.5 Flash (as of Dec 2025):
- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens

Typical run (100 transactions):
- Input tokens: ~8,000
- Output tokens: ~2,000
- **Cost per run**: ~$0.0012 (about 0.1 cent)

With LangSmith tracing enabled, add ~5-10ms overhead per LLM call (negligible).

## Documentation Links

- [LangSmith Docs](https://docs.smith.langchain.com/)
- [LangGraph Tracing](https://langchain-ai.github.io/langgraph/how-tos/tracing/)
- [LangChain Callbacks](https://python.langchain.com/docs/modules/callbacks/)

## Summary

✅ **LangSmith tracing is fully configured and ready to use**

Just add your API key to `.env` and all agent calls, node executions, and workflow steps will be automatically traced in the LangSmith dashboard.

No code changes needed - tracing works out of the box with LangChain/LangGraph!
