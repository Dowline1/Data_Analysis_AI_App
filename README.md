# Assignment 3 - Bank Account Health Analyzer

**Student Name:** Eoin Dowling  
**Student Number:** D00295535  
**Module:** Foundations of Agentic AI Systems  
**Assignment:** CA3 - Multi-Agent Data Analysis System

---

## Overview

A smart AI agent system that analyzes your bank statements and provides financial insights. Upload your bank statement (PDF, CSV, or Excel) and get instant analysis of your spending patterns, subscriptions, and financial health.

## Features

### Core Functionality
- 📄 **Universal File Support** - Works with PDF, CSV, and Excel bank statements from any bank
- 🤖 **Multi-Agent System** - Specialized agents handle different parts of the analysis
- ✅ **Human-in-the-Loop** - You review and approve the extracted data before analysis
- 🛡️ **Safety Guardrails** - Input validation, PII protection, and data quality checks
- 💡 **Smart Insights** - AI-powered recommendations based on your spending
- 📊 **Interactive Dashboard** - Visual charts and metrics via Streamlit UI

### Agent Architecture
- **Data Ingestion Subgraph** - Parses files, maps columns, validates data
- **Analysis Subgraph** - Categorizes transactions, detects subscriptions, calculates metrics
- **Reflection Loops** - Self-corrects errors and validates results
- **HITL Checkpoints** - Human review before final analysis

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Dowline1/Data_Analysis_AI_App.git
cd Data_Analysis_AI_App
```

2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Mac/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

### Run the Streamlit App
```bash
streamlit run app.py
```

Then:
1. Upload your bank statement (PDF, CSV, or Excel)
2. Review the extracted transactions
3. Confirm the data is correct
4. Get instant financial insights!

## Project Structure

```
Data_Analysis_AI_App/
├── src/
│   ├── agents/          # Agent implementations
│   ├── tools/           # File parsing tools
│   ├── guardrails/      # Safety and validation
│   ├── graphs/          # LangGraph workflows
│   ├── state/           # State management
│   └── utils/           # Config and logging
├── data/
│   └── sample_statements/  # Test files
├── tests/               # Unit tests
├── evals/               # Evaluation scripts
└── app.py              # Streamlit interface
```

## Status

🚧 **In Development**

Currently implemented:
- ✅ Project structure
- ✅ PDF/CSV/Excel file parser
- ✅ Configuration management
- ✅ Logging system
- ✅ State definitions
- 🚧 LangGraph agents (in progress)
- 🚧 Streamlit UI (in progress)

---

*More documentation will be added as development progresses.*
