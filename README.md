# Kaggle MCP Server

An MCP (Model Context Protocol) server for autonomous Kaggle competition participation with built-in safety guardrails.

## Features

- **Competition Management**: List, search, and set up Kaggle competitions
- **Data Analysis**: Automatic EDA with statistics and column analysis
- **Model Training**: Cross-validated training with multiple model types
- **Hyperparameter Tuning**: Optuna-based automatic hyperparameter optimization
- **Prediction Generation**: Generate and format submission files
- **Submission Management**: Submit to Kaggle with safety controls
- **Experiment Tracking**: SQLite-based experiment and run tracking

## Safety Guardrails

- Daily submission limits (configurable)
- Approval requirement for submissions
- Dry-run mode for testing
- Training time limits
- Cost controls for compute resources

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd kaggle-mcp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Configuration

### Kaggle API Credentials

Set up your Kaggle API credentials:

1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Place `kaggle.json` in `~/.kaggle/`

Or set environment variables:
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

### MCP Client Configuration

Add to your MCP client configuration (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "kaggle": {
      "command": "python",
      "args": ["-m", "kaggle_mcp.server"],
      "cwd": "/path/to/kaggle-mcp"
    }
  }
}
```

## Available Tools

### Competition Tools

| Tool | Description |
|------|-------------|
| `list_competitions` | List available Kaggle competitions |
| `setup_competition` | Set up workspace for a competition |
| `get_leaderboard` | View competition leaderboard |
| `get_my_submissions` | View your submission history |

### Data & Training Tools

| Tool | Description |
|------|-------------|
| `analyze_data` | Analyze competition data |
| `train_model` | Train a model with cross-validation |
| `tune_hyperparameters` | Auto-tune hyperparameters |
| `generate_predictions` | Generate test predictions |

### Submission Tools

| Tool | Description |
|------|-------------|
| `submit_to_kaggle` | Submit predictions (guarded) |
| `get_status` | View current status and limits |
| `configure_guardrails` | Adjust safety settings |

## Supported Models

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- Linear/Ridge/Lasso Regression
- SVM
- KNN

## Example Workflow

```
1. list_competitions(search="titanic")
2. setup_competition(competition="titanic")
3. analyze_data(file_name="train.csv")
4. train_model(model_type="xgboost", target_column="Survived")
5. tune_hyperparameters(model_type="xgboost", target_column="Survived", n_trials=30)
6. train_model(model_type="xgboost", target_column="Survived", hyperparameters={...})
7. generate_predictions(run_id="abc123", id_column="PassengerId", prediction_column="Survived")
8. submit_to_kaggle(submission_file="path/to/submission.csv", message="XGBoost tuned")
```

## Project Structure

```
kaggle-mcp/
├── kaggle_mcp/
│   ├── __init__.py
│   ├── server.py          # MCP server and tools
│   ├── kaggle_api/        # Kaggle API wrapper
│   │   ├── client.py
│   │   └── models.py
│   ├── ml_runtime/        # ML training
│   │   ├── trainer.py
│   │   ├── models.py
│   │   └── features.py
│   ├── experiment/        # Experiment tracking
│   │   └── manager.py
│   └── guardrails/        # Safety controls
│       ├── submission.py
│       └── cost.py
├── main.py
├── pyproject.toml
└── README.md
```

## License

MIT
