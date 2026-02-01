"""Kaggle MCP Server - Main entry point."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .kaggle_api import KaggleClient, KaggleConfig
from .experiment import ExperimentManager
from .ml_runtime import MLTrainer, TrainingConfig, ModelFactory, ModelType, FeatureEngineer
from .ml_runtime.features import detect_task_type
from .guardrails import SubmissionGuard, SubmissionConfig, CostController, CostConfig


# Global state
class ServerState:
    """Global server state."""
    kaggle_client: Optional[KaggleClient] = None
    experiment_manager: Optional[ExperimentManager] = None
    submission_guard: Optional[SubmissionGuard] = None
    cost_controller: Optional[CostController] = None
    trainer: Optional[MLTrainer] = None
    current_competition: Optional[str] = None
    current_experiment_id: Optional[str] = None
    data_dir: Path = Path(__file__).parent.parent / "kaggle_data"


state = ServerState()

# Create MCP server
server = Server("kaggle-mcp")


def get_tools() -> list[Tool]:
    """Define all available MCP tools."""
    return [
        Tool(
            name="list_competitions",
            description="List available Kaggle competitions. Filter by category or search term.",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Filter by category: featured, research, recruitment, gettingStarted, masters, playground",
                        "enum": ["featured", "research", "recruitment", "gettingStarted", "masters", "playground"],
                    },
                    "search": {
                        "type": "string",
                        "description": "Search term to filter competitions",
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number (default: 1)",
                        "default": 1,
                    },
                },
            },
        ),
        Tool(
            name="check_competition_access",
            description="Check if you have access to a competition (i.e., have accepted the rules). Call this before setup_competition if download fails.",
            inputSchema={
                "type": "object",
                "properties": {
                    "competition": {
                        "type": "string",
                        "description": "Competition reference (e.g., 'titanic')",
                    },
                },
                "required": ["competition"],
            },
        ),
        Tool(
            name="setup_competition",
            description="Set up a competition workspace. Downloads data and initializes experiment tracking. Note: You must have accepted the competition rules on Kaggle first.",
            inputSchema={
                "type": "object",
                "properties": {
                    "competition": {
                        "type": "string",
                        "description": "Competition reference (e.g., 'titanic', 'house-prices-advanced-regression-techniques')",
                    },
                    "description": {
                        "type": "string",
                        "description": "Description for this experiment",
                    },
                },
                "required": ["competition"],
            },
        ),
        Tool(
            name="analyze_data",
            description="Analyze the competition data and return statistics, column types, missing values, etc.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "Name of the data file to analyze (e.g., 'train.csv')",
                        "default": "train.csv",
                    },
                },
            },
        ),
        Tool(
            name="train_model",
            description="Train a machine learning model with cross-validation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "description": "Type of model to train",
                        "enum": [m.value for m in ModelType],
                    },
                    "target_column": {
                        "type": "string",
                        "description": "Name of the target column",
                    },
                    "hyperparameters": {
                        "type": "object",
                        "description": "Custom hyperparameters (optional)",
                    },
                    "feature_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific columns to use as features (optional, uses all non-target by default)",
                    },
                },
                "required": ["model_type", "target_column"],
            },
        ),
        Tool(
            name="tune_hyperparameters",
            description="Automatically tune hyperparameters using Optuna.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "description": "Type of model to tune",
                        "enum": ["xgboost", "lightgbm", "random_forest", "gradient_boosting"],
                    },
                    "target_column": {
                        "type": "string",
                        "description": "Name of the target column",
                    },
                    "n_trials": {
                        "type": "integer",
                        "description": "Number of optimization trials (default: 50)",
                        "default": 50,
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (optional)",
                    },
                },
                "required": ["model_type", "target_column"],
            },
        ),
        Tool(
            name="generate_predictions",
            description="Generate predictions on test data using a trained model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID of the trained model",
                    },
                    "test_file": {
                        "type": "string",
                        "description": "Test data file name (default: test.csv)",
                        "default": "test.csv",
                    },
                    "id_column": {
                        "type": "string",
                        "description": "ID column name for submission",
                    },
                    "prediction_column": {
                        "type": "string",
                        "description": "Name for prediction column in submission",
                    },
                },
                "required": ["run_id", "id_column", "prediction_column"],
            },
        ),
        Tool(
            name="submit_to_kaggle",
            description="Submit predictions to Kaggle. Requires approval and respects daily limits.",
            inputSchema={
                "type": "object",
                "properties": {
                    "submission_file": {
                        "type": "string",
                        "description": "Path to submission file",
                    },
                    "message": {
                        "type": "string",
                        "description": "Submission message/description",
                    },
                    "cv_score": {
                        "type": "number",
                        "description": "CV score for this submission",
                    },
                },
                "required": ["submission_file", "message"],
            },
        ),
        Tool(
            name="get_leaderboard",
            description="Get the competition leaderboard.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of entries to return (default: 20)",
                        "default": 20,
                    },
                },
            },
        ),
        Tool(
            name="get_my_submissions",
            description="Get your submission history for the current competition.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of submissions to return (default: 10)",
                        "default": 10,
                    },
                },
            },
        ),
        Tool(
            name="get_experiment_summary",
            description="Get a summary of the current experiment including all runs and best scores.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="get_status",
            description="Get current status including submission limits, training budget, and experiment state.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="configure_guardrails",
            description="Configure safety guardrails for submissions and training.",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_submissions_per_day": {
                        "type": "integer",
                        "description": "Maximum submissions per day",
                    },
                    "require_approval": {
                        "type": "boolean",
                        "description": "Require manual approval for submissions",
                    },
                    "dry_run_mode": {
                        "type": "boolean",
                        "description": "Enable dry run mode (no actual submissions)",
                    },
                    "max_training_time_seconds": {
                        "type": "integer",
                        "description": "Maximum training time per job in seconds",
                    },
                },
            },
        ),
        Tool(
            name="create_kaggle_notebook",
            description="Create a Kaggle notebook directly in your Kaggle account. The notebook will be linked to the current competition.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title for the notebook (e.g., 'Titanic Random Forest Baseline')",
                    },
                    "code": {
                        "type": "string",
                        "description": "Python code for the notebook. Will be converted to notebook cells.",
                    },
                    "is_private": {
                        "type": "boolean",
                        "description": "Whether the notebook should be private (default: False)",
                        "default": False,
                    },
                    "enable_gpu": {
                        "type": "boolean",
                        "description": "Enable GPU acceleration (default: False)",
                        "default": False,
                    },
                    "enable_internet": {
                        "type": "boolean",
                        "description": "Enable internet access in the notebook (default: True)",
                        "default": True,
                    },
                },
                "required": ["title", "code"],
            },
        ),
    ]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return get_tools()


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    try:
        result = await handle_tool_call(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        error_result = {"error": str(e), "tool": name}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]


async def handle_tool_call(name: str, arguments: dict[str, Any]) -> dict:
    """Route tool calls to handlers."""
    # Ensure initialized
    await ensure_initialized()

    handlers = {
        "list_competitions": handle_list_competitions,
        "check_competition_access": handle_check_competition_access,
        "setup_competition": handle_setup_competition,
        "analyze_data": handle_analyze_data,
        "train_model": handle_train_model,
        "tune_hyperparameters": handle_tune_hyperparameters,
        "generate_predictions": handle_generate_predictions,
        "submit_to_kaggle": handle_submit,
        "get_leaderboard": handle_get_leaderboard,
        "get_my_submissions": handle_get_submissions,
        "get_experiment_summary": handle_get_experiment_summary,
        "get_status": handle_get_status,
        "configure_guardrails": handle_configure_guardrails,
        "create_kaggle_notebook": handle_create_kaggle_notebook,
    }

    handler = handlers.get(name)
    if not handler:
        return {"error": f"Unknown tool: {name}"}

    return await handler(arguments)


async def ensure_initialized():
    """Ensure all components are initialized."""
    if state.kaggle_client is None:
        state.kaggle_client = KaggleClient(KaggleConfig(data_dir=state.data_dir))
        await state.kaggle_client.authenticate()

    if state.experiment_manager is None:
        state.experiment_manager = ExperimentManager(state.data_dir / "experiments.db")

    if state.submission_guard is None:
        state.submission_guard = SubmissionGuard(
            SubmissionConfig(state_file=state.data_dir / ".submission_state.json")
        )

    if state.cost_controller is None:
        state.cost_controller = CostController(
            CostConfig(state_file=state.data_dir / ".cost_state.json")
        )

    if state.trainer is None:
        state.trainer = MLTrainer(
            TrainingConfig(artifact_dir=state.data_dir / "models"),
            timeout_callback=lambda: state.cost_controller.check_job_timeout("current")[0],
        )


async def handle_list_competitions(args: dict) -> dict:
    """List competitions."""
    competitions = await state.kaggle_client.list_competitions(
        category=args.get("category"),
        search=args.get("search"),
        page=args.get("page", 1),
    )

    return {
        "competitions": [
            {
                "ref": c.ref,
                "title": c.title,
                "category": c.category,
                "reward": c.reward,
                "deadline": c.deadline.isoformat() if c.deadline else None,
                "team_count": c.team_count,
                "evaluation_metric": c.evaluation_metric,
            }
            for c in competitions
        ],
        "count": len(competitions),
    }


async def handle_check_competition_access(args: dict) -> dict:
    """Check if user has access to a competition."""
    competition = args["competition"]
    result = await state.kaggle_client.check_competition_access(competition)

    if not result.get("has_access"):
        return {
            "has_access": False,
            "message": "You must accept the competition rules before downloading data.",
            "action_required": f"Please visit the link below and click 'I Understand and Accept' or 'Join Competition'",
            "join_url": result.get("join_url", f"https://www.kaggle.com/competitions/{competition}/rules"),
        }

    return {
        "has_access": True,
        "message": "You have access to this competition's data.",
        "file_count": result.get("file_count", 0),
    }


def get_competition_dir(competition: str) -> Path:
    """Get the base directory for a competition."""
    return state.data_dir / competition


def get_competition_data_dir(competition: str) -> Path:
    """Get the data directory for a competition."""
    return get_competition_dir(competition) / "data"


def get_competition_models_dir(competition: str) -> Path:
    """Get the models directory for a competition."""
    return get_competition_dir(competition) / "models"


def get_competition_scripts_dir(competition: str) -> Path:
    """Get the scripts directory for a competition."""
    return get_competition_dir(competition) / "scripts"


def get_competition_submissions_dir(competition: str) -> Path:
    """Get the submissions directory for a competition."""
    return get_competition_dir(competition) / "submissions"


async def handle_setup_competition(args: dict) -> dict:
    """Set up a competition."""
    competition = args["competition"]

    # Create competition directory structure
    comp_dir = get_competition_dir(competition)
    data_dir = get_competition_data_dir(competition)
    models_dir = get_competition_models_dir(competition)
    scripts_dir = get_competition_scripts_dir(competition)
    submissions_dir = get_competition_submissions_dir(competition)

    for d in [data_dir, models_dir, scripts_dir, submissions_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Download competition data to the data subdirectory
    data_path = await state.kaggle_client.download_competition_data(
        competition,
        path=data_dir,
    )

    # List files
    files = await state.kaggle_client.list_competition_files(competition)

    # Create experiment (store DB in competition folder)
    exp_db_path = comp_dir / "experiments.db"
    state.experiment_manager = ExperimentManager(exp_db_path)

    experiment = await state.experiment_manager.create_experiment(
        competition=competition,
        description=args.get("description", f"Experiment for {competition}"),
        data_version="v1",
    )

    # Update trainer to save models in competition's models folder
    state.trainer = MLTrainer(
        TrainingConfig(artifact_dir=models_dir),
        timeout_callback=lambda: state.cost_controller.check_job_timeout("current")[0],
    )

    state.current_competition = competition
    state.current_experiment_id = experiment.experiment_id

    return {
        "status": "success",
        "competition": competition,
        "experiment_id": experiment.experiment_id,
        "data_path": str(data_path),
        "models_path": str(models_dir),
        "scripts_path": str(scripts_dir),
        "submissions_path": str(submissions_dir),
        "files": files,
    }


async def handle_analyze_data(args: dict) -> dict:
    """Analyze competition data."""
    if not state.current_competition:
        return {"error": "No competition set up. Call setup_competition first."}

    import pandas as pd

    file_name = args.get("file_name", "train.csv")
    data_dir = get_competition_data_dir(state.current_competition)
    file_path = data_dir / file_name

    if not file_path.exists():
        # Try looking in zip extracted folder or parent
        for p in get_competition_dir(state.current_competition).rglob(file_name):
            file_path = p
            break

    if not file_path.exists():
        return {"error": f"File not found: {file_name}"}

    df = pd.read_csv(file_path)

    engineer = FeatureEngineer()
    analysis = engineer.analyze_dataframe(df)

    return {
        "file": file_name,
        "analysis": analysis,
    }


async def handle_train_model(args: dict) -> dict:
    """Train a model."""
    if not state.current_competition:
        return {"error": "No competition set up. Call setup_competition first."}

    # Check cost controller
    can_train, msg = state.cost_controller.can_start_training()
    if not can_train:
        return {"error": f"Training not allowed: {msg}"}

    import pandas as pd

    # Load training data
    data_dir = get_competition_data_dir(state.current_competition)
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        for p in get_competition_dir(state.current_competition).rglob("train.csv"):
            train_path = p
            break

    df = pd.read_csv(train_path)

    model_type = ModelType(args["model_type"])
    target_column = args["target_column"]

    # Create experiment run
    run = await state.experiment_manager.create_run(
        experiment_id=state.current_experiment_id,
        model_type=model_type.value,
        hyperparameters=args.get("hyperparameters", {}),
        features_used=args.get("feature_columns", df.columns.tolist()),
    )

    # Start cost tracking
    job_id = run.run_id
    state.cost_controller.start_job(
        job_id,
        model_type=model_type.value,
        competition=state.current_competition,
    )

    try:
        # Train model
        result = await state.trainer.train(
            train_df=df,
            target_column=target_column,
            model_type=model_type,
            hyperparameters=args.get("hyperparameters"),
            feature_columns=args.get("feature_columns"),
        )

        # Update experiment run
        await state.experiment_manager.update_run(
            run_id=run.run_id,
            cv_score=result.cv_score,
            cv_std=result.cv_std,
            training_time_seconds=result.training_time_seconds,
            status="completed",
            artifact_path=result.artifact_path,
        )

        state.cost_controller.end_job(job_id, status="completed")

        return {
            "status": "success",
            "run_id": result.run_id,
            "model_type": result.model_type,
            "cv_score": result.cv_score,
            "cv_std": result.cv_std,
            "cv_scores": result.cv_scores,
            "training_time_seconds": result.training_time_seconds,
            "feature_importance": dict(
                sorted(
                    (result.feature_importance or {}).items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),
            "artifact_path": result.artifact_path,
        }

    except Exception as e:
        state.cost_controller.end_job(job_id, status="failed")
        await state.experiment_manager.update_run(
            run_id=run.run_id,
            status="failed",
            notes=str(e),
        )
        return {"error": f"Training failed: {e}"}


async def handle_tune_hyperparameters(args: dict) -> dict:
    """Tune hyperparameters."""
    if not state.current_competition:
        return {"error": "No competition set up. Call setup_competition first."}

    import pandas as pd

    data_dir = get_competition_data_dir(state.current_competition)
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        for p in get_competition_dir(state.current_competition).rglob("train.csv"):
            train_path = p
            break

    df = pd.read_csv(train_path)

    model_type = ModelType(args["model_type"])
    target_column = args["target_column"]

    result = await state.trainer.tune_hyperparameters(
        train_df=df,
        target_column=target_column,
        model_type=model_type,
        n_trials=args.get("n_trials", 50),
        timeout=args.get("timeout"),
    )

    return {
        "status": "success",
        "model_type": model_type.value,
        "best_params": result["best_params"],
        "best_score": result["best_score"],
        "n_trials": result["n_trials"],
    }


async def handle_generate_predictions(args: dict) -> dict:
    """Generate predictions."""
    if not state.current_competition:
        return {"error": "No competition set up. Call setup_competition first."}

    import pandas as pd
    import numpy as np

    run_id = args["run_id"]
    test_file = args.get("test_file", "test.csv")
    id_column = args["id_column"]
    prediction_column = args["prediction_column"]

    # Find test file
    data_dir = get_competition_data_dir(state.current_competition)
    test_path = data_dir / test_file
    if not test_path.exists():
        for p in get_competition_dir(state.current_competition).rglob(test_file):
            test_path = p
            break

    if not test_path.exists():
        return {"error": f"Test file not found: {test_file}"}

    # Find model artifact
    models_dir = get_competition_models_dir(state.current_competition)
    run = await state.experiment_manager.get_run(run_id)
    if not run or not run.artifact_path:
        # Try to find by run_id prefix in competition's models folder
        for p in models_dir.glob(f"{run_id}*.pkl"):
            artifact_path = str(p)
            break
        else:
            return {"error": f"Model artifact not found for run: {run_id}"}
    else:
        artifact_path = run.artifact_path

    test_df = pd.read_csv(test_path)
    ids = test_df[id_column].values

    # Generate predictions
    predictions = await state.trainer.predict(
        test_df=test_df.drop(columns=[id_column], errors='ignore'),
        artifact_path=artifact_path,
    )

    # Create submission file
    submissions_dir = get_competition_submissions_dir(state.current_competition)
    submission_df = pd.DataFrame({
        id_column: ids,
        prediction_column: predictions,
    })

    submission_path = submissions_dir / f"submission_{run_id}.csv"
    submission_df.to_csv(submission_path, index=False)

    return {
        "status": "success",
        "submission_file": str(submission_path),
        "rows": len(submission_df),
        "prediction_stats": {
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
        },
    }


async def handle_submit(args: dict) -> dict:
    """Submit to Kaggle."""
    if not state.current_competition:
        return {"error": "No competition set up. Call setup_competition first."}

    submission_file = Path(args["submission_file"])
    message = args["message"]
    cv_score = args.get("cv_score")

    # Check guardrails
    can_submit, reason = await state.submission_guard.check_can_submit(
        state.current_competition,
        cv_score=cv_score,
    )

    if not can_submit:
        return {
            "status": "blocked",
            "reason": reason,
            "submission_summary": state.submission_guard.get_submission_summary(state.current_competition),
        }

    # Request approval
    approved, approval_msg = await state.submission_guard.request_approval(
        state.current_competition,
        str(submission_file),
        cv_score,
    )

    if not approved:
        return {
            "status": "not_approved",
            "reason": approval_msg,
        }

    # Submit
    submission = await state.kaggle_client.submit(
        competition=state.current_competition,
        file_path=submission_file,
        message=message,
    )

    # Record submission
    await state.submission_guard.record_submission(
        competition=state.current_competition,
        file_path=str(submission_file),
        cv_score=cv_score,
        approved=True,
        submitted=True,
        message=message,
    )

    return {
        "status": "success",
        "submission_id": submission.id,
        "message": message,
        "submission_summary": state.submission_guard.get_submission_summary(state.current_competition),
    }


async def handle_get_leaderboard(args: dict) -> dict:
    """Get leaderboard."""
    if not state.current_competition:
        return {"error": "No competition set up. Call setup_competition first."}

    leaderboard = await state.kaggle_client.get_leaderboard(
        state.current_competition,
        limit=args.get("limit", 20),
    )

    return {
        "competition": state.current_competition,
        "leaderboard": [
            {
                "rank": e.rank,
                "team_name": e.team_name,
                "score": e.score,
            }
            for e in leaderboard
        ],
    }


async def handle_get_submissions(args: dict) -> dict:
    """Get submission history."""
    if not state.current_competition:
        return {"error": "No competition set up. Call setup_competition first."}

    submissions = await state.kaggle_client.list_submissions(
        state.current_competition,
        limit=args.get("limit", 10),
    )

    return {
        "competition": state.current_competition,
        "submissions": [
            {
                "id": s.id,
                "date": s.date.isoformat(),
                "description": s.description,
                "status": s.status,
                "public_score": s.public_score,
            }
            for s in submissions
        ],
    }


async def handle_get_experiment_summary(args: dict) -> dict:
    """Get experiment summary."""
    if not state.current_experiment_id:
        return {"error": "No experiment set up. Call setup_competition first."}

    summary = await state.experiment_manager.get_experiment_summary(
        state.current_experiment_id
    )

    runs = await state.experiment_manager.get_runs_for_experiment(
        state.current_experiment_id,
        limit=10,
    )

    return {
        "summary": summary,
        "recent_runs": [
            {
                "run_id": r.run_id,
                "model_type": r.model_type,
                "cv_score": r.cv_score,
                "cv_std": r.cv_std,
                "status": r.status,
                "created_at": r.created_at.isoformat(),
            }
            for r in runs
        ],
    }


async def handle_get_status(args: dict) -> dict:
    """Get current status."""
    status = {
        "current_competition": state.current_competition,
        "current_experiment_id": state.current_experiment_id,
        "cost_status": state.cost_controller.get_status(),
    }

    if state.current_competition:
        status["submission_status"] = state.submission_guard.get_submission_summary(
            state.current_competition
        )

    return status


async def handle_configure_guardrails(args: dict) -> dict:
    """Configure guardrails."""
    if "max_submissions_per_day" in args:
        state.submission_guard.config.max_submissions_per_day = args["max_submissions_per_day"]

    if "require_approval" in args:
        state.submission_guard.config.require_approval = args["require_approval"]

    if "dry_run_mode" in args:
        state.submission_guard.config.dry_run_mode = args["dry_run_mode"]

    if "max_training_time_seconds" in args:
        state.cost_controller.config.max_training_time_seconds = args["max_training_time_seconds"]

    return {
        "status": "success",
        "submission_config": {
            "max_submissions_per_day": state.submission_guard.config.max_submissions_per_day,
            "require_approval": state.submission_guard.config.require_approval,
            "dry_run_mode": state.submission_guard.config.dry_run_mode,
        },
        "cost_config": {
            "max_training_time_seconds": state.cost_controller.config.max_training_time_seconds,
            "max_total_training_time_per_day": state.cost_controller.config.max_total_training_time_per_day,
        },
    }


async def handle_create_kaggle_notebook(args: dict) -> dict:
    """Create a Kaggle notebook directly in the user's account."""
    if not state.current_competition:
        return {"error": "No competition set up. Call setup_competition first."}

    title = args["title"]
    code = args["code"]
    is_private = args.get("is_private", False)
    enable_gpu = args.get("enable_gpu", False)
    enable_internet = args.get("enable_internet", True)

    # Get username from environment
    username = os.environ.get("KAGGLE_USERNAME")
    if not username:
        return {"error": "KAGGLE_USERNAME environment variable not set. Add it to your claude_desktop_config.json"}

    # Convert code to notebook format
    # Split code into cells by looking for markdown comments or double newlines
    cells = []

    # Add title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"# {title}\n", f"\n", f"Competition: {state.current_competition}\n"]
    })

    # Split code into logical cells
    code_blocks = []
    current_block = []
    lines = code.split('\n')

    for line in lines:
        # Start new cell on imports, function definitions, or class definitions
        if current_block and (
            line.startswith('import ') or
            line.startswith('from ') or
            line.startswith('def ') or
            line.startswith('class ') or
            line.startswith('# %%') or
            line.startswith('# CELL:')
        ):
            if current_block:
                code_blocks.append('\n'.join(current_block))
                current_block = []
        current_block.append(line)

    if current_block:
        code_blocks.append('\n'.join(current_block))

    # If no natural splits, just use the whole code as one cell
    if not code_blocks:
        code_blocks = [code]

    # Convert code blocks to cells
    for block in code_blocks:
        if block.strip():
            # Convert block to source array format
            source_lines = [line + '\n' for line in block.split('\n')]
            if source_lines:
                source_lines[-1] = source_lines[-1].rstrip('\n')  # Last line no newline
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source_lines
            })

    notebook_json = json.dumps({
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    })

    # Create slug from title
    slug = title.lower().replace(' ', '-').replace('_', '-')
    # Remove special characters
    slug = ''.join(c for c in slug if c.isalnum() or c == '-')

    try:
        from kagglesdk import KaggleClient as SdkClient
        from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest

        api_token = os.environ.get("KAGGLE_API_TOKEN")
        competition = state.current_competition

        def _create_notebook():
            with SdkClient(api_token=api_token) as client:
                request = ApiSaveKernelRequest()
                request.slug = f"{username}/{slug}"
                request.new_title = title
                request.text = notebook_json
                request.language = "python"
                request.kernel_type = "notebook"
                request.is_private = is_private
                request.enable_gpu = enable_gpu
                request.enable_tpu = False
                request.enable_internet = enable_internet
                request.competition_data_sources = [competition]

                return client.kernels.kernels_api_client.save_kernel(request)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _create_notebook)

        # Also save locally
        scripts_dir = get_competition_scripts_dir(state.current_competition)
        local_path = scripts_dir / f"{slug}.ipynb"
        with open(local_path, 'w') as f:
            f.write(notebook_json)

        return {
            "status": "success",
            "title": title,
            "slug": f"{username}/{slug}",
            "url": response.url if hasattr(response, 'url') and response.url else f"https://www.kaggle.com/code/{username}/{slug}",
            "version": response.version_number if hasattr(response, 'version_number') else 1,
            "local_path": str(local_path),
            "error": response.error if hasattr(response, 'error') and response.error else None,
        }

    except Exception as e:
        return {"error": f"Failed to create notebook: {str(e)}"}


async def run_server():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main():
    """Main entry point."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
