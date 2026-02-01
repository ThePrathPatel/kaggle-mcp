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
from .guardrails import SubmissionGuard, SubmissionConfig


# Global state
class ServerState:
    """Global server state."""
    kaggle_client: Optional[KaggleClient] = None
    submission_guard: Optional[SubmissionGuard] = None
    current_competition: Optional[str] = None
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
            description="Check if you have access to a competition (i.e., have accepted the rules). Call this before creating a notebook.",
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
            name="select_competition",
            description="Select a competition to work with. This sets the current competition context for notebook creation and submissions. The notebook will have access to competition data via /kaggle/input/{competition}/.",
            inputSchema={
                "type": "object",
                "properties": {
                    "competition": {
                        "type": "string",
                        "description": "Competition reference (e.g., 'titanic', 'house-prices-advanced-regression-techniques')",
                    },
                },
                "required": ["competition"],
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
            name="configure_guardrails",
            description="Configure safety guardrails for submissions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_submissions_per_day": {
                        "type": "integer",
                        "description": "Maximum submissions per day",
                    },
                    "dry_run_mode": {
                        "type": "boolean",
                        "description": "Enable dry run mode (no actual submissions)",
                    },
                },
            },
        ),
        Tool(
            name="create_kaggle_notebook",
            description="Create a Kaggle notebook for training and submission. The notebook runs on Kaggle's servers (with optional GPU) and is automatically linked to the current competition. Data is available at /kaggle/input/{competition}/. Set auto_run=True and auto_submit=True for fully automated workflow.",
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
                    "auto_run": {
                        "type": "boolean",
                        "description": "Automatically run the notebook after creation (default: False). If True, waits for execution to complete.",
                        "default": False,
                    },
                    "auto_submit": {
                        "type": "boolean",
                        "description": "Automatically submit the notebook output after execution (default: False). Requires auto_run=True. Expects the notebook to produce 'submission.csv'.",
                        "default": False,
                    },
                    "submission_message": {
                        "type": "string",
                        "description": "Message for the submission (required if auto_submit=True)",
                    },
                },
                "required": ["title", "code"],
            },
        ),
        Tool(
            name="run_kaggle_notebook",
            description="Execute a Kaggle notebook and wait for it to complete. Returns the execution status and output files. Use this after 'create_kaggle_notebook' to run the notebook on Kaggle's servers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "notebook_slug": {
                        "type": "string",
                        "description": "The notebook slug (e.g., 'username/notebook-title')",
                    },
                    "timeout_minutes": {
                        "type": "integer",
                        "description": "Maximum time to wait for completion in minutes (default: 60)",
                        "default": 60,
                    },
                },
                "required": ["notebook_slug"],
            },
        ),
        Tool(
            name="submit_notebook_output",
            description="Submit a notebook's output file to the Kaggle competition. Use this after 'run_kaggle_notebook' completes to submit the predictions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "notebook_slug": {
                        "type": "string",
                        "description": "The notebook slug (e.g., 'username/notebook-title')",
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Name of the output file to submit (e.g., 'submission.csv')",
                        "default": "submission.csv",
                    },
                    "message": {
                        "type": "string",
                        "description": "Submission message/description",
                    },
                },
                "required": ["notebook_slug", "message"],
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
        "select_competition": handle_select_competition,
        "get_leaderboard": handle_get_leaderboard,
        "get_my_submissions": handle_get_submissions,
        "configure_guardrails": handle_configure_guardrails,
        "create_kaggle_notebook": handle_create_kaggle_notebook,
        "run_kaggle_notebook": handle_run_kaggle_notebook,
        "submit_notebook_output": handle_submit_notebook_output,
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

    if state.submission_guard is None:
        state.submission_guard = SubmissionGuard(
            SubmissionConfig(state_file=state.data_dir / ".submission_state.json")
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


def get_competition_scripts_dir(competition: str) -> Path:
    """Get the scripts directory for a competition."""
    scripts_dir = state.data_dir / competition / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    return scripts_dir


async def handle_select_competition(args: dict) -> dict:
    """Select a competition to work with."""
    competition = args["competition"]
    state.current_competition = competition

    return {
        "status": "success",
        "competition": competition,
        "message": f"Selected competition: {competition}. Notebooks will have access to data at /kaggle/input/{competition}/",
    }


async def handle_get_leaderboard(args: dict) -> dict:
    """Get leaderboard."""
    if not state.current_competition:
        return {"error": "No competition selected. Call select_competition first."}

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
        return {"error": "No competition selected. Call select_competition first."}

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


async def handle_configure_guardrails(args: dict) -> dict:
    """Configure guardrails."""
    if "max_submissions_per_day" in args:
        state.submission_guard.config.max_submissions_per_day = args["max_submissions_per_day"]

    if "dry_run_mode" in args:
        state.submission_guard.config.dry_run_mode = args["dry_run_mode"]

    return {
        "status": "success",
        "submission_config": {
            "max_submissions_per_day": state.submission_guard.config.max_submissions_per_day,
            "dry_run_mode": state.submission_guard.config.dry_run_mode,
        },
    }


async def handle_create_kaggle_notebook(args: dict) -> dict:
    """Create a Kaggle notebook directly in the user's account."""
    if not state.current_competition:
        return {"error": "No competition selected. Call select_competition first."}

    title = args["title"]
    code = args["code"]
    is_private = args.get("is_private", False)
    enable_gpu = args.get("enable_gpu", False)
    enable_internet = args.get("enable_internet", True)
    auto_run = args.get("auto_run", False)
    auto_submit = args.get("auto_submit", False)
    submission_message = args.get("submission_message", "")

    if auto_submit and not auto_run:
        return {"error": "auto_submit requires auto_run=True"}
    if auto_submit and not submission_message:
        return {"error": "submission_message is required when auto_submit=True"}

    # Get username from kaggle.json or environment
    username = os.environ.get("KAGGLE_USERNAME")
    if not username:
        # Try to read from ~/.kaggle/kaggle.json
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.exists():
            with open(kaggle_json) as f:
                creds = json.load(f)
                username = creds.get("username")
    if not username:
        return {"error": "Kaggle username not found. Set KAGGLE_USERNAME env var or ensure ~/.kaggle/kaggle.json contains 'username'"}

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
    # Collapse multiple dashes into single dash (Kaggle normalizes these)
    while '--' in slug:
        slug = slug.replace('--', '-')
    # Remove leading/trailing dashes
    slug = slug.strip('-')

    try:
        from kagglesdk import KaggleClient as SdkClient
        from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest

        # Get credentials from kaggle.json
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        with open(kaggle_json) as f:
            creds = json.load(f)
        api_key = creds.get("key")
        kaggle_username = creds.get("username")

        competition = state.current_competition

        def _create_notebook():
            # Use api_token if key starts with KGAT_, otherwise use username/password auth
            if api_key and api_key.startswith("KGAT_"):
                client_ctx = SdkClient(api_token=api_key)
            else:
                client_ctx = SdkClient(username=kaggle_username, password=api_key)
            with client_ctx as client:
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

        notebook_slug = f"{username}/{slug}"
        result = {
            "status": "success",
            "title": title,
            "slug": notebook_slug,
            "url": response.url if hasattr(response, 'url') and response.url else f"https://www.kaggle.com/code/{username}/{slug}",
            "version": response.version_number if hasattr(response, 'version_number') else 1,
            "local_path": str(local_path),
            "error": response.error if hasattr(response, 'error') and response.error else None,
        }

        # Auto-run if requested
        if auto_run:
            run_result = await handle_run_kaggle_notebook({
                "notebook_slug": notebook_slug,
                "timeout_minutes": 60,
            })
            result["run_result"] = run_result

            # Auto-submit if requested
            if auto_submit:
                run_status = run_result.get("status", "")
                run_error = run_result.get("error", "")

                # Try to submit if run succeeded OR if status check failed (notebook may have succeeded)
                should_try_submit = (
                    run_status == "complete" or
                    "403" in str(run_error) or  # Status check permission denied
                    "timeout" in run_status.lower()  # Timeout but notebook might have finished
                )

                if should_try_submit:
                    submit_result = await handle_submit_notebook_output({
                        "notebook_slug": notebook_slug,
                        "output_file": "submission.csv",
                        "message": submission_message,
                    })
                    result["submit_result"] = submit_result
                elif run_status == "error":
                    result["submit_result"] = {
                        "error": f"Cannot auto-submit: notebook execution failed with status '{run_status}'"
                    }
                else:
                    result["submit_result"] = {
                        "error": f"Cannot auto-submit: unexpected run status '{run_status}'"
                    }

        return result

    except Exception as e:
        return {"error": f"Failed to create notebook: {str(e)}"}


def _get_kaggle_sdk_client():
    """Helper to create authenticated Kaggle SDK client."""
    from kagglesdk import KaggleClient as SdkClient
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    with open(kaggle_json) as f:
        creds = json.load(f)
    api_key = creds.get("key")
    kaggle_username = creds.get("username")

    if api_key and api_key.startswith("KGAT_"):
        return SdkClient(api_token=api_key)
    else:
        return SdkClient(username=kaggle_username, password=api_key)


async def handle_run_kaggle_notebook(args: dict) -> dict:
    """Run a Kaggle notebook and wait for completion."""
    import time

    notebook_slug = args["notebook_slug"]
    timeout_minutes = args.get("timeout_minutes", 60)

    try:
        from kagglesdk.kernels.types.kernels_api_service import ApiCreateKernelSessionRequest

        def _run_notebook():
            with _get_kaggle_sdk_client() as client:
                # Start the kernel session (run the notebook)
                request = ApiCreateKernelSessionRequest()
                request.slug = notebook_slug

                response = client.kernels.kernels_api_client.create_kernel_session(request)
                return response

        loop = asyncio.get_event_loop()
        run_response = await loop.run_in_executor(None, _run_notebook)

        # Poll for completion
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        poll_interval = 30  # Check every 30 seconds

        def _check_status():
            with _get_kaggle_sdk_client() as client:
                return client.kernels.kernels_api_client.get_kernel_session_status(notebook_slug)

        final_status = None
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                return {
                    "status": "timeout",
                    "message": f"Notebook execution timed out after {timeout_minutes} minutes",
                    "notebook_slug": notebook_slug,
                }

            status_response = await loop.run_in_executor(None, _check_status)
            status = status_response.status if hasattr(status_response, 'status') else str(status_response)

            # Check if completed (status values: queued, running, complete, error, cancelAcknowledged)
            if status in ["complete", "error", "cancelAcknowledged"]:
                final_status = status
                break

            await asyncio.sleep(poll_interval)

        # Get output files
        def _get_outputs():
            with _get_kaggle_sdk_client() as client:
                return client.kernels.kernels_api_client.list_kernel_session_output(notebook_slug)

        outputs_response = await loop.run_in_executor(None, _get_outputs)
        output_files = []
        if hasattr(outputs_response, 'files'):
            output_files = [f.name if hasattr(f, 'name') else str(f) for f in outputs_response.files]

        return {
            "status": final_status,
            "notebook_slug": notebook_slug,
            "output_files": output_files,
            "execution_time_seconds": int(time.time() - start_time),
            "message": "Notebook execution completed" if final_status == "complete" else f"Notebook ended with status: {final_status}",
        }

    except Exception as e:
        return {"error": f"Failed to run notebook: {str(e)}"}


async def handle_submit_notebook_output(args: dict) -> dict:
    """Submit a notebook's output file to the competition."""
    import requests

    if not state.current_competition:
        return {"error": "No competition set up. Call setup_competition first."}

    notebook_slug = args["notebook_slug"]
    output_file = args.get("output_file", "submission.csv")
    message = args["message"]

    try:
        # Get API token from kaggle.json
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        with open(kaggle_json) as f:
            creds = json.load(f)
        api_token = creds.get("api_token")
        if not api_token:
            return {"error": "No api_token found in kaggle.json. Please generate a new API token from Kaggle settings."}

        # Parse username and kernel slug
        if "/" in notebook_slug:
            user_name, kernel_slug = notebook_slug.split("/", 1)
        else:
            return {"error": f"Invalid notebook_slug format: '{notebook_slug}'. Expected 'username/kernel-name'"}

        # Use Bearer token auth to get output file URLs
        headers = {"Authorization": f"Bearer {api_token}"}
        output_url = f"https://www.kaggle.com/api/v1/kernels/output?userName={user_name}&kernelSlug={kernel_slug}"

        loop = asyncio.get_event_loop()

        def _get_output_info():
            response = requests.get(output_url, headers=headers)
            response.raise_for_status()
            return response.json()

        output_info = await loop.run_in_executor(None, _get_output_info)

        # Find the requested file in the output
        files = output_info.get("files", [])
        file_url = None
        for f in files:
            if f.get("fileName") == output_file or f.get("fileNameNullable") == output_file:
                file_url = f.get("url") or f.get("urlNullable")
                break

        if not file_url:
            available_files = [f.get("fileName", f.get("fileNameNullable", "unknown")) for f in files]
            return {"error": f"Output file '{output_file}' not found. Available files: {available_files}"}

        # Download the file
        def _download_file():
            response = requests.get(file_url)
            response.raise_for_status()
            return response.content

        content = await loop.run_in_executor(None, _download_file)

        # Save to local submissions directory
        submissions_dir = get_competition_submissions_dir(state.current_competition)
        local_file = submissions_dir / f"notebook_{output_file}"

        with open(local_file, 'wb') as f:
            f.write(content)

        # Submit directly using the Kaggle client
        full_message = f"[Notebook: {notebook_slug}] {message}"

        # Check guardrails
        can_submit, reason = await state.submission_guard.check_can_submit(
            state.current_competition,
            cv_score=None,
        )

        if not can_submit:
            return {
                "status": "blocked",
                "reason": reason,
                "notebook_slug": notebook_slug,
                "downloaded_from": output_file,
            }

        # Submit
        submission = await state.kaggle_client.submit(
            competition=state.current_competition,
            file_path=local_file,
            message=full_message,
        )

        # Record submission
        await state.submission_guard.record_submission(
            competition=state.current_competition,
            file_path=str(local_file),
            cv_score=None,
            approved=True,
            submitted=True,
            message=full_message,
        )

        return {
            "status": "success",
            "submission_id": submission.id,
            "message": full_message,
            "notebook_slug": notebook_slug,
            "downloaded_from": output_file,
        }

    except Exception as e:
        return {"error": f"Failed to submit notebook output: {str(e)}"}


def get_competition_submissions_dir(competition: str) -> Path:
    """Get the submissions directory for a competition."""
    submissions_dir = state.data_dir / competition / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    return submissions_dir


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
