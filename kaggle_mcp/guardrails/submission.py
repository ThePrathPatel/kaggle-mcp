"""Submission safety guardrails."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable, Awaitable
import json


@dataclass
class SubmissionConfig:
    """Configuration for submission guardrails."""
    max_submissions_per_day: int = 5
    require_approval: bool = True
    dry_run_mode: bool = False
    min_cv_score_threshold: Optional[float] = None
    min_improvement_threshold: float = 0.001
    cooldown_minutes: int = 10
    state_file: Optional[Path] = None

    def __post_init__(self):
        if self.state_file is None:
            self.state_file = Path("./kaggle_data/.submission_state.json")


@dataclass
class SubmissionRecord:
    """Record of a submission attempt."""
    competition: str
    timestamp: datetime
    file_path: str
    cv_score: Optional[float]
    approved: bool
    submitted: bool
    message: str


@dataclass
class SubmissionState:
    """Tracks submission state across sessions."""
    submissions: list[SubmissionRecord] = field(default_factory=list)
    last_submission_time: dict[str, datetime] = field(default_factory=dict)
    daily_counts: dict[str, int] = field(default_factory=dict)
    best_scores: dict[str, float] = field(default_factory=dict)


class SubmissionGuard:
    """Guards against excessive or unauthorized submissions."""

    def __init__(
        self,
        config: Optional[SubmissionConfig] = None,
        approval_callback: Optional[Callable[[str, str, float], Awaitable[bool]]] = None,
    ):
        self.config = config or SubmissionConfig()
        self.approval_callback = approval_callback
        self._state = SubmissionState()
        self._load_state()

    def _load_state(self):
        """Load state from file if exists."""
        if self.config.state_file and self.config.state_file.exists():
            try:
                with open(self.config.state_file) as f:
                    data = json.load(f)
                    # Parse dates and reconstruct state
                    self._state.daily_counts = data.get("daily_counts", {})
                    self._state.best_scores = data.get("best_scores", {})
                    self._state.last_submission_time = {
                        k: datetime.fromisoformat(v)
                        for k, v in data.get("last_submission_time", {}).items()
                    }
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_state(self):
        """Persist state to file."""
        if self.config.state_file:
            self.config.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "daily_counts": self._state.daily_counts,
                "best_scores": self._state.best_scores,
                "last_submission_time": {
                    k: v.isoformat()
                    for k, v in self._state.last_submission_time.items()
                },
            }
            with open(self.config.state_file, "w") as f:
                json.dump(data, f, indent=2)

    def _reset_daily_if_needed(self, competition: str):
        """Reset daily count if it's a new day."""
        today = datetime.now().date().isoformat()
        key = f"{competition}:{today}"

        # Clean up old entries
        old_keys = [
            k for k in self._state.daily_counts
            if not k.endswith(today)
        ]
        for k in old_keys:
            del self._state.daily_counts[k]

        if key not in self._state.daily_counts:
            self._state.daily_counts[key] = 0

    def get_daily_submissions(self, competition: str) -> int:
        """Get number of submissions today for a competition."""
        self._reset_daily_if_needed(competition)
        today = datetime.now().date().isoformat()
        key = f"{competition}:{today}"
        return self._state.daily_counts.get(key, 0)

    def get_remaining_submissions(self, competition: str) -> int:
        """Get remaining submissions for today."""
        used = self.get_daily_submissions(competition)
        return max(0, self.config.max_submissions_per_day - used)

    async def check_can_submit(
        self,
        competition: str,
        cv_score: Optional[float] = None,
    ) -> tuple[bool, str]:
        """Check if submission is allowed."""
        # Check dry run mode
        if self.config.dry_run_mode:
            return False, "Dry run mode is enabled. No submissions allowed."

        # Check daily limit
        remaining = self.get_remaining_submissions(competition)
        if remaining <= 0:
            return False, f"Daily submission limit ({self.config.max_submissions_per_day}) reached."

        # Check cooldown
        last_time = self._state.last_submission_time.get(competition)
        if last_time:
            cooldown = timedelta(minutes=self.config.cooldown_minutes)
            time_since = datetime.now() - last_time
            if time_since < cooldown:
                remaining_cooldown = cooldown - time_since
                return False, f"Cooldown active. Wait {remaining_cooldown.seconds // 60} minutes."

        # Check minimum CV score threshold
        if self.config.min_cv_score_threshold is not None and cv_score is not None:
            if cv_score < self.config.min_cv_score_threshold:
                return False, f"CV score ({cv_score:.4f}) below threshold ({self.config.min_cv_score_threshold:.4f})."

        # Check improvement threshold
        if cv_score is not None and competition in self._state.best_scores:
            best = self._state.best_scores[competition]
            improvement = cv_score - best
            if improvement < self.config.min_improvement_threshold:
                return False, (
                    f"Score ({cv_score:.4f}) not improved enough over best ({best:.4f}). "
                    f"Minimum improvement: {self.config.min_improvement_threshold:.4f}"
                )

        return True, "Submission allowed."

    async def request_approval(
        self,
        competition: str,
        file_path: str,
        cv_score: Optional[float] = None,
    ) -> tuple[bool, str]:
        """Request approval for submission."""
        if not self.config.require_approval:
            return True, "Auto-approved (approval not required)."

        if self.approval_callback is None:
            return False, "Approval required but no approval callback configured."

        try:
            approved = await self.approval_callback(competition, file_path, cv_score or 0.0)
            if approved:
                return True, "Submission approved."
            else:
                return False, "Submission rejected by user."
        except Exception as e:
            return False, f"Approval request failed: {e}"

    async def record_submission(
        self,
        competition: str,
        file_path: str,
        cv_score: Optional[float] = None,
        approved: bool = True,
        submitted: bool = True,
        message: str = "",
    ):
        """Record a submission attempt."""
        self._reset_daily_if_needed(competition)

        record = SubmissionRecord(
            competition=competition,
            timestamp=datetime.now(),
            file_path=file_path,
            cv_score=cv_score,
            approved=approved,
            submitted=submitted,
            message=message,
        )
        self._state.submissions.append(record)

        if submitted:
            # Update daily count
            today = datetime.now().date().isoformat()
            key = f"{competition}:{today}"
            self._state.daily_counts[key] = self._state.daily_counts.get(key, 0) + 1

            # Update last submission time
            self._state.last_submission_time[competition] = datetime.now()

            # Update best score
            if cv_score is not None:
                current_best = self._state.best_scores.get(competition, float("-inf"))
                self._state.best_scores[competition] = max(current_best, cv_score)

        self._save_state()

    def get_submission_summary(self, competition: str) -> dict:
        """Get summary of submission activity."""
        return {
            "competition": competition,
            "daily_submissions": self.get_daily_submissions(competition),
            "remaining_submissions": self.get_remaining_submissions(competition),
            "max_per_day": self.config.max_submissions_per_day,
            "best_score": self._state.best_scores.get(competition),
            "last_submission": (
                self._state.last_submission_time.get(competition).isoformat()
                if competition in self._state.last_submission_time
                else None
            ),
            "dry_run_mode": self.config.dry_run_mode,
            "require_approval": self.config.require_approval,
        }
