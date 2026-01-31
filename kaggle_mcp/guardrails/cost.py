"""Cost and resource controls for ML training."""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import json
from pathlib import Path


@dataclass
class CostConfig:
    """Configuration for cost controls."""
    max_training_time_seconds: int = 3600  # 1 hour per training run
    max_total_training_time_per_day: int = 14400  # 4 hours per day
    max_memory_mb: int = 8192  # 8GB RAM limit
    max_concurrent_jobs: int = 1
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    state_file: Optional[Path] = None

    def __post_init__(self):
        if self.state_file is None:
            self.state_file = Path("./kaggle_data/.cost_state.json")


@dataclass
class TrainingJob:
    """Represents a training job."""
    job_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0
    status: str = "running"
    model_type: str = ""
    competition: str = ""


@dataclass
class CostState:
    """Tracks resource usage."""
    daily_training_time: dict[str, float] = field(default_factory=dict)
    active_jobs: dict[str, TrainingJob] = field(default_factory=dict)
    completed_jobs: list[TrainingJob] = field(default_factory=list)


class CostController:
    """Controls training costs and resources."""

    def __init__(self, config: Optional[CostConfig] = None):
        self.config = config or CostConfig()
        self._state = CostState()
        self._load_state()

    def _load_state(self):
        """Load state from file."""
        if self.config.state_file and self.config.state_file.exists():
            try:
                with open(self.config.state_file) as f:
                    data = json.load(f)
                    self._state.daily_training_time = data.get("daily_training_time", {})
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_state(self):
        """Save state to file."""
        if self.config.state_file:
            self.config.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "daily_training_time": self._state.daily_training_time,
            }
            with open(self.config.state_file, "w") as f:
                json.dump(data, f, indent=2)

    def _get_today_key(self) -> str:
        """Get key for today's date."""
        return datetime.now().date().isoformat()

    def _reset_daily_if_needed(self):
        """Reset daily counters if it's a new day."""
        today = self._get_today_key()
        # Clean up old entries
        old_keys = [k for k in self._state.daily_training_time if k != today]
        for k in old_keys:
            del self._state.daily_training_time[k]

        if today not in self._state.daily_training_time:
            self._state.daily_training_time[today] = 0.0

    def get_daily_training_time(self) -> float:
        """Get total training time used today in seconds."""
        self._reset_daily_if_needed()
        return self._state.daily_training_time.get(self._get_today_key(), 0.0)

    def get_remaining_training_time(self) -> float:
        """Get remaining training time for today in seconds."""
        used = self.get_daily_training_time()
        return max(0, self.config.max_total_training_time_per_day - used)

    def can_start_training(self, estimated_duration: Optional[float] = None) -> tuple[bool, str]:
        """Check if a new training job can be started."""
        # Check concurrent jobs
        if len(self._state.active_jobs) >= self.config.max_concurrent_jobs:
            return False, f"Maximum concurrent jobs ({self.config.max_concurrent_jobs}) reached."

        # Check daily limit
        remaining = self.get_remaining_training_time()
        if remaining <= 0:
            return False, "Daily training time limit reached."

        if estimated_duration and estimated_duration > remaining:
            return False, (
                f"Estimated duration ({estimated_duration:.0f}s) exceeds "
                f"remaining time ({remaining:.0f}s)."
            )

        if estimated_duration and estimated_duration > self.config.max_training_time_seconds:
            return False, (
                f"Estimated duration ({estimated_duration:.0f}s) exceeds "
                f"max per-job limit ({self.config.max_training_time_seconds}s)."
            )

        return True, "Training allowed."

    def start_job(
        self,
        job_id: str,
        model_type: str = "",
        competition: str = "",
    ) -> TrainingJob:
        """Start tracking a training job."""
        job = TrainingJob(
            job_id=job_id,
            start_time=datetime.now(),
            model_type=model_type,
            competition=competition,
        )
        self._state.active_jobs[job_id] = job
        return job

    def check_job_timeout(self, job_id: str) -> tuple[bool, str]:
        """Check if a job has exceeded its time limit."""
        if job_id not in self._state.active_jobs:
            return False, "Job not found."

        job = self._state.active_jobs[job_id]
        elapsed = (datetime.now() - job.start_time).total_seconds()

        if elapsed > self.config.max_training_time_seconds:
            return True, f"Job exceeded time limit ({elapsed:.0f}s > {self.config.max_training_time_seconds}s)."

        return False, f"Job within time limit ({elapsed:.0f}s / {self.config.max_training_time_seconds}s)."

    def end_job(self, job_id: str, status: str = "completed") -> Optional[TrainingJob]:
        """End a training job and record its duration."""
        if job_id not in self._state.active_jobs:
            return None

        job = self._state.active_jobs.pop(job_id)
        job.end_time = datetime.now()
        job.duration_seconds = (job.end_time - job.start_time).total_seconds()
        job.status = status

        # Update daily training time
        self._reset_daily_if_needed()
        today = self._get_today_key()
        self._state.daily_training_time[today] = (
            self._state.daily_training_time.get(today, 0.0) + job.duration_seconds
        )

        self._state.completed_jobs.append(job)
        self._save_state()

        return job

    def get_status(self) -> dict:
        """Get current resource usage status."""
        return {
            "daily_training_time_used": self.get_daily_training_time(),
            "daily_training_time_remaining": self.get_remaining_training_time(),
            "daily_limit": self.config.max_total_training_time_per_day,
            "active_jobs": len(self._state.active_jobs),
            "max_concurrent_jobs": self.config.max_concurrent_jobs,
            "max_job_duration": self.config.max_training_time_seconds,
            "early_stopping_enabled": self.config.enable_early_stopping,
        }


class TrainingTimer:
    """Context manager for timing training operations."""

    def __init__(self, controller: CostController, job_id: str, **kwargs):
        self.controller = controller
        self.job_id = job_id
        self.kwargs = kwargs
        self._job: Optional[TrainingJob] = None

    def __enter__(self) -> "TrainingTimer":
        self._job = self.controller.start_job(self.job_id, **self.kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "failed" if exc_type else "completed"
        self.controller.end_job(self.job_id, status=status)

    def check_timeout(self) -> bool:
        """Check if the job should be terminated."""
        timed_out, _ = self.controller.check_job_timeout(self.job_id)
        return timed_out

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self._job:
            return (datetime.now() - self._job.start_time).total_seconds()
        return 0.0
