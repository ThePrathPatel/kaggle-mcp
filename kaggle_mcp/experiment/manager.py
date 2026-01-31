"""Experiment tracking and management."""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import aiosqlite


@dataclass
class ExperimentRun:
    """A single experiment run."""
    run_id: str
    experiment_id: str
    model_type: str
    hyperparameters: dict[str, Any]
    features_used: list[str]
    cv_score: Optional[float] = None
    cv_std: Optional[float] = None
    lb_score: Optional[float] = None
    training_time_seconds: float = 0.0
    status: str = "pending"
    artifact_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        d["completed_at"] = self.completed_at.isoformat() if self.completed_at else None
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentRun":
        """Create from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("completed_at"):
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])
        return cls(**data)


@dataclass
class Experiment:
    """An experiment tracking multiple runs for a competition."""
    experiment_id: str
    competition: str
    description: str
    data_version: str
    created_at: datetime = field(default_factory=datetime.now)
    best_cv_score: Optional[float] = None
    best_lb_score: Optional[float] = None
    best_run_id: Optional[str] = None
    status: str = "active"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Experiment":
        """Create from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class ExperimentManager:
    """Manages experiments and runs with SQLite backend."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("./kaggle_data/experiments.db")
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure database is initialized."""
        if self._initialized:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    competition TEXT NOT NULL,
                    description TEXT,
                    data_version TEXT,
                    created_at TEXT NOT NULL,
                    best_cv_score REAL,
                    best_lb_score REAL,
                    best_run_id TEXT,
                    status TEXT DEFAULT 'active'
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    hyperparameters TEXT,
                    features_used TEXT,
                    cv_score REAL,
                    cv_std REAL,
                    lb_score REAL,
                    training_time_seconds REAL,
                    status TEXT DEFAULT 'pending',
                    artifact_path TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    notes TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_experiment
                ON runs(experiment_id)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_competition
                ON experiments(competition)
            """)

            await db.commit()

        self._initialized = True

    async def create_experiment(
        self,
        competition: str,
        description: str = "",
        data_version: str = "v1",
    ) -> Experiment:
        """Create a new experiment."""
        await self._ensure_initialized()

        experiment = Experiment(
            experiment_id=str(uuid.uuid4()),
            competition=competition,
            description=description,
            data_version=data_version,
        )

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO experiments
                (experiment_id, competition, description, data_version, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.experiment_id,
                    experiment.competition,
                    experiment.description,
                    experiment.data_version,
                    experiment.created_at.isoformat(),
                    experiment.status,
                ),
            )
            await db.commit()

        return experiment

    async def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return Experiment.from_dict(dict(row))
        return None

    async def get_experiments_for_competition(
        self,
        competition: str,
    ) -> list[Experiment]:
        """Get all experiments for a competition."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM experiments WHERE competition = ? ORDER BY created_at DESC",
                (competition,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [Experiment.from_dict(dict(row)) for row in rows]

    async def create_run(
        self,
        experiment_id: str,
        model_type: str,
        hyperparameters: dict[str, Any],
        features_used: list[str],
    ) -> ExperimentRun:
        """Create a new run."""
        await self._ensure_initialized()

        run = ExperimentRun(
            run_id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            model_type=model_type,
            hyperparameters=hyperparameters,
            features_used=features_used,
        )

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO runs
                (run_id, experiment_id, model_type, hyperparameters, features_used,
                 status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.experiment_id,
                    run.model_type,
                    json.dumps(hyperparameters),
                    json.dumps(features_used),
                    run.status,
                    run.created_at.isoformat(),
                ),
            )
            await db.commit()

        return run

    async def update_run(
        self,
        run_id: str,
        cv_score: Optional[float] = None,
        cv_std: Optional[float] = None,
        lb_score: Optional[float] = None,
        training_time_seconds: Optional[float] = None,
        status: Optional[str] = None,
        artifact_path: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Optional[ExperimentRun]:
        """Update a run with results."""
        await self._ensure_initialized()

        updates = []
        values = []

        if cv_score is not None:
            updates.append("cv_score = ?")
            values.append(cv_score)
        if cv_std is not None:
            updates.append("cv_std = ?")
            values.append(cv_std)
        if lb_score is not None:
            updates.append("lb_score = ?")
            values.append(lb_score)
        if training_time_seconds is not None:
            updates.append("training_time_seconds = ?")
            values.append(training_time_seconds)
        if status is not None:
            updates.append("status = ?")
            values.append(status)
            if status == "completed":
                updates.append("completed_at = ?")
                values.append(datetime.now().isoformat())
        if artifact_path is not None:
            updates.append("artifact_path = ?")
            values.append(artifact_path)
        if notes is not None:
            updates.append("notes = ?")
            values.append(notes)

        if not updates:
            return await self.get_run(run_id)

        values.append(run_id)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"UPDATE runs SET {', '.join(updates)} WHERE run_id = ?",
                values,
            )
            await db.commit()

        # Update experiment best scores if needed
        run = await self.get_run(run_id)
        if run and cv_score is not None:
            await self._update_experiment_best(run.experiment_id, run)

        return run

    async def _update_experiment_best(self, experiment_id: str, run: ExperimentRun):
        """Update experiment with best scores."""
        experiment = await self.get_experiment(experiment_id)
        if not experiment:
            return

        update_needed = False

        if run.cv_score is not None:
            if experiment.best_cv_score is None or run.cv_score > experiment.best_cv_score:
                experiment.best_cv_score = run.cv_score
                experiment.best_run_id = run.run_id
                update_needed = True

        if run.lb_score is not None:
            if experiment.best_lb_score is None or run.lb_score > experiment.best_lb_score:
                experiment.best_lb_score = run.lb_score
                update_needed = True

        if update_needed:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    UPDATE experiments
                    SET best_cv_score = ?, best_lb_score = ?, best_run_id = ?
                    WHERE experiment_id = ?
                    """,
                    (
                        experiment.best_cv_score,
                        experiment.best_lb_score,
                        experiment.best_run_id,
                        experiment_id,
                    ),
                )
                await db.commit()

    async def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get a run by ID."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    data = dict(row)
                    data["hyperparameters"] = json.loads(data["hyperparameters"] or "{}")
                    data["features_used"] = json.loads(data["features_used"] or "[]")
                    return ExperimentRun.from_dict(data)
        return None

    async def get_runs_for_experiment(
        self,
        experiment_id: str,
        limit: int = 100,
    ) -> list[ExperimentRun]:
        """Get all runs for an experiment."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT * FROM runs
                WHERE experiment_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (experiment_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()
                runs = []
                for row in rows:
                    data = dict(row)
                    data["hyperparameters"] = json.loads(data["hyperparameters"] or "{}")
                    data["features_used"] = json.loads(data["features_used"] or "[]")
                    runs.append(ExperimentRun.from_dict(data))
                return runs

    async def get_best_run(self, experiment_id: str) -> Optional[ExperimentRun]:
        """Get the best run for an experiment."""
        experiment = await self.get_experiment(experiment_id)
        if experiment and experiment.best_run_id:
            return await self.get_run(experiment.best_run_id)
        return None

    async def get_experiment_summary(self, experiment_id: str) -> dict:
        """Get a summary of an experiment."""
        await self._ensure_initialized()

        experiment = await self.get_experiment(experiment_id)
        if not experiment:
            return {}

        runs = await self.get_runs_for_experiment(experiment_id)

        return {
            "experiment_id": experiment.experiment_id,
            "competition": experiment.competition,
            "description": experiment.description,
            "data_version": experiment.data_version,
            "status": experiment.status,
            "total_runs": len(runs),
            "completed_runs": len([r for r in runs if r.status == "completed"]),
            "best_cv_score": experiment.best_cv_score,
            "best_lb_score": experiment.best_lb_score,
            "best_run_id": experiment.best_run_id,
            "model_types_tried": list(set(r.model_type for r in runs)),
            "total_training_time": sum(r.training_time_seconds for r in runs),
        }
