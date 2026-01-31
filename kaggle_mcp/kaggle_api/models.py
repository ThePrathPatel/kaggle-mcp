"""Data models for Kaggle API responses."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class CompetitionCategory(str, Enum):
    """Competition categories on Kaggle."""
    FEATURED = "featured"
    RESEARCH = "research"
    RECRUITMENT = "recruitment"
    GETTING_STARTED = "gettingStarted"
    MASTERS = "masters"
    PLAYGROUND = "playground"


class EvaluationMetric(str, Enum):
    """Common evaluation metrics."""
    ACCURACY = "accuracy"
    AUC = "auc"
    LOG_LOSS = "logLoss"
    RMSE = "rmse"
    MAE = "mae"
    MAP = "map"
    F1 = "f1"
    CUSTOM = "custom"


@dataclass
class Competition:
    """Represents a Kaggle competition."""
    id: str
    ref: str
    title: str
    description: str
    organization: str
    category: str
    reward: Optional[str]
    deadline: Optional[datetime]
    kernel_count: int
    team_count: int
    user_has_entered: bool
    evaluation_metric: str
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_api_response(cls, data: dict) -> "Competition":
        """Create Competition from Kaggle API response."""
        deadline = None
        if data.get("deadline"):
            try:
                deadline = datetime.fromisoformat(data["deadline"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return cls(
            id=str(data.get("id", "")),
            ref=data.get("ref", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            organization=data.get("organizationName", ""),
            category=data.get("category", ""),
            reward=data.get("reward"),
            deadline=deadline,
            kernel_count=data.get("kernelCount", 0),
            team_count=data.get("teamCount", 0),
            user_has_entered=data.get("userHasEntered", False),
            evaluation_metric=data.get("evaluationMetric", ""),
            tags=data.get("tags", []),
        )


@dataclass
class Dataset:
    """Represents a Kaggle dataset."""
    ref: str
    title: str
    size: int
    last_updated: Optional[datetime]
    download_count: int
    vote_count: int
    usability_rating: float

    @classmethod
    def from_api_response(cls, data: dict) -> "Dataset":
        """Create Dataset from Kaggle API response."""
        last_updated = None
        if data.get("lastUpdated"):
            try:
                last_updated = datetime.fromisoformat(
                    data["lastUpdated"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        return cls(
            ref=data.get("ref", ""),
            title=data.get("title", ""),
            size=data.get("totalBytes", 0),
            last_updated=last_updated,
            download_count=data.get("downloadCount", 0),
            vote_count=data.get("voteCount", 0),
            usability_rating=data.get("usabilityRating", 0.0),
        )


@dataclass
class Submission:
    """Represents a Kaggle submission."""
    id: int
    ref: str
    date: datetime
    description: str
    status: str
    public_score: Optional[float]
    private_score: Optional[float]

    @classmethod
    def from_api_response(cls, data: dict) -> "Submission":
        """Create Submission from Kaggle API response."""
        date = datetime.now()
        if data.get("date"):
            try:
                date = datetime.fromisoformat(data["date"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        public_score = None
        if data.get("publicScore"):
            try:
                public_score = float(data["publicScore"])
            except (ValueError, TypeError):
                pass

        private_score = None
        if data.get("privateScore"):
            try:
                private_score = float(data["privateScore"])
            except (ValueError, TypeError):
                pass

        return cls(
            id=data.get("id", 0),
            ref=data.get("ref", ""),
            date=date,
            description=data.get("description", ""),
            status=data.get("status", ""),
            public_score=public_score,
            private_score=private_score,
        )


@dataclass
class LeaderboardEntry:
    """Represents a leaderboard entry."""
    team_id: int
    team_name: str
    submission_date: datetime
    score: float
    rank: int

    @classmethod
    def from_api_response(cls, data: dict, rank: int) -> "LeaderboardEntry":
        """Create LeaderboardEntry from Kaggle API response."""
        submission_date = datetime.now()
        if data.get("submissionDate"):
            try:
                submission_date = datetime.fromisoformat(
                    data["submissionDate"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        score = 0.0
        if data.get("score"):
            try:
                score = float(data["score"])
            except (ValueError, TypeError):
                pass

        return cls(
            team_id=data.get("teamId", 0),
            team_name=data.get("teamName", ""),
            submission_date=submission_date,
            score=score,
            rank=rank,
        )
