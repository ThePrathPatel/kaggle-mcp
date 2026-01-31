"""Kaggle API client wrapper with safety controls."""

import os
import json
import asyncio
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .models import Competition, Dataset, Submission, LeaderboardEntry


@dataclass
class KaggleConfig:
    """Configuration for Kaggle API client."""
    username: Optional[str] = None
    key: Optional[str] = None
    config_dir: Optional[Path] = None
    data_dir: Path = Path("./kaggle_data")

    def __post_init__(self):
        if self.config_dir is None:
            self.config_dir = Path.home() / ".kaggle"


class KaggleClient:
    """Async wrapper around Kaggle API with safety controls."""

    def __init__(self, config: Optional[KaggleConfig] = None):
        self.config = config or KaggleConfig()
        self._api = None
        self._authenticated = False

    async def authenticate(self) -> bool:
        """Authenticate with Kaggle API."""
        try:
            # Run in executor since kaggle library is synchronous
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._setup_api)
            self._authenticated = True
            return True
        except Exception as e:
            raise KaggleAuthError(f"Failed to authenticate with Kaggle: {e}")

    def _setup_api(self):
        """Set up the Kaggle API (synchronous)."""
        # Set credentials if provided
        if self.config.username and self.config.key:
            os.environ["KAGGLE_USERNAME"] = self.config.username
            os.environ["KAGGLE_KEY"] = self.config.key

        from kaggle.api.kaggle_api_extended import KaggleApi
        self._api = KaggleApi()
        self._api.authenticate()

    def _ensure_authenticated(self):
        """Ensure API is authenticated."""
        if not self._authenticated or self._api is None:
            raise KaggleAuthError("Not authenticated. Call authenticate() first.")

    async def list_competitions(
        self,
        category: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: str = "latestDeadline",
        page: int = 1,
    ) -> list[Competition]:
        """List available competitions."""
        self._ensure_authenticated()

        def _list():
            competitions = self._api.competitions_list(
                category=category,
                sort_by=sort_by,
                search=search,
                page=page,
            )
            return [Competition.from_api_response(c.__dict__) for c in competitions]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _list)

    async def get_competition(self, competition: str) -> Competition:
        """Get details for a specific competition."""
        self._ensure_authenticated()

        def _get():
            # competitions_list with search for exact match
            competitions = self._api.competitions_list(search=competition)
            for c in competitions:
                if c.ref == competition:
                    return Competition.from_api_response(c.__dict__)
            raise KaggleNotFoundError(f"Competition not found: {competition}")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get)

    async def download_competition_data(
        self,
        competition: str,
        path: Optional[Path] = None,
        force: bool = False,
    ) -> Path:
        """Download competition data files."""
        self._ensure_authenticated()

        if path is None:
            path = self.config.data_dir / competition

        def _download():
            path.mkdir(parents=True, exist_ok=True)
            self._api.competition_download_files(
                competition=competition,
                path=str(path),
                force=force,
                quiet=False,
            )
            return path

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _download)

    async def list_competition_files(self, competition: str) -> list[dict]:
        """List files available for a competition."""
        self._ensure_authenticated()

        def _list():
            files = self._api.competition_list_files(competition)
            return [
                {
                    "name": f.name,
                    "size": f.size,
                    "creation_date": str(f.creationDate) if hasattr(f, 'creationDate') else None,
                }
                for f in files
            ]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _list)

    async def submit(
        self,
        competition: str,
        file_path: Path,
        message: str,
    ) -> Submission:
        """Submit a prediction file to a competition."""
        self._ensure_authenticated()

        if not file_path.exists():
            raise KaggleError(f"Submission file not found: {file_path}")

        def _submit():
            result = self._api.competition_submit(
                file_name=str(file_path),
                message=message,
                competition=competition,
            )
            # The API returns a submission object or confirmation
            return result

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _submit)

        # Fetch the latest submission to get full details
        submissions = await self.list_submissions(competition, limit=1)
        if submissions:
            return submissions[0]

        # Return a basic submission object if we can't fetch details
        from datetime import datetime
        return Submission(
            id=0,
            ref=competition,
            date=datetime.now(),
            description=message,
            status="pending",
            public_score=None,
            private_score=None,
        )

    async def list_submissions(
        self,
        competition: str,
        limit: int = 20,
    ) -> list[Submission]:
        """List submissions for a competition."""
        self._ensure_authenticated()

        def _list():
            submissions = self._api.competition_submissions(competition)
            return [
                Submission.from_api_response(s.__dict__)
                for s in submissions[:limit]
            ]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _list)

    async def get_leaderboard(
        self,
        competition: str,
        limit: int = 50,
    ) -> list[LeaderboardEntry]:
        """Get competition leaderboard."""
        self._ensure_authenticated()

        def _get():
            leaderboard = self._api.competition_leaderboard_view(competition)
            return [
                LeaderboardEntry.from_api_response(entry.__dict__, rank=i + 1)
                for i, entry in enumerate(leaderboard[:limit])
            ]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get)

    async def get_my_position(self, competition: str) -> Optional[LeaderboardEntry]:
        """Get current user's position on leaderboard."""
        self._ensure_authenticated()

        def _get():
            try:
                # Get user's submissions to find their best score
                submissions = self._api.competition_submissions(competition)
                if not submissions:
                    return None

                # Get leaderboard to find position
                leaderboard = self._api.competition_leaderboard_view(competition)

                # Find current user in leaderboard
                for i, entry in enumerate(leaderboard):
                    # Match by checking if it's the authenticated user
                    # This is a simplified check
                    if hasattr(entry, 'teamName'):
                        return LeaderboardEntry.from_api_response(entry.__dict__, rank=i + 1)

                return None
            except Exception:
                return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get)

    async def list_datasets(
        self,
        search: Optional[str] = None,
        sort_by: str = "hottest",
        page: int = 1,
    ) -> list[Dataset]:
        """List available datasets."""
        self._ensure_authenticated()

        def _list():
            datasets = self._api.dataset_list(
                search=search,
                sort_by=sort_by,
                page=page,
            )
            return [Dataset.from_api_response(d.__dict__) for d in datasets]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _list)

    async def download_dataset(
        self,
        dataset: str,
        path: Optional[Path] = None,
        force: bool = False,
    ) -> Path:
        """Download a dataset."""
        self._ensure_authenticated()

        if path is None:
            path = self.config.data_dir / "datasets" / dataset.replace("/", "_")

        def _download():
            path.mkdir(parents=True, exist_ok=True)
            self._api.dataset_download_files(
                dataset=dataset,
                path=str(path),
                force=force,
                unzip=True,
            )
            return path

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _download)


class KaggleError(Exception):
    """Base exception for Kaggle operations."""
    pass


class KaggleAuthError(KaggleError):
    """Authentication error."""
    pass


class KaggleNotFoundError(KaggleError):
    """Resource not found error."""
    pass


class KaggleRateLimitError(KaggleError):
    """Rate limit exceeded error."""
    pass
