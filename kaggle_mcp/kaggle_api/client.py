"""Kaggle API client wrapper with safety controls."""

import os
import json
import asyncio
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

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
    """Async wrapper around Kaggle SDK with safety controls."""

    def __init__(self, config: Optional[KaggleConfig] = None):
        self.config = config or KaggleConfig()
        self._sdk_client = None
        self._authenticated = False
        self._api_token = None
        self._username = None
        self._key = None

    async def authenticate(self) -> bool:
        """Authenticate with Kaggle API."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._setup_credentials)
            self._authenticated = True
            return True
        except Exception as e:
            raise KaggleAuthError(f"Failed to authenticate with Kaggle: {e}")

    def _setup_credentials(self):
        """Set up credentials (synchronous)."""
        # Set config directory
        if self.config.config_dir:
            os.environ["KAGGLE_CONFIG_DIR"] = str(self.config.config_dir)

        # Priority 1: Explicit config
        api_token = None
        username = self.config.username
        key = self.config.key

        # Priority 2: Environment variables
        if not api_token:
            api_token = os.environ.get("KAGGLE_API_TOKEN")
        if not username:
            username = os.environ.get("KAGGLE_USERNAME")
        if not key:
            key = os.environ.get("KAGGLE_KEY")

        # Priority 3: kaggle.json file
        kaggle_json = self.config.config_dir / "kaggle.json"
        if kaggle_json.exists():
            with open(kaggle_json) as f:
                creds = json.load(f)
                if not username:
                    username = creds.get("username")
                if not key and not api_token:
                    key = creds.get("key")

        # Check if key is actually an access token (KGAT_ prefix)
        if key and key.startswith("KGAT_"):
            api_token = key
            key = None

        if not api_token and (not username or not key):
            raise KaggleAuthError(
                f"No Kaggle credentials found. Either:\n"
                f"  1. Create {kaggle_json} with your API key, or\n"
                f"  2. Set KAGGLE_API_TOKEN environment variable, or\n"
                f"  3. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables\n"
                f"Get your API key from: https://www.kaggle.com/settings/account"
            )

        self._api_token = api_token
        self._username = username
        self._key = key

        # Initialize the SDK client
        from kagglesdk import KaggleClient as SdkClient
        self._sdk_client = SdkClient(
            api_token=api_token,
            username=username if not api_token else None,
            password=key if not api_token else None,
        )

    def _ensure_authenticated(self):
        """Ensure API is authenticated."""
        if not self._authenticated or self._sdk_client is None:
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
            from kagglesdk.competitions.types.competition_api_service import ApiListCompetitionsRequest
            from kagglesdk.competitions.types.competition_enums import HostSegment

            with self._sdk_client as client:
                request = ApiListCompetitionsRequest()
                if search:
                    request.search = search
                if category:
                    category_map = {
                        'featured': HostSegment.HOST_SEGMENT_FEATURED,
                        'research': HostSegment.HOST_SEGMENT_RESEARCH,
                        'recruitment': HostSegment.HOST_SEGMENT_RECRUITMENT,
                        'gettingStarted': HostSegment.HOST_SEGMENT_GETTING_STARTED,
                        'masters': HostSegment.HOST_SEGMENT_MASTERS,
                        'playground': HostSegment.HOST_SEGMENT_PLAYGROUND,
                    }
                    if category in category_map:
                        request.category = category_map[category]
                request.page = page

                response = client.competitions.competition_api_client.list_competitions(request)
                return [
                    Competition(
                        id=str(c.id) if c.id else "",
                        ref=c.ref or "",
                        title=c.title or "",
                        description=c.description or "",
                        organization=c.organization_name or "",
                        category=str(c.category) if c.category else "",
                        reward=c.reward or "",
                        deadline=c.deadline,
                        kernel_count=c.kernel_count or 0,
                        team_count=c.team_count or 0,
                        user_has_entered=c.user_has_entered or False,
                        evaluation_metric=c.evaluation_metric or "",
                    )
                    for c in (response.competitions or [])
                ]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _list)

    async def get_competition(self, competition: str) -> Competition:
        """Get details for a specific competition."""
        competitions = await self.list_competitions(search=competition)
        for c in competitions:
            if c.ref == competition:
                return c
        raise KaggleNotFoundError(f"Competition not found: {competition}")

    async def download_competition_data(
        self,
        competition: str,
        path: Optional[Path] = None,
        force: bool = False,
    ) -> Path:
        """Download competition data files using SDK directly."""
        self._ensure_authenticated()

        if path is None:
            path = self.config.data_dir / competition

        def _download():
            from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest

            path.mkdir(parents=True, exist_ok=True)

            # Check if already downloaded (unless force=True)
            if not force and any(path.iterdir()):
                return path

            with self._sdk_client as client:
                request = ApiDownloadDataFilesRequest()
                request.competition_name = competition
                response = client.competitions.competition_api_client.download_data_files(request)

                if response.status_code != 200:
                    raise KaggleError(f"Download failed: {response.status_code} {response.reason}")

                # Extract the zip file
                with zipfile.ZipFile(BytesIO(response.content)) as zf:
                    zf.extractall(path)

            return path

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _download)

    async def list_competition_files(self, competition: str) -> list[dict]:
        """List files available for a competition."""
        self._ensure_authenticated()

        def _list():
            from kagglesdk.competitions.types.competition_api_service import ApiListDataFilesRequest

            with self._sdk_client as client:
                request = ApiListDataFilesRequest()
                request.competition_name = competition
                response = client.competitions.competition_api_client.list_data_files(request)
                return [
                    {
                        "name": f.name,
                        "size": f.total_bytes,
                        "creation_date": str(f.creation_date) if f.creation_date else None,
                    }
                    for f in (response.files or [])
                ]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _list)

    async def submit(
        self,
        competition: str,
        file_path: Path,
        message: str,
    ) -> Submission:
        """Submit a prediction file to a competition using SDK."""
        self._ensure_authenticated()

        if not file_path.exists():
            raise KaggleError(f"Submission file not found: {file_path}")

        def _submit():
            from kagglesdk.competitions.types.competition_api_service import (
                ApiStartSubmissionUploadRequest,
                ApiCreateSubmissionRequest,
            )
            import requests

            file_size = file_path.stat().st_size
            file_name = file_path.name

            with self._sdk_client as client:
                # Step 1: Start the upload and get a signed URL
                start_request = ApiStartSubmissionUploadRequest()
                start_request.competition_name = competition
                start_request.file_name = file_name
                start_request.content_length = file_size
                start_request.last_modified_epoch_seconds = int(file_path.stat().st_mtime)

                start_response = client.competitions.competition_api_client.start_submission_upload(start_request)

                # Step 2: Upload the file to the signed URL
                create_url = start_response.create_url
                if create_url:
                    with open(file_path, 'rb') as f:
                        upload_response = requests.put(
                            create_url,
                            data=f.read(),
                            headers={'Content-Type': 'application/octet-stream'}
                        )
                        if upload_response.status_code not in [200, 201]:
                            raise KaggleError(f"Upload failed: {upload_response.status_code}")

                # Step 3: Create the submission
                create_request = ApiCreateSubmissionRequest()
                create_request.competition_name = competition
                create_request.blob_file_tokens = start_response.token
                create_request.submission_description = message

                client.competitions.competition_api_client.create_submission(create_request)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _submit)

        # Fetch the latest submission to get full details
        submissions = await self.list_submissions(competition, limit=1)
        if submissions:
            return submissions[0]

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
            from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest

            with self._sdk_client as client:
                request = ApiListSubmissionsRequest()
                request.competition_name = competition
                response = client.competitions.competition_api_client.list_submissions(request)
                return [
                    Submission(
                        id=s.ref or 0,
                        ref=competition,
                        date=s.date,
                        description=s.description or "",
                        status=s.status.name if s.status else "unknown",
                        public_score=s.public_score,
                        private_score=s.private_score,
                    )
                    for s in (response.submissions or [])[:limit]
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
            from kagglesdk.competitions.types.competition_api_service import ApiGetLeaderboardRequest

            with self._sdk_client as client:
                request = ApiGetLeaderboardRequest()
                request.competition_name = competition
                response = client.competitions.competition_api_client.get_leaderboard(request)
                return [
                    LeaderboardEntry(
                        rank=i + 1,
                        team_id=entry.team_id or 0,
                        team_name=entry.team_name or "",
                        submission_date=entry.submission_date or datetime.now(),
                        score=float(entry.score) if entry.score else 0.0,
                    )
                    for i, entry in enumerate((response.submissions or [])[:limit])
                ]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get)

    async def get_my_position(self, competition: str) -> Optional[LeaderboardEntry]:
        """Get current user's position on leaderboard."""
        try:
            submissions = await self.list_submissions(competition, limit=1)
            if not submissions:
                return None

            leaderboard = await self.get_leaderboard(competition, limit=1000)
            return None
        except Exception:
            return None

    async def list_datasets(
        self,
        search: Optional[str] = None,
        sort_by: str = "hottest",
        page: int = 1,
    ) -> list[Dataset]:
        """List available datasets."""
        self._ensure_authenticated()

        def _list():
            from kagglesdk.datasets.types.dataset_api_service import ApiListDatasetsRequest

            with self._sdk_client as client:
                request = ApiListDatasetsRequest()
                if search:
                    request.search = search
                request.page = page

                response = client.datasets.dataset_api_client.list_datasets(request)
                return [
                    Dataset(
                        ref=d.ref or "",
                        title=d.title or "",
                        size=d.total_bytes or 0,
                        last_updated=d.last_updated,
                        download_count=d.download_count or 0,
                        vote_count=d.vote_count or 0,
                        usability_rating=d.usability_rating or 0.0,
                    )
                    for d in (response.datasets or [])
                ]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _list)

    async def download_dataset(
        self,
        dataset: str,
        path: Optional[Path] = None,
        force: bool = False,
    ) -> Path:
        """Download a dataset using SDK directly."""
        self._ensure_authenticated()

        if path is None:
            path = self.config.data_dir / "datasets" / dataset.replace("/", "_")

        def _download():
            from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest

            path.mkdir(parents=True, exist_ok=True)

            if not force and any(path.iterdir()):
                return path

            with self._sdk_client as client:
                request = ApiDownloadDatasetRequest()
                # Dataset ref format is "owner/dataset-name"
                parts = dataset.split("/")
                if len(parts) == 2:
                    request.owner_slug = parts[0]
                    request.dataset_slug = parts[1]
                else:
                    request.dataset_slug = dataset

                response = client.datasets.dataset_api_client.download_dataset(request)

                if response.status_code != 200:
                    raise KaggleError(f"Download failed: {response.status_code}")

                # Extract the zip file
                with zipfile.ZipFile(BytesIO(response.content)) as zf:
                    zf.extractall(path)

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
