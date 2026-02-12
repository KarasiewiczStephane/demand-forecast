"""Kaggle dataset downloader for the Store Sales competition.

Handles authentication, downloading, and extraction of competition
data files from the Kaggle API.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _get_kaggle_api() -> Any:
    """Lazily import and authenticate the Kaggle API client.

    Returns:
        Authenticated KaggleApi instance.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


class KaggleDownloader:
    """Downloads and extracts Kaggle competition datasets.

    Attributes:
        competition: Kaggle competition slug identifier.
        download_path: Local directory to store downloaded files.
    """

    def __init__(self, competition: str, download_path: str) -> None:
        """Initialize the downloader with competition and target path.

        Args:
            competition: Kaggle competition identifier string.
            download_path: Directory where files will be downloaded.
        """
        self.competition = competition
        self.download_path = Path(download_path)
        self.api = _get_kaggle_api()

    def download(self) -> Path:
        """Download and extract competition files.

        Returns:
            Path to the directory containing extracted files.

        Raises:
            Exception: If download or extraction fails.
        """
        self.download_path.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s dataset...", self.competition)

        self.api.competition_download_files(
            self.competition,
            path=str(self.download_path),
            quiet=False,
        )

        zip_path = self.download_path / f"{self.competition}.zip"
        if zip_path.exists():
            logger.info("Extracting %s...", zip_path.name)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(self.download_path)
            zip_path.unlink()

        logger.info("Dataset downloaded to %s", self.download_path)
        return self.download_path
