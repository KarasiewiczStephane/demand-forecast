"""Tests for the Kaggle dataset downloader module."""

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestKaggleDownloader:
    """Tests for KaggleDownloader class."""

    @patch("src.data.downloader._get_kaggle_api")
    def test_init(self, mock_get_api: MagicMock) -> None:
        """Downloader should authenticate on initialization."""
        from src.data.downloader import KaggleDownloader

        downloader = KaggleDownloader("test-competition", "/tmp/test")
        assert downloader.competition == "test-competition"
        mock_get_api.assert_called_once()

    @patch("src.data.downloader._get_kaggle_api")
    def test_download_creates_directory(
        self, mock_get_api: MagicMock, tmp_path: Path
    ) -> None:
        """Download should create the target directory."""
        from src.data.downloader import KaggleDownloader

        target = tmp_path / "nested" / "dir"
        downloader = KaggleDownloader("test-comp", str(target))
        downloader.download()
        assert target.exists()

    @patch("src.data.downloader._get_kaggle_api")
    def test_download_calls_api(self, mock_get_api: MagicMock, tmp_path: Path) -> None:
        """Download should call the Kaggle API with correct parameters."""
        from src.data.downloader import KaggleDownloader

        downloader = KaggleDownloader("my-comp", str(tmp_path))
        downloader.download()
        downloader.api.competition_download_files.assert_called_once_with(
            "my-comp", path=str(tmp_path), quiet=False
        )

    @patch("src.data.downloader._get_kaggle_api")
    def test_download_extracts_zip(
        self, mock_get_api: MagicMock, tmp_path: Path
    ) -> None:
        """Download should extract and remove the zip file."""
        from src.data.downloader import KaggleDownloader

        zip_path = tmp_path / "my-comp.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.csv", "a,b\n1,2\n")

        downloader = KaggleDownloader("my-comp", str(tmp_path))
        result = downloader.download()

        assert result == tmp_path
        assert (tmp_path / "data.csv").exists()
        assert not zip_path.exists()

    @patch("src.data.downloader._get_kaggle_api")
    def test_download_returns_path(
        self, mock_get_api: MagicMock, tmp_path: Path
    ) -> None:
        """Download should return the download directory path."""
        from src.data.downloader import KaggleDownloader

        downloader = KaggleDownloader("comp", str(tmp_path))
        result = downloader.download()
        assert result == tmp_path
