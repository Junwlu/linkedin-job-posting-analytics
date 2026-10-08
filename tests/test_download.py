"""Downloader contract tests: tiny fixtures, no network, no real dataset."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def downloader(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
    spec = importlib.util.spec_from_file_location("download_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = b"job_id,applies,views\n1,2,3\n"
    monkeypatch.setattr(module, "EXPECTED_BYTES", len(payload))
    monkeypatch.setattr(module, "EXPECTED_SHA256", hashlib.sha256(payload).hexdigest())
    return module, payload


def fake_client(monkeypatch, payload, calls, before_return=None):
    def download(handle, *, path, output_dir):
        calls.append((handle, path))
        result = Path(output_dir) / path
        result.write_bytes(payload)
        if before_return:
            before_return()
        return str(result)
    monkeypatch.setitem(sys.modules, "kagglehub", SimpleNamespace(dataset_download=download))


def test_pinned_download_then_existing_file_is_reused(downloader, monkeypatch, tmp_path):
    module, payload = downloader
    calls = []
    fake_client(monkeypatch, payload, calls)
    target = tmp_path / "raw" / "postings.csv"
    assert "Downloaded" in module.acquire(target)
    assert target.read_bytes() == payload
    assert calls == [("arshkon/linkedin-job-postings/versions/13", "postings.csv")]
    assert "Verified" in module.acquire(target)
    assert len(calls) == 1


@pytest.mark.parametrize("content", [b"short", b"job_id,applies,views\n1,2,4\n"])
def test_invalid_existing_file_is_not_changed_or_downloaded(downloader, monkeypatch, tmp_path, content):
    module, payload = downloader
    calls = []
    fake_client(monkeypatch, payload, calls)
    target = tmp_path / "postings.csv"
    target.write_bytes(content)
    with pytest.raises(ValueError, match="mismatch"):
        module.acquire(target)
    assert target.read_bytes() == content
    assert calls == []


def test_verify_only_missing_does_not_contact_client(downloader, monkeypatch, tmp_path):
    module, payload = downloader
    calls = []
    fake_client(monkeypatch, payload, calls)
    with pytest.raises(ValueError, match="No input to verify"):
        module.acquire(tmp_path / "missing.csv", verify_only=True)
    assert calls == []


def test_download_hash_mismatch_does_not_create_output(downloader, monkeypatch, tmp_path):
    module, payload = downloader
    calls = []
    fake_client(monkeypatch, payload.replace(b"2,3", b"2,4"), calls)
    target = tmp_path / "postings.csv"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.acquire(target)
    assert not target.exists()


def test_concurrent_target_is_preserved(downloader, monkeypatch, tmp_path):
    module, payload = downloader
    target = tmp_path / "postings.csv"
    calls = []
    fake_client(monkeypatch, payload, calls, before_return=lambda: target.write_bytes(b"other process"))
    with pytest.raises(FileExistsError):
        module.acquire(target)
    assert target.read_bytes() == b"other process"


def test_copy_failure_removes_only_new_partial(downloader, monkeypatch, tmp_path):
    module, payload = downloader
    calls = []
    fake_client(monkeypatch, payload, calls)
    target = tmp_path / "postings.csv"
    sentinel = tmp_path / "unrelated.csv"
    sentinel.write_bytes(b"retain")
    def fail_copy(source, destination, length):
        destination.write(b"partial")
        raise OSError("simulated disk failure")
    monkeypatch.setattr(module.shutil, "copyfileobj", fail_copy)
    with pytest.raises(OSError, match="simulated disk failure"):
        module.acquire(target)
    assert not target.exists()
    assert sentinel.read_bytes() == b"retain"


@pytest.mark.parametrize("returned_path", ["directory", "missing"])
def test_client_returning_nonfile_does_not_create_output(downloader, monkeypatch, tmp_path, returned_path):
    module, _ = downloader
    def unexpected_result(handle, *, path, output_dir):
        return output_dir if returned_path == "directory" else str(Path(output_dir) / "absent.csv")
    monkeypatch.setitem(sys.modules, "kagglehub", SimpleNamespace(dataset_download=unexpected_result))
    target = tmp_path / "postings.csv"
    with pytest.raises(ValueError, match="not a regular file"):
        module.acquire(target)
    assert not target.exists()


def test_corrupted_copy_fails_postwrite_verification_and_cleans_new_file(downloader, monkeypatch, tmp_path):
    module, payload = downloader
    calls = []
    fake_client(monkeypatch, payload, calls)
    def corrupt_copy(source, destination, length):
        destination.write(payload.replace(b"2,3", b"2,4"))
    monkeypatch.setattr(module.shutil, "copyfileobj", corrupt_copy)
    target = tmp_path / "postings.csv"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.acquire(target)
    assert not target.exists()
