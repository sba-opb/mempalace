from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path

import pytest

from mempalace.palace import (
    _lock_mine_lock_file,
    _mine_lock_path,
    _open_mine_lock_file,
    _unlock_mine_lock_file,
    mine_lock,
)


def _set_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))


def _wait_for_path(path: Path, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(0.01)
    return path.exists()


def _stale_waiter_target(
    lock_path: str,
    source_file: str,
    opened_flag: str,
    entered_flag: str,
    release_flag: str,
    result_q,
) -> None:
    try:
        from mempalace.palace import (
            _acquire_open_mine_lock_file as acquire_open,
            _open_mine_lock_file as open_lock,
            _unlock_mine_lock_file as unlock_file,
            mine_lock as public_mine_lock,
        )

        lf = open_lock(lock_path, create=True)
        Path(opened_flag).touch()
        current = acquire_open(lf, lock_path)
        result_q.put(("first-acquire-current", current))
        if current:
            Path(entered_flag).touch()
            _wait_for_path(Path(release_flag))
            unlock_file(lf)
            lf.close()
            result_q.put(("done", True))
            return

        lf.close()
        with public_mine_lock(source_file):
            Path(entered_flag).touch()
            _wait_for_path(Path(release_flag))
        result_q.put(("done", True))
    except BaseException as exc:  # pragma: no cover - surfaced through queue
        result_q.put(("error", repr(exc)))


def test_mine_lock_removes_uncontended_lock_file(tmp_path, monkeypatch):
    _set_home(monkeypatch, tmp_path)
    source_file = str(tmp_path / "source.txt")
    lock_path = Path(_mine_lock_path(source_file))

    with mine_lock(source_file):
        assert lock_path.exists()

    assert not lock_path.exists()

    with mine_lock(source_file):
        assert lock_path.exists()

    assert not lock_path.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX inode replacement regression")
def test_mine_lock_retries_when_waiter_wakes_on_unlinked_inode(tmp_path, monkeypatch):
    """A waiter on an unlinked lock inode must not enter the critical section.

    This models the race from issue #1800: process A removes the path after
    release while process B was already waiting on the old inode and process C
    has locked a replacement path. B must reject the stale inode and retry.
    """
    _set_home(monkeypatch, tmp_path)
    source_file = str(tmp_path / "source.txt")
    lock_path = Path(_mine_lock_path(source_file))

    old_lf = _open_mine_lock_file(str(lock_path), create=True)
    replacement_lf = None
    child = None
    try:
        assert _lock_mine_lock_file(old_lf, blocking=False)

        opened_flag = tmp_path / "opened"
        entered_flag = tmp_path / "entered"
        release_flag = tmp_path / "release"
        ctx = multiprocessing.get_context("spawn")
        result_q = ctx.Queue()
        child = ctx.Process(
            target=_stale_waiter_target,
            args=(
                str(lock_path),
                source_file,
                str(opened_flag),
                str(entered_flag),
                str(release_flag),
                result_q,
            ),
        )
        child.start()
        assert _wait_for_path(opened_flag), "waiter did not open the original lock file"

        os.remove(lock_path)
        replacement_lf = _open_mine_lock_file(str(lock_path), create=True)
        assert _lock_mine_lock_file(replacement_lf, blocking=False)

        _unlock_mine_lock_file(old_lf)
        old_lf.close()
        old_lf = None

        assert result_q.get(timeout=10) == ("first-acquire-current", False)
        time.sleep(0.2)
        assert not entered_flag.exists(), "waiter entered while replacement lock was held"

        _unlock_mine_lock_file(replacement_lf)
        replacement_lf.close()
        replacement_lf = None

        assert _wait_for_path(entered_flag), "waiter did not retry on the replacement path"
        release_flag.touch()
        assert result_q.get(timeout=10) == ("done", True)
        child.join(timeout=10)
        assert child.exitcode == 0
        assert not lock_path.exists()
    finally:
        if child is not None and child.is_alive():
            child.terminate()
            child.join(timeout=5)
        if replacement_lf is not None:
            try:
                _unlock_mine_lock_file(replacement_lf)
            except Exception:
                pass
            replacement_lf.close()
        if old_lf is not None:
            try:
                _unlock_mine_lock_file(old_lf)
            except Exception:
                pass
            old_lf.close()
