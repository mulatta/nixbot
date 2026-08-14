"""Tests for clone/worktree management using real git repositories."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import shutil
import signal
import subprocess
import time
from typing import TYPE_CHECKING

import pytest

from nixbot import gitrepo
from nixbot.gitrepo import (
    FetchCredentials,
    GitError,
    MergeConflictError,
    RepoManager,
    StaticCredentialsProvider,
    run_git,
)

from .support import git, init_upstream

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

KEY = "github/acme/widget"


@pytest.fixture
def upstream(tmp_path: Path) -> Path:
    return init_upstream(tmp_path / "upstream", {"file.txt": "hello\n"})


@pytest.fixture
def manager(tmp_path: Path) -> RepoManager:
    return RepoManager(tmp_path / "state")


async def fetch(manager: RepoManager, upstream: Path) -> None:
    await manager.fetch(KEY, str(upstream), ["+refs/heads/*:refs/heads/*"])


async def test_fetch_and_worktree(manager: RepoManager, upstream: Path) -> None:
    await fetch(manager, upstream)
    sha = git(upstream, "rev-parse", "HEAD")

    wt = await manager.checkout_for_build(KEY, "build-1", base_commit=sha)
    assert (wt.path / "file.txt").read_text() == "hello\n"
    assert await wt.rev_parse("HEAD") == sha
    await manager.remove_worktree(wt)
    assert not wt.path.exists()


async def test_fetch_updates_existing_clone(
    manager: RepoManager, upstream: Path
) -> None:
    await fetch(manager, upstream)
    (upstream / "file.txt").write_text("v2\n")
    git(upstream, "commit", "-am", "update")
    await fetch(manager, upstream)
    sha = git(upstream, "rev-parse", "HEAD")

    wt = await manager.checkout_for_build(KEY, "build-2", base_commit=sha)
    assert (wt.path / "file.txt").read_text() == "v2\n"
    await manager.remove_worktree(wt)


async def test_blobless_clone_uses_nixbot_maintenance_policy(
    manager: RepoManager,
    upstream: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    git(upstream, "config", "uploadpack.allowFilter", "true")
    commands: list[list[str]] = []
    original_run_git = gitrepo.run_git

    async def record_run_git(
        args: list[str],
        *,
        cwd: Path | None = None,
        credentials: FetchCredentials | None = None,
    ) -> str:
        commands.append(args)
        return await original_run_git(args, cwd=cwd, credentials=credentials)

    monkeypatch.setattr(gitrepo, "run_git", record_run_git)
    await manager.fetch(KEY, upstream.as_uri(), ["+refs/heads/*:refs/heads/*"])
    clone = manager.clone_path(KEY)

    assert git(clone, "config", "--get", "repack.writeBitmaps") == "false"
    assert git(clone, "config", "--get", "maintenance.auto") == "false"
    assert git(clone, "config", "--get", "gc.autoDetach") == "false"
    clone_command = next(args for args in commands if args[0] == "clone")
    for key, value in gitrepo.MANAGED_REPO_CONFIG:
        assert f"--config={key}={value}" in clone_command
    fetch_command = next(args for args in commands if args[0] == "fetch")
    assert "--no-auto-maintenance" in fetch_command

    monkeypatch.setenv("GIT_CONFIG_COUNT", str(len(gitrepo.MANAGED_REPO_CONFIG)))
    for index, (key, _value) in enumerate(gitrepo.MANAGED_REPO_CONFIG):
        monkeypatch.setenv(f"GIT_CONFIG_KEY_{index}", key)
        monkeypatch.setenv(f"GIT_CONFIG_VALUE_{index}", "true")
    for key, value in gitrepo.MANAGED_REPO_CONFIG:
        effective = await run_git(["config", "--get", key], cwd=clone)
        assert effective.strip() == value


async def test_gc_migrates_and_repacks_existing_blobless_clone(
    manager: RepoManager, upstream: Path
) -> None:
    git(upstream, "config", "uploadpack.allowFilter", "true")
    await manager.fetch(KEY, upstream.as_uri(), ["+refs/heads/*:refs/heads/*"])
    clone = manager.clone_path(KEY)
    sha = git(upstream, "rev-parse", "HEAD")
    worktree = await manager.checkout_for_build(KEY, "gc-live", base_commit=sha)

    for key, _value in gitrepo.MANAGED_REPO_CONFIG:
        git(clone, "config", key, "true")
        git(clone, "config", "--add", key, "true")
    git(clone, "config", "gc.autoPackLimit", "1")
    gc_log = clone / "gc.log"
    gc_log.write_text(
        "warning: Failed to write bitmap index. "
        "Packfile doesn't have full closure\n"
        "fatal: failed to run repack\n"
    )
    packs_before = len(list((clone / "objects" / "pack").glob("pack-*.pack")))
    assert packs_before > 1

    await manager.gc()

    for key, _value in gitrepo.MANAGED_REPO_CONFIG:
        assert git(clone, "config", "--get-all", key) == "false"
    assert not gc_log.exists()
    packs_after = len(list((clone / "objects" / "pack").glob("pack-*.pack")))
    assert packs_after < packs_before
    assert (worktree.path / "file.txt").read_text() == "hello\n"
    await manager.remove_worktree(worktree)


async def test_unremovable_gc_log_does_not_block_fetch(
    manager: RepoManager,
    upstream: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    await fetch(manager, upstream)
    clone = manager.clone_path(KEY)
    gc_log = clone / "gc.log"
    gc_log.write_text(f"warning: {gitrepo.BITMAP_GC_FAILURE}\n")
    path_type = type(gc_log)
    original_unlink = path_type.unlink

    unlink_error = PermissionError("injected unlink failure")

    def fail_gc_log_unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path == gc_log:
            raise unlink_error
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(path_type, "unlink", fail_gc_log_unlink)
    with caplog.at_level("WARNING"):
        await manager.fetch(KEY, upstream.as_uri(), ["+refs/heads/*:refs/heads/*"])

    assert gc_log.exists()
    assert "failed to clear partial-clone gc log" in caplog.text


@pytest.mark.parametrize("error_kind", ["git", "os"])
async def test_gc_failure_is_logged_without_skipping_other_clones(
    manager: RepoManager,
    upstream: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error_kind: str,
) -> None:
    other_key = "github/acme/other"
    await fetch(manager, upstream)
    await manager.fetch(other_key, upstream.as_uri(), ["+refs/heads/*:refs/heads/*"])
    failed_clone = manager.clone_path(KEY)
    other_clone = manager.clone_path(other_key)
    git(failed_clone, "config", "repack.writeBitmaps", "true")
    git(other_clone, "config", "repack.writeBitmaps", "true")
    gc_attempts: list[Path] = []
    os_error = OSError("injected OS failure")
    original_run_git = gitrepo.run_git

    async def fail_one_gc(
        args: list[str],
        *,
        cwd: Path | None = None,
        credentials: FetchCredentials | None = None,
    ) -> str:
        if args == ["gc", "--auto"]:
            assert cwd is not None
            gc_attempts.append(cwd)
            if cwd == failed_clone:
                if error_kind == "git":
                    raise GitError(args, 1, "injected GIT failure")
                raise os_error
        return await original_run_git(args, cwd=cwd, credentials=credentials)

    monkeypatch.setattr(gitrepo, "run_git", fail_one_gc)
    with caplog.at_level("WARNING"):
        await manager.gc()

    assert set(gc_attempts) == {failed_clone, other_clone}
    assert git(failed_clone, "config", "--get", "repack.writeBitmaps") == "false"
    assert git(other_clone, "config", "--get", "repack.writeBitmaps") == "false"
    failure = next(
        record for record in caplog.records if record.message == "git gc failed"
    )
    assert failure.__dict__["project"] == KEY
    assert failure.__dict__["stderr"] == f"injected {error_kind.upper()} failure"


async def test_repository_reads_wait_for_gc(
    manager: RepoManager,
    upstream: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await fetch(manager, upstream)
    gc_started = asyncio.Event()
    release_gc = asyncio.Event()
    read_started = asyncio.Event()
    original_run_git = gitrepo.run_git

    async def gated_run_git(
        args: list[str],
        *,
        cwd: Path | None = None,
        credentials: FetchCredentials | None = None,
    ) -> str:
        if args == ["gc", "--auto"]:
            gc_started.set()
            await release_gc.wait()
        elif args[0] == "show":
            read_started.set()
        return await original_run_git(args, cwd=cwd, credentials=credentials)

    monkeypatch.setattr(gitrepo, "run_git", gated_run_git)
    gc_task = asyncio.create_task(manager.gc())
    await gc_started.wait()
    read_task = asyncio.create_task(manager.show_file(KEY, "HEAD", "file.txt"))
    await asyncio.sleep(0)
    assert not read_started.is_set()

    release_gc.set()
    await gc_task
    assert await read_task == "hello\n"


async def test_cancelled_worktree_add_is_cleaned_up(
    manager: RepoManager,
    upstream: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await fetch(manager, upstream)
    sha = git(upstream, "rev-parse", "HEAD")
    add_finished = asyncio.Event()
    hold_result = asyncio.Event()
    original_run_git = gitrepo.run_git

    async def pause_after_add(
        args: list[str],
        *,
        cwd: Path | None = None,
        credentials: FetchCredentials | None = None,
    ) -> str:
        output = await original_run_git(args, cwd=cwd, credentials=credentials)
        if args[:2] == ["worktree", "add"]:
            add_finished.set()
            await hold_result.wait()
        return output

    monkeypatch.setattr(gitrepo, "run_git", pause_after_add)
    create_task = asyncio.create_task(manager.create_worktree(KEY, "cancelled", sha))
    await add_finished.wait()
    create_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await create_task

    assert not list(manager.worktrees_dir.iterdir())
    worktrees = await original_run_git(
        ["worktree", "list", "--porcelain"], cwd=manager.clone_path(KEY)
    )
    assert str(manager.worktrees_dir) not in worktrees


async def test_cancelled_worktree_removal_finishes_cleanup(
    manager: RepoManager,
    upstream: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await fetch(manager, upstream)
    sha = git(upstream, "rev-parse", "HEAD")
    worktree = await manager.create_worktree(KEY, "remove-cancelled", sha)
    remove_started = asyncio.Event()
    hold_remove = asyncio.Event()
    original_run_git = gitrepo.run_git

    async def pause_remove(
        args: list[str],
        *,
        cwd: Path | None = None,
        credentials: FetchCredentials | None = None,
    ) -> str:
        if args[:2] == ["worktree", "remove"]:
            remove_started.set()
            await hold_remove.wait()
        return await original_run_git(args, cwd=cwd, credentials=credentials)

    monkeypatch.setattr(gitrepo, "run_git", pause_remove)
    remove_task = asyncio.create_task(manager.remove_worktree(worktree))
    await remove_started.wait()
    assert worktree.path.resolve() in manager._active_worktrees  # noqa: SLF001
    remove_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await remove_task

    assert not worktree.path.exists()
    assert worktree.path.resolve() not in manager._active_worktrees  # noqa: SLF001
    registered = await original_run_git(
        ["worktree", "list", "--porcelain"], cwd=worktree.clone_path
    )
    assert str(worktree.path) not in registered


async def test_worktree_merge_waits_for_gc(
    manager: RepoManager,
    upstream: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await fetch(manager, upstream)
    sha = git(upstream, "rev-parse", "HEAD")
    worktree = await manager.create_worktree(KEY, "merge-lock", sha)
    gc_started = asyncio.Event()
    release_gc = asyncio.Event()
    merge_started = asyncio.Event()
    original_run_git = gitrepo.run_git

    async def gated_run_git(
        args: list[str],
        *,
        cwd: Path | None = None,
        credentials: FetchCredentials | None = None,
    ) -> str:
        if args == ["gc", "--auto"]:
            gc_started.set()
            await release_gc.wait()
        elif "merge" in args:
            merge_started.set()
        return await original_run_git(args, cwd=cwd, credentials=credentials)

    monkeypatch.setattr(gitrepo, "run_git", gated_run_git)
    gc_task = asyncio.create_task(manager.gc())
    await gc_started.wait()
    merge_task = asyncio.create_task(worktree.merge("HEAD"))
    await asyncio.sleep(0)
    merge_waited = not merge_started.is_set()

    release_gc.set()
    await gc_task
    await merge_task
    merge_ran = merge_started.is_set()
    await manager.remove_worktree(worktree)
    assert merge_waited
    assert merge_ran


async def test_pr_merge_and_tree_hash_dedup(
    manager: RepoManager, upstream: Path
) -> None:
    base = git(upstream, "rev-parse", "HEAD")
    git(upstream, "checkout", "-b", "pr")
    (upstream / "feature.txt").write_text("feature\n")
    git(upstream, "add", ".")
    git(upstream, "commit", "-m", "feature")
    head = git(upstream, "rev-parse", "HEAD")
    git(upstream, "checkout", "main")
    await fetch(manager, upstream)

    wt1 = await manager.checkout_for_build(
        KEY, "b1", base_commit=base, head_commit=head
    )
    tree1 = await wt1.tree_hash()
    await manager.remove_worktree(wt1)
    # Same content again (re-push scenario): tree hash identical
    # even though the merge commits differ.
    wt2 = await manager.checkout_for_build(
        KEY, "b2", base_commit=base, head_commit=head
    )
    tree2 = await wt2.tree_hash()
    await manager.remove_worktree(wt2)
    assert tree1 == tree2


async def test_merge_conflict_raises(manager: RepoManager, upstream: Path) -> None:
    git(upstream, "checkout", "-b", "pr")
    (upstream / "file.txt").write_text("pr version\n")
    git(upstream, "commit", "-am", "pr change")
    head = git(upstream, "rev-parse", "HEAD")
    git(upstream, "checkout", "main")
    (upstream / "file.txt").write_text("main version\n")
    git(upstream, "commit", "-am", "main change")
    base = git(upstream, "rev-parse", "HEAD")
    await fetch(manager, upstream)

    with pytest.raises(MergeConflictError):
        await manager.checkout_for_build(KEY, "b3", base_commit=base, head_commit=head)
    # Worktree cleaned up after conflict.
    assert not list(manager.worktrees_dir.glob("b3-*"))


async def test_reclone_on_corruption(manager: RepoManager, upstream: Path) -> None:
    await fetch(manager, upstream)
    clone = manager.clone_path(KEY)
    # Destroy the object store. Next fetch must transparently re-clone.
    shutil.rmtree(clone / "objects")
    await fetch(manager, upstream)
    sha = git(upstream, "rev-parse", "HEAD")

    wt = await manager.checkout_for_build(KEY, "b4", base_commit=sha)
    await manager.remove_worktree(wt)


async def test_transient_fetch_error_keeps_clone(
    manager: RepoManager, upstream: Path
) -> None:
    await fetch(manager, upstream)
    clone = manager.clone_path(KEY)
    # Unreachable remote but healthy clone: the clone (which backs
    # in-flight builds' worktrees) must survive.
    with pytest.raises(GitError):
        await manager.fetch(
            KEY, str(upstream / "does-not-exist"), ["+refs/heads/*:refs/heads/*"]
        )
    assert (clone / "HEAD").exists()
    assert any((clone / "objects").rglob("*"))


async def test_cleanup_resolves_symlinked_paths(tmp_path: Path, upstream: Path) -> None:
    # git reports symlink-resolved worktree paths. Cleanup must compare
    # resolved paths or it deletes live worktrees.
    real_state = tmp_path / "real-state"
    real_state.mkdir()
    link = tmp_path / "state-link"
    link.symlink_to(real_state)
    manager = RepoManager(link)
    await fetch(manager, upstream)
    sha = git(upstream, "rev-parse", "HEAD")
    live = await manager.checkout_for_build(KEY, "live", base_commit=sha)
    await manager.cleanup()
    assert live.path.exists()


@pytest.fixture
def submodule(upstream: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Add a file:// submodule to upstream. Allows the file protocol
    (blocked by default since CVE-2022-39253) via environment-based
    git config passed through by run_git."""
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "protocol.file.allow")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "always")
    sub = upstream.parent / "submodule"
    sub.mkdir()
    git(sub, "init", "-b", "main")
    (sub / "inner.txt").write_text("inner\n")
    git(sub, "add", ".")
    git(sub, "commit", "-m", "inner")
    git(
        upstream,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(sub),
        "vendored",
    )
    git(upstream, "commit", "-m", "add submodule")
    return sub


@pytest.mark.usefixtures("submodule")
async def test_submodules_checked_out(manager: RepoManager, upstream: Path) -> None:
    await fetch(manager, upstream)
    sha = git(upstream, "rev-parse", "HEAD")

    wt = await manager.checkout_for_build(KEY, "sub-build", base_commit=sha)
    assert (wt.path / "vendored" / "inner.txt").read_text() == "inner\n"
    await manager.remove_worktree(wt)


@pytest.mark.usefixtures("submodule")
async def test_submodule_fetch_does_not_hold_repository_lock(
    manager: RepoManager,
    upstream: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await fetch(manager, upstream)
    sha = git(upstream, "rev-parse", "HEAD")
    submodule_started = asyncio.Event()
    release_submodule = asyncio.Event()
    read_started = asyncio.Event()
    original_run_git = gitrepo.run_git

    async def gated_run_git(
        args: list[str],
        *,
        cwd: Path | None = None,
        credentials: FetchCredentials | None = None,
    ) -> str:
        if args[:2] == ["submodule", "update"]:
            submodule_started.set()
            await release_submodule.wait()
        elif args[0] == "show":
            read_started.set()
        return await original_run_git(args, cwd=cwd, credentials=credentials)

    monkeypatch.setattr(gitrepo, "run_git", gated_run_git)
    checkout_task = asyncio.create_task(
        manager.checkout_for_build(KEY, "sub-lock", base_commit=sha)
    )
    await submodule_started.wait()
    read_task = asyncio.create_task(manager.show_file(KEY, "HEAD", "file.txt"))
    await asyncio.sleep(0)
    read_did_not_wait = read_started.is_set()

    release_submodule.set()
    worktree = await checkout_task
    assert await read_task == "hello\n"
    await manager.remove_worktree(worktree)
    assert read_did_not_wait


async def test_failed_submodule_checkout_removes_worktree(
    manager: RepoManager, upstream: Path, submodule: Path
) -> None:
    await fetch(manager, upstream)
    sha = git(upstream, "rev-parse", "HEAD")
    # Submodule source gone: the update fails and the half-initialized
    # worktree must not leak (it would stay registered forever).
    shutil.rmtree(submodule)

    with pytest.raises(GitError):
        await manager.checkout_for_build(KEY, "sub-fail", base_commit=sha)
    assert not list(manager.worktrees_dir.glob("sub-fail-*"))


async def test_cleanup_rechecks_active_effect_clones_before_sweep(
    manager: RepoManager,
    upstream: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await fetch(manager, upstream)
    effect_clone = manager.worktrees_dir / "effect-race"
    effect_clone.mkdir()
    (effect_clone / "live").write_text("still running")
    list_started = asyncio.Event()
    release_list = asyncio.Event()
    original_run_git = gitrepo.run_git

    async def gated_run_git(
        args: list[str],
        *,
        cwd: Path | None = None,
        credentials: FetchCredentials | None = None,
    ) -> str:
        if args == ["worktree", "list", "--porcelain"]:
            list_started.set()
            await release_list.wait()
        return await original_run_git(args, cwd=cwd, credentials=credentials)

    monkeypatch.setattr(gitrepo, "run_git", gated_run_git)
    cleanup_task = asyncio.create_task(manager.cleanup())
    await list_started.wait()
    manager._active_worktrees.add(effect_clone.resolve())  # noqa: SLF001
    release_list.set()
    await cleanup_task

    assert (effect_clone / "live").read_text() == "still running"
    manager.remove_effect_clone(effect_clone)


async def test_cleanup_sweeps_orphans(manager: RepoManager, upstream: Path) -> None:
    await fetch(manager, upstream)
    sha = git(upstream, "rev-parse", "HEAD")

    live = await manager.checkout_for_build(KEY, "live", base_commit=sha)
    orphan = manager.worktrees_dir / "orphan"
    orphan.mkdir()
    (orphan / "junk").write_text("x")

    await manager.cleanup()
    assert not orphan.exists()
    assert live.path.exists()
    await manager.gc()


async def test_cleanup_prunes_stale_pr_refs(
    manager: RepoManager, upstream: Path
) -> None:
    """PR refs accumulate forever otherwise: --prune only covers the
    refspecs of the current fetch."""
    sha = git(upstream, "rev-parse", "HEAD")
    old_env = {
        "GIT_COMMITTER_DATE": "2005-04-07T22:13:13",
        "GIT_AUTHOR_DATE": "2005-04-07T22:13:13",
    }
    subprocess.run(  # noqa: S603
        ["git", "-C", str(upstream), "commit", "--allow-empty", "-m", "old pr"],
        env={**os.environ, **old_env},
        check=True,
    )
    old_sha = git(upstream, "rev-parse", "HEAD")
    git(upstream, "update-ref", "refs/pull/1/head", old_sha)
    git(upstream, "update-ref", "refs/merge-requests/2/head", old_sha)
    git(upstream, "update-ref", "refs/pull/3/head", sha)  # recent
    git(upstream, "reset", "--hard", sha)
    await manager.fetch(
        KEY,
        str(upstream),
        [
            "+refs/heads/*:refs/heads/*",
            "+refs/pull/1/*:refs/pull/1/*",
            "+refs/merge-requests/2/*:refs/merge-requests/2/*",
            "+refs/pull/3/*:refs/pull/3/*",
        ],
    )
    await manager.cleanup()
    clone = manager.clone_path(KEY)
    refs = git(clone, "for-each-ref", "--format=%(refname)")
    assert "refs/pull/1/head" not in refs
    assert "refs/merge-requests/2/head" not in refs
    assert "refs/pull/3/head" in refs
    assert "refs/heads/main" in refs or "refs/heads/master" in refs


async def test_cleanup_removes_stale_orphan_files(
    manager: RepoManager, upstream: Path
) -> None:
    """Crash-leaked side-files next to worktrees (e.g. effects secrets)
    must be swept once old enough. Fresh files stay."""
    await fetch(manager, upstream)
    stale = manager.worktrees_dir / "stale-secret"
    stale.write_text("s3cret")
    old = time.time() - 2 * 86400
    os.utime(stale, (old, old))
    fresh = manager.worktrees_dir / "fresh-file"
    fresh.write_text("x")
    await manager.cleanup()
    assert not stale.exists()
    assert fresh.exists()


async def test_cleanup_aborts_when_worktree_list_fails(
    manager: RepoManager, upstream: Path
) -> None:
    """A failing `git worktree list` must abort the sweep (fail
    closed), not be treated as "no worktrees"."""
    await fetch(manager, upstream)
    sha = git(upstream, "rev-parse", "HEAD")
    live = await manager.checkout_for_build(KEY, "live", base_commit=sha)
    # Forget the in-memory registration to exercise the git-metadata
    # path alone, then corrupt the clone so `git worktree list` fails.
    manager._active_worktrees.clear()  # noqa: SLF001
    (manager.clone_path(KEY) / "HEAD").unlink()
    await manager.cleanup()
    assert live.path.exists()


async def test_cleanup_keeps_registered_worktrees_after_reclone(
    manager: RepoManager, upstream: Path
) -> None:
    """After a corruption re-clone the new clone knows no worktrees;
    live builds' worktrees must survive via the in-memory registry."""
    await fetch(manager, upstream)
    sha = git(upstream, "rev-parse", "HEAD")
    live = await manager.checkout_for_build(KEY, "live", base_commit=sha)
    # Simulate corruption + re-clone: fresh clone, no registered worktrees.
    shutil.rmtree(manager.clone_path(KEY))
    await fetch(manager, upstream)
    await manager.cleanup()
    assert live.path.exists()


@pytest.mark.usefixtures("submodule")
async def test_submodules_fetched_without_credentials(
    manager: RepoManager,
    upstream: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """.gitmodules is PR-controlled: a malicious PR could point a
    submodule at another private repo on the same forge and exfiltrate
    it via build outputs. Submodule checkout must not see the fetch
    credentials."""
    await fetch(manager, upstream)
    sha = git(upstream, "rev-parse", "HEAD")
    netrc = tmp_path / "netrc"
    netrc.write_text("machine example.com login x password y\n")
    creds = FetchCredentials(netrc_file=netrc)

    calls: list[tuple[list[str], FetchCredentials | None]] = []
    orig_run_git = gitrepo.run_git

    async def spy_run_git(
        args: list[str],
        *,
        cwd: Path | None = None,
        credentials: FetchCredentials | None = None,
    ) -> str:
        calls.append((args, credentials))
        return await orig_run_git(args, cwd=cwd, credentials=credentials)

    monkeypatch.setattr(gitrepo, "run_git", spy_run_git)

    # Default: no credentials reach the submodule checkout.
    wt = await manager.checkout_for_build(
        KEY, "sub-creds", base_commit=sha, credentials=creds
    )
    await manager.remove_worktree(wt)
    # Explicit opt-in forwards them.
    wt = await manager.checkout_for_build(
        KEY,
        "sub-creds-2",
        base_commit=sha,
        credentials=creds,
        submodule_credentials=creds,
    )
    await manager.remove_worktree(wt)
    submodule_calls = [c for c in calls if c[0][0] == "submodule"]
    assert len(submodule_calls) == 2  # noqa: PLR2004 — one call per checkout
    assert submodule_calls[0][1] is None
    assert submodule_calls[1][1] is creds


async def test_checkout_lazy_fetch_needs_credentials(
    manager: RepoManager,
    upstream: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blob-less clone lazy-fetches missing objects from origin during
    worktree checkout and merge. When origin needs authentication, those
    fetches only succeed if the primary repo's credentials are
    forwarded, reproducing the private-repo build failure.

    A fake `ssh` gates on the `-i <key>` flag that
    FetchCredentials.git_ssh_command emits: no credentials means no key
    means the promisor fetch fails just like the real bug."""
    git(upstream, "config", "uploadpack.allowFilter", "true")
    base = git(upstream, "rev-parse", "HEAD")
    git(upstream, "checkout", "-b", "pr")
    (upstream / "feature.txt").write_text("feature\n")
    git(upstream, "add", ".")
    git(upstream, "commit", "-m", "feature")
    head = git(upstream, "rev-parse", "HEAD")
    git(upstream, "checkout", "main")

    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake_ssh = bindir / "ssh"
    fake_ssh.write_text(
        "#!/bin/sh\n"
        'case " $* " in *" -i "*) ;; *) echo "no key" >&2; exit 255;; esac\n'
        'for a in "$@"; do last=$a; done\n'
        'exec sh -c "$last"\n'
    )
    fake_ssh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")

    key = tmp_path / "id"
    key.write_text("dummy\n")
    creds = FetchCredentials(ssh_private_key_file=key)
    # ssh://fake<abs-path> routes through the fake ssh, which runs
    # git-upload-pack against the local upstream repo.
    url = f"ssh://fake{upstream}"
    await manager.fetch(KEY, url, ["+refs/heads/*:refs/heads/*"], credentials=creds)

    with pytest.raises(GitError, match="promisor remote"):
        await manager.checkout_for_build(
            KEY, "no-creds", base_commit=base, head_commit=head
        )

    wt = await manager.checkout_for_build(
        KEY, "with-creds", base_commit=base, head_commit=head, credentials=creds
    )
    try:
        assert (wt.path / "feature.txt").read_text() == "feature\n"
    finally:
        await manager.remove_worktree(wt)


async def test_static_credentials_provider(tmp_path: Path) -> None:
    netrc = tmp_path / "netrc"
    netrc.write_text("machine example.com login x password y\n")
    provider = StaticCredentialsProvider(netrc)
    creds = await provider.get("https://example.com/r.git")
    assert creds.netrc_file == netrc
    assert (await StaticCredentialsProvider().get("https://x/y.git")).netrc_file is None


async def test_run_git_error_includes_stderr(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="failed"):
        await run_git(["rev-parse", "HEAD"], cwd=tmp_path)


async def test_wait_uncancelled_propagates_inner_cancellation() -> None:
    task = asyncio.create_task(asyncio.sleep(60))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await gitrepo._wait_uncancelled(task)  # noqa: SLF001


async def test_run_git_cancellation_kills_process_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    parent_pid_file = tmp_path / "parent.pid"
    child_pid_file = tmp_path / "child.pid"
    fake = bindir / "git"
    fake.write_text(
        "#!/bin/sh\n"
        f"echo $$ > {shlex.quote(str(parent_pid_file))}\n"
        "sleep 30 &\n"
        f"echo $! > {shlex.quote(str(child_pid_file))}\n"
        "wait\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    task = asyncio.create_task(run_git(["status"]))
    pids: list[int] = []
    try:
        for _attempt in range(100):
            if parent_pid_file.exists() and child_pid_file.exists():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("fake git process did not start")
        pids = [int(parent_pid_file.read_text()), int(child_pid_file.read_text())]

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        for _attempt in range(100):
            if not any(_process_exists(pid) for pid in pids):
                break
            await asyncio.sleep(0.01)
        assert not any(_process_exists(pid) for pid in pids)
    finally:
        task.cancel()
        for pid in pids:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def test_merge_infra_failure_is_not_conflict(
    manager: RepoManager, upstream: Path
) -> None:
    # A git failure that is not a content conflict (here: merging a
    # nonexistent ref) must surface as GitError so callers treat it as
    # transient/infra, not as a permanent merge conflict.
    base = git(upstream, "rev-parse", "HEAD")
    await fetch(manager, upstream)

    wt = await manager.create_worktree(KEY, "b-infra", base)
    try:
        with pytest.raises(GitError):
            await wt.merge("0" * 40)
    finally:
        await manager.remove_worktree(wt)


async def test_run_git_passes_proxy_env_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fetches go through libcurl: the service's proxy/CA configuration
    # must survive run_git's env scrubbing.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "git"
    fake.write_text('#!/bin/sh\necho "proxy=$https_proxy ca=$SSL_CERT_FILE"\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setenv("https_proxy", "http://proxy:3128")
    monkeypatch.setenv("SSL_CERT_FILE", "/etc/ssl/ca.pem")
    out = await run_git(["version"])
    assert out.strip() == "proxy=http://proxy:3128 ca=/etc/ssl/ca.pem"


@pytest.mark.usefixtures("submodule")
async def test_clone_for_effect(
    manager: RepoManager, upstream: Path, tmp_path: Path
) -> None:
    await fetch(manager, upstream)
    sha = git(upstream, "rev-parse", "HEAD")
    # Production ordering: the build worktree runs first and materializes
    # the blobs of this commit in the blobless mirror.
    await manager.checkout_for_build(KEY, "eval", base_commit=sha)
    dest = tmp_path / "effect-checkout"

    await manager.clone_for_effect(
        KEY,
        dest,
        commit=sha,
        push_url="https://x-access-token:tok123@github.com/acme/widget",
    )
    # Standalone: own object store, no alternates into the shared clone.
    assert git(dest, "rev-parse", "--git-common-dir") == ".git"
    assert not (dest / ".git" / "objects" / "info" / "alternates").exists()
    assert git(dest, "rev-parse", "HEAD") == sha
    assert (dest / "vendored" / "inner.txt").read_text() == "inner\n"
    # Pushable origin plus ssh/https rewrites, in the clone's config only.
    assert git(dest, "remote", "get-url", "origin") == (
        "https://x-access-token:tok123@github.com/acme/widget"
    )
    assert "git@github.com:" in (dest / ".git" / "config").read_text()
    assert "tok123" not in (manager.clone_path(KEY) / "config").read_text()
