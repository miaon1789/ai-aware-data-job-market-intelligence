import os

from scripts.purge_private_data import expired_files


def test_expired_files_selects_old_files_without_deleting(tmp_path):
    old = tmp_path / "old.json"
    recent = tmp_path / "recent.csv"
    old.write_text("private")
    recent.write_text("private")
    now = 2_000_000_000.0
    os.utime(old, (now - 100 * 86400, now - 100 * 86400))
    os.utime(recent, (now - 10 * 86400, now - 10 * 86400))

    selected = expired_files(tmp_path, retention_days=90, now=now)
    assert selected == [old]
    assert old.exists()
    assert recent.exists()


def test_docker_volume_helpers_degrade_without_docker(monkeypatch):
    """A machine with no Docker must still run the filesystem purge."""

    import scripts.purge_private_data as purge

    monkeypatch.setattr(purge.shutil, "which", lambda name: None)
    assert purge.existing_private_volumes() == []


def test_only_known_private_volumes_are_reported(monkeypatch):
    """The purge must never propose removing a volume it does not own."""

    import subprocess

    import scripts.purge_private_data as purge

    monkeypatch.setattr(purge, "_docker_available", lambda: True)
    monkeypatch.setattr(
        purge.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a[0], 0, stdout="doccano-db\npostgres-data\nsomeone-elses-volume\n", stderr=""
        ),
    )
    assert purge.existing_private_volumes() == ["doccano-db"]


def test_the_documented_private_volume_is_the_doccano_database():
    """docs/ADZUNA_USAGE.md promises a complete purge; this is what it covers."""

    import scripts.purge_private_data as purge

    assert "doccano-db" in purge.PRIVATE_DOCKER_VOLUMES
