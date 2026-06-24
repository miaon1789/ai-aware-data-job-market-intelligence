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
