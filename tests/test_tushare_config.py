from factorzen.config import tushare_config
from factorzen.config.settings import ROOT


def test_tushare_env_file_is_project_root_env():
    assert tushare_config._env_file == ROOT / ".env"


def test_int_env_strips_inline_comment_and_falls_back(monkeypatch):
    """行内注释/非数字值不应让 import 期 int() 崩溃：剥注释、失败回退默认。"""
    from factorzen.config.tushare_config import _int_env

    monkeypatch.setenv("X_TEST_INT", "2000 # 注释")
    assert _int_env("X_TEST_INT", "100") == 2000
    monkeypatch.setenv("X_TEST_INT", "notanumber")
    assert _int_env("X_TEST_INT", "100") == 100
