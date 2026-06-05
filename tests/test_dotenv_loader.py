"""tushare_config._load_dotenv 的单测：覆盖 BOM/CRLF/引号/注释/不覆盖已有键。

回归保护：此前用 utf-8 打开含 BOM 的 .env 会把首行键读成 \\ufeffKEY 而静默失效。
"""

from __future__ import annotations

from factorzen.config.tushare_config import _load_dotenv


def test_load_plain(tmp_path):
    f = tmp_path / ".env"
    f.write_text("TUSHARE_TOKEN=abc123\n", encoding="utf-8")
    env: dict[str, str] = {}
    _load_dotenv(f, env)
    assert env["TUSHARE_TOKEN"] == "abc123"


def test_load_strips_bom(tmp_path):
    """带 UTF-8 BOM 的首行键不应被污染成 \\ufeffKEY。"""
    f = tmp_path / ".env"
    f.write_bytes(b"\xef\xbb\xbfTUSHARE_TOKEN=abc123\n")
    env: dict[str, str] = {}
    _load_dotenv(f, env)
    assert env["TUSHARE_TOKEN"] == "abc123"
    assert "﻿TUSHARE_TOKEN" not in env


def test_load_handles_crlf(tmp_path):
    f = tmp_path / ".env"
    f.write_bytes(b"TUSHARE_TOKEN=abc123\r\nFOO=bar\r\n")
    env: dict[str, str] = {}
    _load_dotenv(f, env)
    assert env["TUSHARE_TOKEN"] == "abc123"
    assert env["FOO"] == "bar"


def test_load_strips_quotes(tmp_path):
    f = tmp_path / ".env"
    f.write_text('A="dq"\nB=\'sq\'\n', encoding="utf-8")
    env: dict[str, str] = {}
    _load_dotenv(f, env)
    assert env["A"] == "dq"
    assert env["B"] == "sq"


def test_load_skips_comments_and_blanks(tmp_path):
    f = tmp_path / ".env"
    f.write_text("# comment\n\nKEY=val\n   \n", encoding="utf-8")
    env: dict[str, str] = {}
    _load_dotenv(f, env)
    assert env == {"KEY": "val"}


def test_load_does_not_override_existing(tmp_path):
    f = tmp_path / ".env"
    f.write_text("TUSHARE_TOKEN=from_file\n", encoding="utf-8")
    env = {"TUSHARE_TOKEN": "from_env"}
    _load_dotenv(f, env)
    assert env["TUSHARE_TOKEN"] == "from_env"  # 已有键不被覆盖


def test_load_missing_file_is_noop(tmp_path):
    env: dict[str, str] = {}
    _load_dotenv(tmp_path / "nope.env", env)
    assert env == {}


def test_load_value_with_equals_sign(tmp_path):
    """值内含 '=' 时按首个 '=' 切分，保留其余。"""
    f = tmp_path / ".env"
    f.write_text("URL=https://x/y?a=1&b=2\n", encoding="utf-8")
    env: dict[str, str] = {}
    _load_dotenv(f, env)
    assert env["URL"] == "https://x/y?a=1&b=2"
