from factorzen.config import tushare_config
from factorzen.config.settings import ROOT


def test_tushare_env_file_is_project_root_env():
    assert tushare_config._env_file == ROOT / ".env"
