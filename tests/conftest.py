"""pytest 配置：确保项目根目录与 tests 目录在 sys.path 中。"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for path in (ROOT, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(autouse=True)
def _isolate_kb_index(tmp_path_factory, monkeypatch):
    """把检索默认指向空索引目录。

    否则本机一旦建过真实索引，跑测试会去加载 BGE-M3 并依赖真实语料，
    既慢又不可复现。个别用例需要别的目录时自行 monkeypatch.setenv 覆盖。
    """
    from kb.search import clear_cache

    empty_dir = tmp_path_factory.mktemp("kb_index_empty")
    monkeypatch.setenv("KB_INDEX_DIR", str(empty_dir))
    clear_cache()
    yield
    clear_cache()
