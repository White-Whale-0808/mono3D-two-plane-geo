"""
utils/env_setup.py
載入 .env 並設定平台相關的動態函式庫搜尋路徑。

必須在任何會觸發 C extension import（pyelsed、cv2）的 import 之前呼叫：
    from utils.env_setup import setup_env
    setup_env()
"""

import os
import pathlib

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def setup_env() -> None:
    _load_dotenv()
    _register_dll_paths()


def _load_dotenv() -> None:
    env_file = _PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


def _register_dll_paths() -> None:
    # Windows (Python 3.8+) 不再從 PATH 自動搜尋 DLL，需明確呼叫 add_dll_directory。
    # macOS / Linux 由動態連結器自動處理，不需要任何額外設定。
    if not hasattr(os, "add_dll_directory"):
        return
    opencv_bin = os.environ.get("OPENCV_BIN_PATH", "")
    if opencv_bin and os.path.isdir(opencv_bin):
        os.add_dll_directory(opencv_bin)
        os.environ["PATH"] = opencv_bin + os.pathsep + os.environ.get("PATH", "")
