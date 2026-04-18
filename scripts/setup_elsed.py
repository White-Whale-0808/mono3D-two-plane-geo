"""
scripts/setup_elsed.py
自動 clone pyelsed 並套用 Python 3.12 / Windows 相容補丁。
執行方式：python scripts/setup_elsed.py
"""
import subprocess
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
ELSED_SRC = ROOT / "elsed_src"
REPO_URL = "https://github.com/iago-suarez/ELSED.git"
PYBIND11_TAG = "v2.13.6"


def run(cmd, **kwargs):
    print(f"  > {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def main():
    # 1. Clone（若已存在則跳過）
    if not ELSED_SRC.exists():
        print("[1/5] Clone pyelsed ...")
        run(["git", "clone", REPO_URL, str(ELSED_SRC)])
    else:
        print("[1/5] elsed_src 已存在，跳過 clone")

    # 2. pybind11 是 submodule，clone 後必須明確初始化才能取得內容
    print("[2/5] 初始化 git submodule（含 pybind11）...")
    run(["git", "-C", str(ELSED_SRC), "submodule", "update", "--init", "--recursive"])

    # 3. CMakeLists.txt：將最小版本改為 3.5 以相容現代 CMake
    print("[3/5] 修正 CMakeLists.txt ...")
    cmake = ELSED_SRC / "CMakeLists.txt"
    lines = cmake.read_text(encoding="utf-8").splitlines()
    lines[0] = "cmake_minimum_required(VERSION 3.5)"
    cmake.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 4. EdgeDrawer.h：加入 _USE_MATH_DEFINES 解決 Windows 找不到 M_PI
    print("[4/5] 修正 EdgeDrawer.h ...")
    header = ELSED_SRC / "src" / "EdgeDrawer.h"
    content = header.read_text(encoding="utf-8")
    define = "#define _USE_MATH_DEFINES"
    if define not in content:
        header.write_text(define + "\n" + content, encoding="utf-8")

    # 5. pybind11：切換至 v2.13.6 以相容 Python 3.12 C-API
    print(f"[5/5] 切換 pybind11 至 {PYBIND11_TAG} ...")
    run(["git", "-C", str(ELSED_SRC / "pybind11"), "checkout", "-f", PYBIND11_TAG])

    print("\nelsed_src 設定完成！接下來執行：uv sync")


if __name__ == "__main__":
    main()
