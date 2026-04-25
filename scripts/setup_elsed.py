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

    # 3. CMakeLists.txt：將最小版本改為 3.5，並加入 _USE_MATH_DEFINES 全域定義
    print("[3/5] 修正 CMakeLists.txt ...")
    cmake = ELSED_SRC / "CMakeLists.txt"
    lines = cmake.read_text(encoding="utf-8").splitlines()
    lines[0] = "cmake_minimum_required(VERSION 3.5)"
    math_def_block = (
        "\n# Windows MSVC requires this to expose M_PI / M_PI_2 from <cmath>\n"
        "if(MSVC)\n    add_compile_definitions(_USE_MATH_DEFINES)\nendif()\n"
    )
    cxx_std_line = next(
        (i for i, l in enumerate(lines) if "CMAKE_CXX_STANDARD" in l), None
    )
    if cxx_std_line is not None and "_USE_MATH_DEFINES" not in "\n".join(lines):
        lines.insert(cxx_std_line + 1, math_def_block)
    cmake.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 4. EdgeDrawer.h：加入 _USE_MATH_DEFINES 解決 Windows 找不到 M_PI
    print("[4/6] 修正 EdgeDrawer.h ...")
    header = ELSED_SRC / "src" / "EdgeDrawer.h"
    content = header.read_text(encoding="utf-8")
    define = "#define _USE_MATH_DEFINES"
    if define not in content:
        header.write_text(define + "\n" + content, encoding="utf-8")

    # 5. setup.py：讀取 OpenCV_DIR / OPENCV_DIR 環境變數並傳給 CMake
    #    opencv-python (PyPI) 只含 Python binding，不含 CMake 能找到的 C++ headers，
    #    必須靠系統安裝的 OpenCV SDK，路徑透過環境變數傳入。
    print("[5/6] 修正 setup.py（注入 OpenCV_DIR）...")
    setup_py = ELSED_SRC / "setup.py"
    setup_src = setup_py.read_text(encoding="utf-8")
    inject = (
        "            opencv_dir = os.environ.get('OpenCV_DIR') or os.environ.get('OPENCV_DIR')\n"
        "            if opencv_dir:\n"
        "                cmake_args += ['-DOpenCV_DIR=' + opencv_dir]\n"
    )
    marker = "            build_args += ['--', '/m']\n"
    if inject not in setup_src:
        setup_py.write_text(setup_src.replace(marker, marker + inject), encoding="utf-8")

    # 6. pybind11：切換至 v2.13.6 以相容 Python 3.12 C-API
    print(f"[6/6] 切換 pybind11 至 {PYBIND11_TAG} ...")
    run(["git", "-C", str(ELSED_SRC / "pybind11"), "checkout", "-f", PYBIND11_TAG])

    print("\nelsed_src 設定完成！接下來執行：uv sync")


if __name__ == "__main__":
    main()
