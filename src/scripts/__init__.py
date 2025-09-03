import yaml
import shutil
import tomllib
import subprocess
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # 项目根目录

with open(ROOT / "pyproject.toml", "rb") as f:
    config = tomllib.load(f)


def set_readthedocs():
    with open(ROOT / "docs/.readthedocs.yaml", "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    with open(ROOT / ".python-version", "r") as file:
        python_version = file.readline().strip()
    data["build"]["tools"]["python"] = python_version
    with open(ROOT / "docs/.readthedocs.yaml", "w", encoding="utf-8") as file:
        yaml.dump(data, file)


def write_requirements():
    """
    从 pyproject.toml 中提取依赖并写入 requirements.txt
    """
    with open(ROOT / "requirements.txt", "w") as f:
        for dep in config["project"]["dependencies"]:
            f.write(dep + "\n")
    with open(ROOT / "docs/requirements.txt", "w") as f:
        for dep in config["dependency-groups"]["docs"]:
            f.write(dep + "\n")


def run_command(command):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # 合并错误输出到标准输出
        text=True,  # 返回字符串而非字节
        bufsize=1,  # 行缓冲模式
    )
    while True:
        line = process.stdout.readline()
        if not line:
            break
        print(line.strip())


def build_docs():
    """
    构建文档
    """
    set_readthedocs()
    write_requirements()
    html_dir = Path("docs/build/html")
    if html_dir.exists():
        shutil.rmtree(html_dir)
    doctrees_dir = Path("docs/build/doctrees")
    if doctrees_dir.exists():
        shutil.rmtree(doctrees_dir)
    run_command(["uv", "sync"])
    run_command(["uv", "run", "sphinx-build", "-M", "html", "docs/source/", "docs/build/"])
