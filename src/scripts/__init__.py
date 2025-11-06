import yaml
import shutil
import tomllib
from pathlib import Path
from mltools import utils

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # 项目根目录

with open(ROOT / "pyproject.toml", "rb") as f:
    config = tomllib.load(f)


def set_readthedocs():
    """
    设置 readthedocs 配置文件
    """
    with open(ROOT / ".python-version", "r") as file:
        python_version = file.readline().strip()
    with open(ROOT / "docs/.readthedocs.yaml", "r+", encoding="utf-8") as file:
        data = yaml.safe_load(file)
        data["build"]["tools"]["python"] = python_version
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
    utils.bash_command(["uv", "sync"])
    utils.bash_command(["uv", "run", "sphinx-build", "-M", "html", "docs/source/", "docs/build/"])
