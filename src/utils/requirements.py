import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # 项目根目录

if sys.version_info >= (3, 11):
    # Python 3.11 或更高
    import tomllib as tomli
else:
    # Python 3.8 ~ 3.10
    import tomli

def write_requirements():
    """
    从 pyproject.toml 中提取依赖并写入 requirements.txt
    """
    with open(ROOT / "pyproject.toml", "rb") as f:
        config = tomli.load(f)
    with open(ROOT / "requirements.txt", "w") as f:
        for dep in config["project"]["dependencies"]:
            f.write(dep + "\n")
    with open(ROOT / "docs/requirements.txt", "w") as f:
        for dep in config["dependency-groups"]["dev"]:
            f.write(dep + "\n")