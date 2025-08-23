import shutil
from pathlib import Path
import subprocess
from utils import requirements


def build_docs():
    """
    构建文档
    """
    requirements.write_requirements()
    html_dir = Path("docs/build/html")
    if html_dir.exists():
        shutil.rmtree(html_dir)
    doctrees_dir = Path("docs/build/doctrees")
    if doctrees_dir.exists():
        shutil.rmtree(doctrees_dir)
    process = subprocess.Popen(
        ["uv", "run", "sphinx-build", "-M", "html", "docs/source/", "docs/build/"],
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
