# mltools

## 介绍

mltools：一个专注于机器学习工具开发的开源项目。

## 安装命令

```bash
pip install wliafe-mltools
```

建议使用uv安装。

```bash
uv add wliafe-mltools
```

## 使用说明

1. mltools需要python3.8及以上版本。
2. 样例可以参考example文件夹。

## 版本更新

### 1.0.12

+ `utils` 模块下新增 `set_readthedocs` 函数，用于设置 Read the Docs 文档的 Python 版本。

### 1.0.11

+ 新增库文档
+ 创建 `results` 和 `data` 文件夹时自动创建 `.gitignore` 文件。
+ 新增 `utils` 模块，包含以下函数：
  + `write_requirements` 函数，用于从 `pyproject.toml` 中提取依赖并写入 `requirements.txt`。
  + `run_command` 函数，用于运行 shell 命令。
  + `build_docs` 函数，用于构建文档。
