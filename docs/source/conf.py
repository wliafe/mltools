# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path
import tomllib

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # 项目根目录

sys.path.insert(0, str(ROOT / "src"))

with open(ROOT / "pyproject.toml", "rb") as f:
    config = tomllib.load(f)

project = "mltools"
copyright = "2025, wliafe"
author = "wliafe"
release = config["project"]["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = []

language = "zh_CN"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
