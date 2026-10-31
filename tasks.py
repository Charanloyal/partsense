"""Cross-platform task runner for PartSense pipeline (Windows, macOS, Linux).

Usable directly via `python tasks.py <command>` or via `invoke <command>`.
"""

from __future__ import annotations

import subprocess
import sys

try:
    from invoke import task  # type: ignore
except ImportError:
    def task(func):
        return func


def _run(cmd: list[str]) -> int:
    """Execute command in current environment and stream output."""
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode


@task
def install(c=None) -> None:
    """Install runtime dependencies."""
    _run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task
def install_dev(c=None) -> None:
    """Install dev dependencies and editable package."""
    _run([sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"])
    _run([sys.executable, "-m", "pip", "install", "-e", "."])


@task
def data(c=None) -> None:
    """Ingest, validate, and prepare data splits."""
    _run([sys.executable, "-m", "partsense.data.pipeline", "--stage", "data"])


@task
def synthetic(c=None) -> None:
    """Generate seeded synthetic warranty and aftersales layer in DuckDB."""
    _run([sys.executable, "-m", "partsense.synthetic"])


@task
def train(c=None) -> None:
    """Train XGBoost failure classifier, OvR diagnoser, and Isolation Forest."""
    _run([sys.executable, "-m", "partsense.models.train"])


@task
def evaluate(c=None) -> None:
    """Run model evaluation, threshold tuning, SHAP explainability, and metrics export."""
    _run([sys.executable, "-m", "partsense.models.evaluate"])


@task
def test(c=None) -> None:
    """Run pytest suite."""
    _run([sys.executable, "-m", "pytest", "tests/"])


@task
def lint(c=None) -> None:
    """Run ruff linter and format checker."""
    _run([sys.executable, "-m", "ruff", "check", "src/", "tests/", "app/"])
    _run([sys.executable, "-m", "ruff", "format", "--check", "src/", "tests/", "app/"])


@task
def format(c=None) -> None:
    """Format and auto-fix code."""
    _run([sys.executable, "-m", "ruff", "format", "src/", "tests/", "app/"])
    _run([sys.executable, "-m", "ruff", "check", "--fix", "src/", "tests/", "app/"])


@task
def app(c=None) -> None:
    """Run Streamlit web dashboard."""
    _run([sys.executable, "-m", "streamlit", "run", "app/Home.py"])


if __name__ == "__main__":
    commands = {
        "install": install,
        "install-dev": install_dev,
        "data": data,
        "synthetic": synthetic,
        "train": train,
        "evaluate": evaluate,
        "test": test,
        "lint": lint,
        "format": format,
        "app": app,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: python tasks.py [{'|'.join(commands.keys())}]")
        sys.exit(1)

    target_func = commands[sys.argv[1]]
    if hasattr(target_func, "body"):
        target_func.body(None)
    else:
        target_func(None)
