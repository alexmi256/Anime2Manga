venv := ".venv"
python := venv / "bin/python"
ruff := venv / "bin/ruff"
pyrefly := venv / "bin/pyrefly"
output_dir := "output"

default:
    @just --list

# Create/refresh the virtual environment and install dependencies.
install:
    python3 -m venv {{venv}}
    {{python}} -m pip install --upgrade pip
    {{python}} -m pip install -r requirements-dev.txt
    {{python}} -m pip install -e .

# Run the test suite.
test:
    {{python}} -m pytest

# Run the test suite with coverage.
coverage:
    {{python}} -m pytest --cov=anime2manga --cov-report=term-missing

# Lint the source and tests.
lint:
    {{ruff}} check src tests

# Auto-format the source and tests.
format:
    {{ruff}} format src tests

# Type-check the source and tests.
typecheck:
    {{pyrefly}} check src tests

# Everything CI would run.
check: lint typecheck test

# Convert the bundled sample input (input.mkv) into output/report.md.
run:
    PYTHONPATH=src {{python}} -m anime2manga input.mkv -o {{output_dir}}

# Remove generated output so a fresh `just run` leaves no stale files behind.
clean:
    rm -rf {{output_dir}}

# Build source and wheel distributions.
build:
    {{python}} -m build

# Publish to PyPI.
publish:
    {{python}} -m twine upload dist/*
