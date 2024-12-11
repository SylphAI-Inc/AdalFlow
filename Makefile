# Define variables for common directories and commands
PYTHON = poetry run
SRC_DIR = .

# Default target: Show help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  setup            Install dependencies and set up pre-commit hooks"
	@echo "  format           Run Black and Ruff to format the code"
	@echo "  lint             Run Ruff to check code quality"
	@echo "  test             Run tests with pytest"
	@echo "  precommit        Run pre-commit hooks on all files"
	@echo "  clean            Clean up temporary files and build artifacts"

# Install dependencies and set up pre-commit hooks
.PHONY: setup
setup:
	poetry install
	poetry run pre-commit install

# Format code using Black and Ruff
.PHONY: format
format:
	$(PYTHON) black $(SRC_DIR) --config pyproject.toml
	$(PYTHON) ruff check --fix $(SRC_DIR)
# remove git ls-files | xargs pre-commit run black --files, causes a circular dependency

# Run lint checks using Ruff
.PHONY: lint
lint:
	$(PYTHON) black --check $(SRC_DIR) --config pyproject.toml
	$(PYTHON) ruff check $(SRC_DIR)

# Run all pre-commit hooks on all files
.PHONY: precommit
precommit:
	$(PYTHON) pre-commit run --all-files

# Run tests
.PHONY: test
test:
	$(PYTHON) pytest

# Clean up temporary files and build artifacts
.PHONY: clean
clean:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf __pycache__
	rm -rf build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
