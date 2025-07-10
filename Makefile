.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync
	@echo "🚀 Installing package in editable mode"
	@uv pip install -e .
	@uv run pre-commit install

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Static type checking: Running mypy"
	@uv run mypy
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	@uv run deptry .

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --doctest-modules

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: test-install
test-install: build
	@echo "--- Creating temporary uv virtual environment for testing installation ---"
	@rm -rf tmp_test_env
	@uv venv tmp_test_env --seed
	@echo "--- Installing wheel into temporary environment using uv ---"
	@uv pip install --python ./tmp_test_env/bin/python dist/*.whl
	@echo "--- Verifying installation by running ffsplat-view --help ---"
	@./tmp_test_env/bin/ffsplat-view --help
	@echo "--- Cleaning up temporary virtual environment ---"
	@rm -rf tmp_test_env
	@echo "--- Installation test successful ---"

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: clean-all
clean-all: clean-build ## Clean everything - build artifacts, cache files, venv, and lock file
	@echo "🚀 Removing all build and runtime artifacts"
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@rm -rf .venv
	@rm -f uv.lock
	@echo "🧹 Cleaned all artifacts"

.PHONY: clean-install
clean-install: clean-all install ## Clean everything and reinstall
	@echo "✨ Fresh install completed"

.PHONY: publish
publish: ## Publish a release to PyPI.
	@echo "🚀 Publishing."
	@uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish.

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@uv run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@uv run mkdocs serve

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
