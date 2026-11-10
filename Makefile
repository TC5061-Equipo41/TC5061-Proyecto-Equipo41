#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mibici
PYTHON_VERSION = 3.14
PYTHON_INTERPRETER = python
VALIDATION_VENV ?= $(HOME)/.venvs/mibici-validation

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Correr todas las pruebas (unitarias, reproducibilidad e higiene del repo)
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	

## Seguridad: bandit (código), pip-audit (dependencias) y detect-secrets (credenciales)
.PHONY: security
security:
	@mkdir -p reports/security-scan
	@echo ">>> bandit: análisis estático del código"
	$(PYTHON_INTERPRETER) -m bandit -c pyproject.toml -r mibici
	$(PYTHON_INTERPRETER) -m bandit -q -c pyproject.toml -r mibici -f html -o reports/security-scan/bandit.html
	@echo ">>> pip-audit: vulnerabilidades conocidas en las dependencias"
	$(PYTHON_INTERPRETER) -m pip_audit -r requirements.txt
	@echo ">>> detect-secrets: credenciales en archivos versionados"
	git ls-files -z | xargs -0 $(PYTHON_INTERPRETER) -m detect_secrets.pre_commit_hook --baseline .secrets.baseline
	@echo ">>> Seguridad OK"

## Crear el entorno de validación de datos (Python 3.12 + DeepChecks)
.PHONY: validation-env
validation-env:
	uv venv --python 3.12 "$(VALIDATION_VENV)"
	VIRTUAL_ENV="$(VALIDATION_VENV)" uv pip install -r requirements-validation.txt

## Validar los datos de P1, P2 y P3 con DeepChecks (reportes en reports/validation/)
.PHONY: validate
validate:
	$(PYTHON_INTERPRETER) -m mibici.validation.prepare_inputs
	PYTHONPATH=. "$(VALIDATION_VENV)/bin/python" -m mibici.validation.validate_p3_data

## Drift de datos con Evidently: entrenamiento (REFERENCE) contra producción (CURRENT)
## Reportes en reports/drift/. Ejemplo: make drift CURRENT=2026_10:2026_10
REFERENCE ?= 2025_10:2026_09
CURRENT ?= 2026_10:2026_10
.PHONY: drift
drift:
	$(PYTHON_INTERPRETER) -m mibici.monitoring.drift --reference $(REFERENCE) --current $(CURRENT)

## Pruebas + seguridad + validación de datos
.PHONY: check
check: test security validate

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) mibici/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
