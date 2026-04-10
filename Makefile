# Install production
.PHONY: install
install:
	pip install -e .

# Install development
.PHONY: install-dev
install-dev:
	pip install -e .[dev,gym]

# Build
.PHONY: build
build:
	pip-compile --resolver=backtracking --output-file=requirements.txt pyproject.toml
	pip-compile --resolver=backtracking --extra=dev --output-file=requirements-dev.txt pyproject.toml

# Build docs
# NOTE: May require `rm -rf ./docs/source/api` even with --remove-old, but doesn't work on Windows when folder doesn't exist
.PHONY: build-docs
build-docs:
	sphinx-apidoc --separate --remove-old -t ./docs/source/_templates -o ./docs/source/api ./fishyrl
	make -C ./docs html

# Run tests
.PHONY: test 
test:
	pytest -v

# Lint
.PHONY: lint
lint:
	ruff check

# Do everything needed for commit
.PHONY: pre-commit
pre-commit: build build-docs lint test

# Help command
.DEFAULT_GOAL := help
# See <https://gist.github.com/klmr/575725c7e05d8780505a> for explanation.
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr-1)";echo;sed -ne"/^## /{h;s/.*//;:d" -e"H;n;s/^## //;td" -e"s/:.*//;G;s/\\n## /---/;s/\\n/ /g;p;}" ${MAKEFILE_LIST}|LC_ALL='C' sort -f|awk -F --- -v n=$$(tput cols) -v i=19 -v a="$$(tput setaf 6)" -v z="$$(tput sgr0)" '{printf"%s%*s%s ",a,-i,$$1,z;m=split($$2,w," ");l=n-i;for(j=1;j<=m;j++){l-=length(w[j])+1;if(l<= 0){l=n-i-length(w[j])-1;printf"\n%*s ",-i," ";}printf"%s ",w[j];}printf"\n";}'|more $(shell test $(shell uname) == Darwin && echo '-Xr')
