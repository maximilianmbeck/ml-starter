# Makefile

define HELP_MESSAGE
                    ML Template
                    -----------

# Installing

1. Create a new Conda environment: `conda create --name ml-project python=3.10`
2. Activate the environment: `conda activate ml-project`
3. Install the package: `make install-dev`
4. Rebuild C++ extensions: `make build-ext`

# Running Tests

1. Run autoformatting: `make format`
2. Run static checks: `make static-checks`
3. Run unit tests: `make test`

endef
export HELP_MESSAGE

all:
	@echo "$$HELP_MESSAGE"
.PHONY: all

# ------------------------ #
#          Build           #
# ------------------------ #

install-torch-nightly:
	@pip install --pre torch --index-url https://download.pytorch.org/whl/nightly
.PHONY: install-torch-nightly

install:
	@pip install --verbose -e .
.PHONY: install

install-dev:
	@pip install --verbose -e '.[dev]'
.PHONY: install

build-ext:
	@python setup.py build_ext --inplace
.PHONY: build-ext

clean:
	rm -rf build dist *.so **/*.so **/*.pyi **/*.pyc **/*.pyd **/*.pyo **/__pycache__ *.egg-info .eggs/ .ruff_cache/
.PHONY: clean

# ------------------------ #
#       Static Checks      #
# ------------------------ #

py-files := $$(git ls-files '*.py')
cpp-files := $$(git ls-files '*.c' '*.cpp' '*.h' '*.hpp' '*.cu' '*.cuh')
cmake-files := $$(git ls-files '*/CMakeLists.txt')

format:
	black $(py-files)
	ruff --fix $(py-files)
	cmake-format -i $(cmake-files)
	clang-format -i $(cpp-files)
.PHONY: format

static-checks:
	black --diff --check $(py-files)
	ruff $(py-files)
	mypy --install-types --non-interactive $(py-files)
	darglint $(py-files)
	cmake-format --check $(cmake-files) > /dev/null
	clang-format --dry-run --Werror $(cpp-files) > /dev/null
.PHONY: lint

mypy-daemon:
	dmypy run -- $(py-files)
.PHONY: mypy-daemon

# ------------------------ #
#        Unit tests        #
# ------------------------ #

test:
	python -m pytest
.PHONY: test
