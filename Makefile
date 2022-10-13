# Makefile

all: install
.PHONY: all

# ---------------
# Specify backend
# ---------------

# Detects backend.
ifdef CUDA_HOME
BACKEND := cuda
else ifeq ($(shell uname -p), arm)
BACKEND := metal
else
BACKEND := cpu
endif

# Gets the Torch version to install.
TORCH_VERSION_BASE := 1.12.1
TORCHVISION_VERSION_BASE := 0.13.1
CUDA_VERSION := cu116
ifeq ($(BACKEND),cuda)
TORCH_VERSION := $(TORCH_VERSION_BASE)+$(CUDA_VERSION)
TORCHVISION_VERSION := $(TORCHVISION_VERSION_BASE)+$(CUDA_VERSION)
else ifeq ($(BACKEND),metal)
TORCH_VERSION := $(TORCH_VERSION_BASE)
TORCHVISION_VERSION := $(TORCHVISION_VERSION_BASE)
else ifeq ($(BACKEND),cpu)
TORCH_VERSION := $(TORCH_VERSION_BASE)+cpu
TORCHVISION_VERSION := $(TORCHVISION_VERSION_BASE)+cpu
else
$(error "Unsupported backend: $(BACKEND)")
endif

# --------------
# Build commands
# --------------

install-conda:
	conda install -c sarcasm -c conda-forge --file requirements/conda.txt
.PHONY: install-conda

install-pip:
	pip install --find-links https://download.pytorch.org/whl/torch_stable.html torch==$(TORCH_VERSION) torchvision==$(TORCHVISION_VERSION)
	pip install -r requirements/pip.txt
.PHONY: install-pip

install-dependencies: install-conda install-pip
.PHONY: install-dependencies

install:
	TORCH_VERSION=$(TORCH_VERSION) pip install --find-links https://download.pytorch.org/whl/torch_stable.html -e '.[dev]'
.PHONY: install

clean:
	rm -rf build/ **/*.egg-info **/*.pyc **/*.so ml/**/*.pyi ml/**/*.so
.PHONY: clean

# ---------------
# Static analysis
# ---------------

py-files := $$(git ls-files '*.py')
cpp-files := $$(git ls-files '*.c' '*.cpp' '*.h' '*.hpp' '*.cu' '*.cuh')
cmake-files := $$(git ls-files '*/CMakeLists.txt')

format:
	cmake-format -i $(cmake-files)
	clang-format -i $(cpp-files)
	black $(py-files)
	isort $(py-files)
.PHONY: format

lint:
	black --diff --check $(py-files)
	isort --check-only $(py-files)
	mypy $(py-files)
	flake8 --count --show-source --statistics $(py-files)
	pylint $(py-files)
.PHONY: lint

# ----------
# Unit tests
# ----------

test:
	pytest .
.PHONY: test
