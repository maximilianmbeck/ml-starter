# ML Project Template

This is a general-purpose template for machine learning projects in PyTorch. It includes a simple MNIST example which can be deleted.

## Getting Started

First, install the project:

```bash

```

## Architecture

A new project is broken down into five parts:

1. *Task*: Defines the dataset and calls the model on a sample. Similar to a [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html).
2. *Model*: Just a PyTorch `nn.Module`
3. *Trainer*: Defines the main training loop, and optionally how to distribute training when using multiple GPUs
4. *Optimizer*: Just a PyTorch `optim.Optimizer`
5. *LR Scheduler*: Just a PyTorch `optim.LRScheduler`

Most projects should just have to implement the Task and Model, and use a default trainer, optimizer

## Features

This repository implements some features which I find useful when starting ML projects.

### C++ Extensions

This template makes it easy to add custom C++ extensions to your PyTorch project. The demo includes a custom TorchScript-compatible nucleus sampling function, although more complex extensions are possible.

- [Custom TorchScript Op Tutorial](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html)
- [PyTorch CMake Extension Reference](https://github.com/pytorch/extension-script)

### Github Actions

This template automatically runs `black`, `isort`, `pylint` and `mypy` against your repository as a Github action. You can enable push-blocking until these tests pass.

## Design Philosophy

With most ML frameworks, there's usually a question like:

> How much is this framework "barebones" verses adding additional utility (which might weigh it down)?

The answer for this framework is, "however much I feel like". I built this for myself and my own projects and I am the main customer, but I hope other people might find it useful as well.
