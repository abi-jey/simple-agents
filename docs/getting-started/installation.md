# Installation

## Requirements

- Python 3.11 or higher
- An API key from OpenAI, Anthropic, or Google

## Install from PyPI

```bash
pip install simple-agents
```

## Install with Development Dependencies

```bash
pip install simple-agents[dev]
```

This includes:

- pytest, pytest-asyncio, pytest-cov for testing
- ruff for linting and formatting
- mypy for type checking
- pre-commit for git hooks

## Install with Documentation Dependencies

```bash
pip install simple-agents[docs]
```

## Install from Source

```bash
git clone https://github.com/abi-jey/simple-agents.git
cd simple-agents
pip install -e ".[dev]"
```

## Verify Installation

```python
from simple_agents import Agent, Provider, ProviderType

print("simple-agents installed successfully!")
```
