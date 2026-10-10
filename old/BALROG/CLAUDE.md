# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BALROG (Benchmarking Agentic LLM and VLM Reasoning On Games) is a benchmark for evaluating LLMs and VLMs on long-horizon interactive tasks using reinforcement learning environments. The system supports multiple game environments and various agent architectures.

## Essential Commands

```bash
# Setup development environment
conda create -n balrog python=3.10 -y
conda activate balrog
pip install -e ".[dev]"
balrog-post-install
pre-commit install

# Run evaluation
python eval.py agent.type=<agent_type> client.client_name=<client> client.model_id=<model>

# Run specific tests
pytest balrog/tests/test_evaluation.py -k "test_name"
pytest balrog/tests/test_evaluation.py::TestEvaluation::test_specific

# Code quality checks
black balrog/ --line-length 120
isort balrog/ --profile black
flake8 balrog/
pylint balrog/

# Build package
python -m build
```

## Architecture

### Core Components

1. **Agent System** (`balrog/agents/`):
   - Base class: `BaseAgent` - all agents inherit from this
   - Factory: `AgentFactory.create()` - creates agent instances
   - Key agents: `naive`, `chain_of_thought`, `robust_naive`, `robust_cot`, `few_shot`

2. **Environment Wrappers** (`balrog/environments/`):
   - Each environment has its own wrapper inheriting from `BaseEnv`
   - Wrappers handle observation/action translation between raw environment and agent
   - Supported: NLE, MiniHack, BabyAI, Crafter, TextWorld, BabaIsAI

3. **LLM Clients** (`balrog/client.py`):
   - All clients implement common interface
   - Factory pattern: `APIClient.create()`
   - Handles retries, rate limiting, and error handling

4. **Configuration** (`balrog/config/`):
   - Uses Hydra for configuration management
   - Main config: `balrog/config/config.yaml`
   - Override with command line: `key=value` syntax

5. **Evaluation Pipeline** (`balrog/evaluator.py`):
   - `Evaluator` class orchestrates the evaluation
   - Handles parallel execution with multiprocessing
   - Manages trajectory saving and result aggregation

### Key Design Patterns

- **Factory Pattern**: Used for creating agents and clients
- **Wrapper Pattern**: Environment wrappers provide consistent interface
- **Base Classes**: Abstract base classes ensure consistent interfaces
- **Configuration-Driven**: Hydra allows flexible configuration without code changes

### Adding New Components

**New Agent**:
1. Create class in `balrog/agents/` inheriting from `BaseAgent`
2. Implement `act()` method
3. Register in `AgentFactory._agent_registry`

**New Environment**:
1. Create wrapper in `balrog/environments/` inheriting from `BaseEnv`
2. Implement required methods: `reset()`, `step()`, `render()`, etc.
3. Add configuration in `balrog/config/config.yaml`

**New LLM Client**:
1. Add client class in `balrog/client.py`
2. Implement query interface with retry logic
3. Register in `APIClient.create()` method

## Testing Strategy

- Unit tests use pytest with parameterized testing
- Test files mirror source structure in `balrog/tests/`
- Use `DummyClient` for testing without API calls
- Environment tests use fixed seeds for reproducibility

## Important Notes

- The codebase uses black formatting with 120 character line length
- Import order is enforced by isort with black profile
- Pre-commit hooks run automatically on commit
- Hydra configuration allows runtime overrides without modifying files
- Trajectory files are saved in JSON format for analysis
- The `balrog-post-install` command sets up required game files