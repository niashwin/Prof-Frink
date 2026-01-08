# Contributing to FRINK

Thank you for your interest in contributing to FRINK! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)
- [Documentation](#documentation)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and considerate
- Use welcoming and inclusive language
- Be patient with newcomers
- Focus on what is best for the community

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/frink.git
   cd frink
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/niashwin/homer.git
   ```

## Development Setup

### Prerequisites

- Python 3.12 or higher
- Git
- Claude Code CLI (for plugin development)

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks (if available):
   ```bash
   pre-commit install
   ```

### Verify Installation

Run the test suite to ensure everything is set up correctly:
```bash
pytest tests/
```

## Making Changes

### Branch Naming Convention

Create a feature branch from `main`:

```
frink/{feature-name}      # New features
frink/fix/{issue-name}    # Bug fixes
frink/docs/{topic}        # Documentation updates
frink/test/{component}    # Test additions
```

Example:
```bash
git checkout -b frink/add-visualization-gate
```

### Commit Messages

Use clear, descriptive commit messages following this format:

```
<type>(<scope>): <short description>

<optional body>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding/updating tests
- `refactor`: Code restructuring (no behavior change)
- `style`: Formatting, linting
- `chore`: Build, CI, dependencies

Examples:
```bash
git commit -m "feat(literature): add PRISMA diagram generator"
git commit -m "fix(quality-gates): correct empty dataset validation"
git commit -m "docs: update installation instructions"
```

## Testing

### Running Tests

Run all tests:
```bash
pytest tests/
```

Run unit tests only:
```bash
pytest tests/unit/
```

Run E2E tests only:
```bash
pytest tests/e2e/
```

Run with coverage:
```bash
pytest --cov=lib --cov-report=html tests/
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place E2E tests in `tests/e2e/`
- Use descriptive test names: `test_{function}_{scenario}_{expected_outcome}`
- Include docstrings explaining what each test validates

Example:
```python
def test_literature_gate_fails_with_insufficient_papers():
    """Test that literature gate fails when fewer than 20 papers are retrieved."""
    # Test implementation
```

### Test Requirements

Before submitting a PR:

1. All existing tests must pass
2. New features should have accompanying tests
3. Bug fixes should have regression tests
4. Aim for >85% code coverage

## Submitting Changes

### Pull Request Process

1. Update your branch with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Push your branch:
   ```bash
   git push origin frink/your-feature
   ```

3. Create a Pull Request on GitHub

4. Fill in the PR template with:
   - Summary of changes
   - Type of change
   - Testing performed
   - Documentation updates

### PR Review Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated if needed
- [ ] No merge conflicts
- [ ] Self-review completed

## Code Style

### Python Style

- **Formatter**: Black (line-length=100)
- **Linter**: Ruff
- **Type Hints**: Required for all public functions
- **Docstrings**: Google style

Run formatters:
```bash
black lib/ tests/
ruff check lib/ tests/
```

Run type checking:
```bash
mypy lib/
```

### Example Code Style

```python
def execute_story(
    self,
    story: UserStory,
    context: dict[str, Any]
) -> StoryResult:
    """Execute a single user story from the research PRD.

    Args:
        story: The UserStory to execute
        context: Execution context including project_dir and progress

    Returns:
        StoryResult indicating success or failure

    Raises:
        SkillNotFoundError: If required skill is not available
    """
    # Implementation
    pass
```

## Documentation

### Code Documentation

- All public functions must have docstrings
- Use Google-style docstrings
- Include type hints for all parameters and return values
- Document exceptions that may be raised

### Project Documentation

- Update README.md for user-facing changes
- Update CHANGELOG.md for all notable changes
- Add inline comments for complex logic

## Architecture Overview

Understanding the key components helps with contributing:

```
lib/
├── schemas.py           # Pydantic models for topic and PRD
├── prd_generator.py     # PRD generation with 32 user stories
├── quality_gates.py     # Quality validation gates
├── research_loop.py     # Main autonomous loop
└── db/
    ├── schema.sql       # SQLite schema
    └── manager.py       # Database operations
```

### Key Concepts

- **Research Topic**: Input specification (hypothesis, datasets)
- **PRD (Product Requirements Document)**: 32 user stories across 8 stages
- **Quality Gates**: Validation checkpoints between stages
- **Research Loop**: Orchestrates story execution

## Getting Help

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Join discussions in GitHub Discussions

## Recognition

Contributors will be recognized in:
- CHANGELOG.md for significant contributions
- README.md contributors section

Thank you for contributing to FRINK!
