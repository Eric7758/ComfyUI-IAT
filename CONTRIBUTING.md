# Contributing to ComfyUI-IAT

Thank you for your interest in contributing to ComfyUI-IAT! This document provides guidelines and instructions for contributing.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Submitting Changes](#submitting-changes)
- [Coding Standards](#coding-standards)
- [Testing](#testing)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to:
- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Respect different viewpoints and experiences

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ComfyUI-IAT.git
   cd ComfyUI-IAT
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/Eric7758/ComfyUI-IAT.git
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites
- Python 3.8+
- ComfyUI installed
- Git

### Install Development Dependencies
```bash
pip install -r requirements.txt
pip install pytest black flake8 isort
```

### Pre-commit Hooks (Optional)
```bash
pip install pre-commit
pre-commit install
```

## Making Changes

### Types of Contributions

#### 1. Bug Fixes
- Check existing issues first
- Create a new issue if none exists
- Reference the issue in your PR

#### 2. New Features
- Discuss major features in an issue first
- Keep features focused and modular
- Add documentation for new features

#### 3. Documentation
- Fix typos and unclear instructions
- Add examples and tutorials
- Improve README and docstrings

#### 4. Performance Improvements
- Profile before optimizing
- Document performance gains
- Ensure backward compatibility

### Code Structure

```
ComfyUI-IAT/
├── py/nodes/           # Node implementations
│   ├── qwen35_nodes.py
│   ├── qwen35_runtime.py
│   ├── image_nodes.py
│   └── input_output_nodes.py
├── docs/               # Documentation
├── examples/           # Example workflows
├── tests/              # Unit tests
├── __init__.py         # Plugin entry point
└── config.yaml         # Configuration
```

### Adding a New Node

1. **Create node file** in `py/nodes/`:
   ```python
   class MyNewNode:
       @classmethod
       def INPUT_TYPES(cls):
           return {
               "required": {
                   "input_param": ("STRING", {"default": ""}),
               }
           }
       
       RETURN_TYPES = ("STRING",)
       RETURN_NAMES = ("output",)
       FUNCTION = "process"
       CATEGORY = "IAT/Custom"
       
       def process(self, input_param):
           # Your logic here
           return (result,)
   
   NODE_CLASS_MAPPINGS = {
       "MyNewNode by IAT": MyNewNode,
   }
   ```

2. **Add docstrings** following Google style
3. **Add tests** in `tests/`
4. **Update documentation**
5. **Add example workflow** if applicable

## Submitting Changes

### Before Submitting

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests**:
   ```bash
   pytest tests/
   ```

3. **Check code style**:
   ```bash
   black py/
   flake8 py/
   isort py/
   ```

4. **Update documentation** if needed

### Creating a Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR on GitHub**:
   - Use the PR template
   - Describe what changed and why
   - Reference related issues
   - Add screenshots for UI changes

3. **PR Title Format**:
   - `[BUG]` for bug fixes
   - `[FEAT]` for new features
   - `[DOCS]` for documentation
   - `[REFACTOR]` for code refactoring

### PR Review Process

1. Maintainers will review within a few days
2. Address review comments
3. Keep discussion focused and respectful
4. Squash commits if requested

## Coding Standards

### Python Style Guide

Follow PEP 8 with these specifics:

#### Formatting
- Use **Black** for formatting
- Line length: 100 characters max
- Use double quotes for strings

#### Imports
```python
# Standard library
import os
import sys
from typing import List, Optional

# Third-party
from PIL import Image
import torch

# Local
from .qwen35_runtime import generate_text
```

#### Docstrings
Use Google style docstrings:
```python
def my_function(param1: str, param2: int) -> bool:
    """
    Short description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When input is invalid
    """
```

#### Type Hints
Use type hints for function signatures:
```python
def process_text(text: str, max_length: int = 512) -> str:
    ...
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `Qwen35PromptEnhancerNode`)
- **Functions/Variables**: `snake_case` (e.g., `enhance_prompt`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_VARIANT`)
- **Private**: Prefix with underscore (e.g., `_internal_function`)

### Comments

- Use comments to explain "why", not "what"
- Keep comments up-to-date with code
- Prefer clear code over comments

## Testing

### Writing Tests

```python
import pytest
from py.nodes.qwen35_nodes import QwenTranslatorNode

def test_translator_empty_input():
    node = QwenTranslatorNode()
    result = node.translate("", "Qwen3.5-0.8B", "None", "auto", 512, 0.1, True, 1)
    assert result == ("",)

def test_translator_english_passthrough():
    node = QwenTranslatorNode()
    result = node.translate("hello world", "Qwen3.5-0.8B", "None", "auto", 512, 0.1, True, 1)
    assert result == ("hello world",)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=py/ --cov-report=html

# Run specific test file
pytest tests/test_nodes.py

# Run with verbose output
pytest -v
```

### Test Coverage

Aim for:
- 80%+ coverage for new code
- 100% coverage for critical paths
- Test edge cases and error handling

## Documentation

### README Updates

When adding features:
1. Update feature list
2. Add usage examples
3. Update configuration section if needed

### Code Documentation

- Add module docstrings
- Document all public functions
- Include type hints
- Provide usage examples in docstrings

### Changelog

Update `UPDATE.md` with:
- Version number
- Date
- List of changes
- Breaking changes (if any)

## Questions?

- Open an issue for questions
- Join ComfyUI Discord for discussions
- Check existing issues and PRs first

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

Thank you for contributing to ComfyUI-IAT!
