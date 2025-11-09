# Contributing to live-audio-classifier

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/dmanzato/live-audio-classifier.git`
3. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Install in development mode: `pip install -e .`

## Development Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to public functions and classes
- Keep functions focused and modular

### Testing
- Add tests for new features
- Ensure existing tests pass: `python -m pytest tests/` (when tests are added)
- Test on different platforms if possible (macOS, Linux, Windows)

### Pull Requests
1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Test thoroughly
4. Update documentation if needed
5. Submit a pull request with a clear description

### Commit Messages
- Use clear, descriptive commit messages
- Reference issues when applicable: `Fix #123: description`

## Areas for Contribution

- Additional model architectures
- Data augmentation techniques
- Performance optimizations
- Documentation improvements
- Bug fixes
- Test coverage
- Example notebooks

## Questions?

Open an issue for discussion or questions about contributing.

