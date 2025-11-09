# Tests

This directory contains unit tests for the live-audio-classifier project.

## Running Tests

### Install test dependencies

```bash
pip install -r requirements.txt
```

### Run all tests

```bash
pytest tests/
```

### Run with coverage

```bash
pytest tests/ --cov=. --cov-report=html
```

### Run specific test file

```bash
pytest tests/test_models.py
pytest tests/test_transforms.py
```

### Run with verbose output

```bash
pytest tests/ -v
```

## Test Structure

- `test_models.py`: Tests for model architectures (SmallCNN)
- `test_transforms.py`: Tests for audio transforms (mel spectrogram, SpecAugment)
- `test_dataset.py`: Tests for dataset loading (UrbanSound8K)
- `test_utils.py`: Tests for utility functions
- `conftest.py`: Pytest fixtures and configuration

## Writing New Tests

When adding new features, add corresponding tests:

1. Create a test file or add to existing one
2. Use descriptive test function names starting with `test_`
3. Use fixtures from `conftest.py` when appropriate
4. Test both happy path and edge cases
5. Run tests before committing

## Example Test

```python
def test_my_function():
    """Test description."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = my_function(input_data)
    
    # Assert
    assert result is not None
    assert result.shape == expected_shape
```

