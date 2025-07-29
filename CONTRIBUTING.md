# Contributing to Music Transition Transformer

Thank you for your interest in contributing to this project! Here are some guidelines to help you get started.

## Development Setup

1. Fork the repository
2. Clone your fork locally
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep lines under 88 characters when possible

## Testing

Run the test suite before submitting any changes:
```bash
python test_music_transformer.py
```

## Submitting Changes

1. Create a new branch for your feature/fix
2. Make your changes
3. Test your changes thoroughly
4. Commit with a clear, descriptive message
5. Push to your fork
6. Create a pull request

## Reporting Issues

When reporting issues, please include:
- Python version
- PyTorch version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages (if any)
