1. Run `pytest` for all tests.
2. Run all tests in a specific file:
```
pytest tests/test_file.py
```
3. Run a specific test function or class and even its methods within a class:
```
pytest tests/test_file.py::test_function
```
4. Run a specific test with verbose output:
```
pytest tests/test_file.py::test_function -v -s
```
5. Run a specific test with verbose output and show local variables:
```
pytest tests/test_file.py::test_function -vv
```
6. Run a specific test with verbose output and show local variables and capture output:
```
pytest tests/test_file.py::test_function -vv -s
```

# About API keys
For test, we dont pass the real API keys. We use the `pytest` library to mock the API keys.

In the action workflow, we use the `secrets` to pass the test API keys.
