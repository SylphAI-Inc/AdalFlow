
#Setup
Install the required packages:
```
poetry install --with test
``` 

1. Run `pytest` for all tests.
2. Run all tests in a specific file:
```
pytest tests/test_file.py
```
1. Run a specific test function or class and even its methods within a class:
```
pytest tests/test_file.py::test_function
```
1. Run a specific test with verbose output:
```
pytest tests/test_file.py::test_function -v -s
```
1. Run a specific test with verbose output and show local variables:
```
pytest tests/test_file.py::test_function -vv
```
1. Run a specific test with verbose output and show local variables and capture output:
```
pytest tests/test_file.py::test_function -vv -s
```

# About API keys
For test, we dont pass the real API keys. We use the `pytest` library to mock the API keys.

In the action workflow, we use the `secrets` to pass the test API keys.
