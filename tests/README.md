# NewsCrawler Tests

This directory contains tests for the NewsCrawler application.

## Test Structure

The tests are organized as follows:

- `conftest.py`: Contains shared test fixtures
- `api/`: Tests for the API endpoints
  - `test_articles.py`: Tests for the articles router
  - `test_search.py`: Tests for the search router
  - `test_llm.py`: Tests for the LLM router
  - `test_health.py`: Tests for the health router

## Running the Tests

You can run the tests using the `run_tests.py` script in the root directory:

```bash
python run_tests.py
```

This will run all tests and generate a coverage report.

### Running Specific Tests

To run a specific test file:

```bash
python run_tests.py tests/api/test_articles.py
```

To run a specific test function:

```bash
python run_tests.py tests/api/test_articles.py::test_get_articles
```

### Coverage Report

The test runner generates a coverage report in HTML format in the `coverage_html` directory. You can open `coverage_html/index.html` in a web browser to view the report.

## Test Database

The tests use an in-memory SQLite database instead of PostgreSQL for faster testing. This means that PostgreSQL-specific features like vector search are mocked in the tests.

## Mocking

The tests use the `unittest.mock` module to mock external dependencies like the LLM client and vector search functionality. This allows us to test the API endpoints without actually calling these external services.

## Adding New Tests

When adding new tests, follow these guidelines:

1. Create a new test file in the appropriate directory
2. Use the shared fixtures from `conftest.py` where possible
3. Mock external dependencies
4. Test both success and error cases
5. Test edge cases and boundary conditions 