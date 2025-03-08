# Code Quality Guidelines

This document defines engineering standards and code quality guidelines for NewsCrawler.

## Engineering Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code style.
- Ensure all functions and classes have clear, structured docstrings following [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
- Code must be fully type-annotated using Python's typing module.
- Enforce a modular architecture for maintainability.
- Apply the DRY (Don't Repeat Yourself) principle.
- Use meaningful variable and function names.
- Keep functions and methods small and focused on a single responsibility.
- Follow the SOLID principles for object-oriented design.

## AI Code Review Workflow

AI agents must self-audit code for:

1. **Readability**:
   - Clear variable and function names
   - Appropriate comments for complex logic
   - Consistent formatting and style

2. **Maintainability**:
   - Modular design with clear separation of concerns
   - Minimal dependencies between components
   - Proper error handling and logging

3. **Scalability**:
   - Efficient algorithms and data structures
   - Asynchronous processing where appropriate
   - Resource management and cleanup

4. **Performance optimizations**:
   - Identify and address potential bottlenecks
   - Minimize redundant operations
   - Optimize database queries and I/O operations

5. **Adherence to project standards**:
   - Consistent coding style
   - Proper use of project-specific patterns
   - Compliance with security guidelines

## Documentation Requirements

- All public APIs must be fully documented with:
  - Function/method purpose
  - Parameter descriptions
  - Return value descriptions
  - Exception information
  - Usage examples

- Complex algorithms must include explanatory comments that describe:
  - The algorithm's purpose
  - Time and space complexity
  - Key steps in the implementation
  - Any assumptions or limitations

- Architecture decisions must be documented with rationale:
  - Problem being solved
  - Alternatives considered
  - Reasons for the chosen approach
  - Trade-offs and limitations

- Code examples should be provided for non-trivial functionality:
  - Basic usage examples
  - Examples for common use cases
  - Examples for handling edge cases

## Testing Standards

- All code must have appropriate unit tests:
  - Test each function/method in isolation
  - Cover both normal and edge cases
  - Mock external dependencies

- Test coverage should be at least 90%:
  - Line coverage
  - Branch coverage
  - Condition coverage

- Integration tests must be provided for component interactions:
  - Test API endpoints
  - Test database interactions
  - Test third-party service integrations

- Edge cases must be explicitly tested:
  - Empty inputs
  - Invalid inputs
  - Resource limitations
  - Error conditions

## Automated Checks

- Run pre-commit hooks to format and lint code:
  - [black](https://black.readthedocs.io/) for code formatting
  - [isort](https://pycqa.github.io/isort/) for import sorting
  - [flake8](https://flake8.pycqa.org/) for linting
  - [mypy](https://mypy.readthedocs.io/) for type checking

- Use automated static analysis tools to catch potential issues:
  - [bandit](https://bandit.readthedocs.io/) for security vulnerabilities
  - [pylint](https://pylint.pycqa.org/) for code quality
  - [pydocstyle](https://www.pydocstyle.org/) for docstring style

- Implement continuous integration to verify code quality:
  - Run all tests on each pull request
  - Enforce code coverage thresholds
  - Block merges that don't meet quality standards

## Language-Specific Guidelines

### Python

- Use Python 3.10+ for all components.
- Leverage type hints for better code documentation and static analysis.
- Use dataclasses or Pydantic models for data structures.
- Prefer composition over inheritance.
- Use context managers for resource management.
- Leverage async/await for I/O-bound operations.
- Use f-strings for string formatting.
- Follow the Zen of Python (PEP 20).

### JavaScript/Node.js (for Puppeteer)

- Use ES6+ syntax.
- Follow the [Airbnb JavaScript Style Guide](https://github.com/airbnb/javascript).
- Use async/await for asynchronous operations.
- Properly handle promises and errors.
- Use TypeScript for type safety.
- Minimize global state.
- Use ESLint for code quality enforcement.

### SQL

- Use consistent naming conventions:
  - Snake case for table and column names
  - Plural for table names
  - Singular for column names
- Write readable and maintainable queries:
  - Use appropriate indentation
  - Break long queries into multiple lines
  - Use meaningful aliases
- Optimize queries for performance:
  - Use appropriate indexes
  - Avoid SELECT *
  - Use JOINs appropriately
- Use parameterized queries to prevent SQL injection.

## NewsCrawler-Specific Guidelines

### Web Scraping Code

- Implement proper rate limiting and politeness:
  - Respect robots.txt
  - Use appropriate delays between requests
  - Implement exponential backoff for retries

- Handle errors gracefully:
  - Connection timeouts
  - HTTP errors
  - Parsing failures
  - JavaScript execution errors

- Implement proper logging:
  - Log all requests and responses
  - Log errors with context
  - Log performance metrics

- Use appropriate User-Agent strings:
  - Identify the crawler honestly
  - Include contact information

### Database Interactions

- Use SQLAlchemy ORM for database operations.
- Implement proper connection pooling.
- Use transactions for multi-step operations.
- Implement proper error handling and rollback.
- Use migrations for schema changes.

### API Development

- Follow RESTful principles.
- Implement proper input validation.
- Use appropriate HTTP status codes.
- Implement proper error handling and responses.
- Document all endpoints with OpenAPI/Swagger.
- Implement rate limiting and authentication where appropriate. 