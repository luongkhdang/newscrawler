# Error Handling and Retry Mechanisms for Newspaper4k

## Overview
This document analyzes the current error handling capabilities of Newspaper4k and proposes enhanced retry mechanisms to improve the robustness of the NewsCrawler system. Effective error handling and retry strategies are essential for maintaining high scraping success rates in production environments.

## Current Error Handling in Newspaper4k

### Built-in Error Handling

Newspaper4k provides basic error handling, primarily through exceptions:

1. **Download Exceptions**:
   - `ArticleException`: Base exception for article-related errors
   - `ArticleDownloadException`: Raised when article download fails
   - Network timeouts and connection errors from the underlying requests library

2. **Parsing Exceptions**:
   - `ArticleParseException`: Raised when article parsing fails
   - Various exceptions from underlying libraries (e.g., lxml)

3. **Configuration Options**:
   ```python
   config = newspaper.Config()
   config.request_timeout = 10  # Timeout for HTTP requests in seconds
   config.browser_user_agent = 'Custom User Agent'  # To avoid being blocked
   ```

### Limitations of Built-in Error Handling

1. **No Automatic Retries**: Newspaper4k does not automatically retry failed requests
2. **Limited Timeout Configuration**: Only a single timeout parameter for all requests
3. **Basic Exception Hierarchy**: Limited differentiation between error types
4. **No Backoff Strategy**: No exponential backoff or jitter for retries
5. **No Circuit Breaking**: No mechanism to temporarily stop requests to problematic domains

## Current Implementation in NewsCrawler

Our current implementation in `src/scrapers/newspaper_scraper/newspaper_scraper.py` includes basic error handling:

```python
def download_article(self, url):
    """Download an article with error handling."""
    try:
        article = Article(url, config=self.config)
        article.download()
        return article
    except Exception as e:
        logger.error(f"Error downloading article from {url}: {str(e)}")
        raise ArticleDownloadException(f"Failed to download article: {str(e)}")
```

## Enhanced Error Handling and Retry Strategy

### 1. Comprehensive Exception Hierarchy

Create a more detailed exception hierarchy to better handle different error types:

```python
class ScraperException(Exception):
    """Base exception for all scraper-related errors."""
    pass

class NetworkException(ScraperException):
    """Exception for network-related errors."""
    pass

class TimeoutException(NetworkException):
    """Exception for timeout errors."""
    pass

class RateLimitException(NetworkException):
    """Exception for rate limiting or throttling."""
    pass

class ParsingException(ScraperException):
    """Exception for content parsing errors."""
    pass

class PaywallException(ScraperException):
    """Exception for paywall detection."""
    pass

class RobotsException(ScraperException):
    """Exception for robots.txt violations."""
    pass
```

### 2. Retry Mechanism with Exponential Backoff

Implement a robust retry mechanism with exponential backoff and jitter:

```python
def download_with_retry(self, url, max_retries=3, base_delay=1, max_delay=60):
    """
    Download an article with retry logic and exponential backoff.
    
    Args:
        url: The URL to download
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        
    Returns:
        Downloaded article object
        
    Raises:
        ArticleDownloadException: If all retry attempts fail
    """
    retries = 0
    last_exception = None
    
    while retries <= max_retries:
        try:
            # Check domain health before attempting
            if self.circuit_breaker.is_open(get_domain(url)):
                raise CircuitOpenException(f"Circuit breaker open for {get_domain(url)}")
                
            # Wait according to rate limiter
            domain = get_domain(url)
            self.rate_limiter.wait(domain)
            
            # Attempt to download
            article = Article(url, config=self.config)
            article.download()
            
            # Record successful download
            self.circuit_breaker.record_success(domain)
            return article
            
        except (requests.Timeout, TimeoutException) as e:
            last_exception = TimeoutException(f"Timeout downloading {url}: {str(e)}")
            # Longer backoff for timeouts
            delay_factor = 2
            
        except (requests.ConnectionError, requests.HTTPError) as e:
            status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            
            if status_code == 429:
                last_exception = RateLimitException(f"Rate limited on {url}: {str(e)}")
                # Much longer backoff for rate limiting
                delay_factor = 4
                # Update rate limiter
                self.rate_limiter.increase_delay(domain)
                
            elif status_code and 500 <= status_code < 600:
                last_exception = NetworkException(f"Server error on {url}: {str(e)}")
                delay_factor = 1.5
                
            else:
                last_exception = NetworkException(f"Connection error on {url}: {str(e)}")
                delay_factor = 1
                
        except Exception as e:
            last_exception = ScraperException(f"Error downloading {url}: {str(e)}")
            delay_factor = 1
            
        # Record failure in circuit breaker
        self.circuit_breaker.record_failure(domain)
        
        # Calculate backoff with jitter
        if retries < max_retries:
            delay = min(max_delay, base_delay * (delay_factor ** retries))
            jitter = random.uniform(0, 0.1 * delay)  # 10% jitter
            time.sleep(delay + jitter)
            
        retries += 1
        
    # If we get here, all retries failed
    raise ArticleDownloadException(f"Failed to download article after {max_retries} retries: {str(last_exception)}")
```

### 3. Circuit Breaker Pattern

Implement a circuit breaker to temporarily stop requests to problematic domains:

```python
class CircuitBreaker:
    """
    Circuit breaker to prevent repeated requests to failing domains.
    
    The circuit breaker has three states:
    - CLOSED: Requests are allowed to proceed
    - OPEN: Requests are blocked (circuit is "tripped")
    - HALF-OPEN: A limited number of test requests are allowed to check if the domain is healthy
    """
    
    CLOSED = 'closed'
    OPEN = 'open'
    HALF_OPEN = 'half-open'
    
    def __init__(self, failure_threshold=5, recovery_timeout=300, test_requests=2):
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of consecutive failures before opening the circuit
            recovery_timeout: Time in seconds before attempting recovery (half-open state)
            test_requests: Number of successful test requests needed to close the circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.test_requests = test_requests
        
        self.domains = {}  # Domain state tracking
        self.lock = threading.Lock()
        
    def is_open(self, domain):
        """Check if the circuit is open for a domain."""
        with self.lock:
            if domain not in self.domains:
                self.domains[domain] = {
                    'state': self.CLOSED,
                    'failures': 0,
                    'last_failure_time': 0,
                    'test_successes': 0
                }
                return False
                
            domain_state = self.domains[domain]
            
            # If circuit is open, check if recovery timeout has elapsed
            if domain_state['state'] == self.OPEN:
                if time.time() - domain_state['last_failure_time'] > self.recovery_timeout:
                    # Move to half-open state
                    domain_state['state'] = self.HALF_OPEN
                    domain_state['test_successes'] = 0
                    return False
                return True
                
            # If half-open, allow limited test requests
            if domain_state['state'] == self.HALF_OPEN:
                return False
                
            # Circuit is closed
            return False
            
    def record_success(self, domain):
        """Record a successful request to a domain."""
        with self.lock:
            if domain not in self.domains:
                return
                
            domain_state = self.domains[domain]
            
            if domain_state['state'] == self.CLOSED:
                # Reset failure count on success in closed state
                domain_state['failures'] = 0
                
            elif domain_state['state'] == self.HALF_OPEN:
                # Count successful test requests in half-open state
                domain_state['test_successes'] += 1
                
                # If enough test requests succeed, close the circuit
                if domain_state['test_successes'] >= self.test_requests:
                    domain_state['state'] = self.CLOSED
                    domain_state['failures'] = 0
                    
    def record_failure(self, domain):
        """Record a failed request to a domain."""
        with self.lock:
            if domain not in self.domains:
                self.domains[domain] = {
                    'state': self.CLOSED,
                    'failures': 1,
                    'last_failure_time': time.time(),
                    'test_successes': 0
                }
                return
                
            domain_state = self.domains[domain]
            domain_state['last_failure_time'] = time.time()
            
            if domain_state['state'] == self.CLOSED:
                # Increment failure count in closed state
                domain_state['failures'] += 1
                
                # If failure threshold reached, open the circuit
                if domain_state['failures'] >= self.failure_threshold:
                    domain_state['state'] = self.OPEN
                    
            elif domain_state['state'] == self.HALF_OPEN:
                # Any failure in half-open state reopens the circuit
                domain_state['state'] = self.OPEN
                domain_state['test_successes'] = 0
```

### 4. Error Classification and Logging

Implement detailed error classification and logging:

```python
def classify_error(self, exception, url):
    """
    Classify an exception for better error handling and reporting.
    
    Args:
        exception: The exception to classify
        url: The URL that caused the exception
        
    Returns:
        dict: Classification details including type, severity, and recommended action
    """
    domain = get_domain(url)
    classification = {
        'url': url,
        'domain': domain,
        'exception_type': type(exception).__name__,
        'message': str(exception),
        'timestamp': datetime.now().isoformat(),
        'severity': 'medium',  # Default severity
        'recommended_action': None
    }
    
    # Classify by exception type
    if isinstance(exception, TimeoutException):
        classification['error_type'] = 'timeout'
        classification['severity'] = 'low'
        classification['recommended_action'] = 'increase_timeout'
        
    elif isinstance(exception, RateLimitException):
        classification['error_type'] = 'rate_limit'
        classification['severity'] = 'high'
        classification['recommended_action'] = 'increase_delay'
        
    elif isinstance(exception, PaywallException):
        classification['error_type'] = 'paywall'
        classification['severity'] = 'medium'
        classification['recommended_action'] = 'skip_domain'
        
    elif isinstance(exception, RobotsException):
        classification['error_type'] = 'robots_violation'
        classification['severity'] = 'high'
        classification['recommended_action'] = 'respect_robots'
        
    elif isinstance(exception, ParsingException):
        classification['error_type'] = 'parsing_error'
        classification['severity'] = 'medium'
        classification['recommended_action'] = 'update_parser'
        
    elif isinstance(exception, NetworkException):
        classification['error_type'] = 'network_error'
        classification['severity'] = 'medium'
        classification['recommended_action'] = 'retry'
        
    else:
        classification['error_type'] = 'unknown'
        classification['severity'] = 'medium'
        classification['recommended_action'] = 'investigate'
        
    # Log the classified error
    log_method = getattr(logger, classification['severity'], logger.error)
    log_method(f"Error scraping {url}: {classification['error_type']} - {str(exception)}")
    
    # Store error for analysis
    self._store_error(classification)
    
    return classification
    
def _store_error(self, classification):
    """Store error classification for analysis and reporting."""
    # Append to in-memory error log
    self.error_log.append(classification)
    
    # If we have a database connection, store in database
    if hasattr(self, 'db_session') and self.db_session:
        try:
            error_record = ScraperError(
                url=classification['url'],
                domain=classification['domain'],
                error_type=classification['error_type'],
                exception_type=classification['exception_type'],
                message=classification['message'],
                severity=classification['severity'],
                timestamp=datetime.now()
            )
            self.db_session.add(error_record)
            self.db_session.commit()
        except Exception as e:
            logger.error(f"Failed to store error in database: {str(e)}")
```

## Testing Strategy

### 1. Unit Tests

Create comprehensive unit tests for error handling and retry mechanisms:

```python
def test_retry_mechanism():
    """Test the retry mechanism with different error types."""
    scraper = NewspaperScraper()
    
    # Mock requests to simulate different error scenarios
    with patch('newspaper.Article.download') as mock_download:
        # Test timeout error
        mock_download.side_effect = [
            requests.Timeout("Connection timed out"),
            requests.Timeout("Connection timed out"),
            None  # Success on third try
        ]
        article = scraper.download_with_retry("https://example.com/article")
        assert mock_download.call_count == 3
        
        # Test rate limit error
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_download.side_effect = [
            requests.HTTPError("Too Many Requests", response=mock_response),
            None  # Success on second try
        ]
        article = scraper.download_with_retry("https://example.com/article")
        assert mock_download.call_count == 2
        
        # Test max retries exceeded
        mock_download.side_effect = requests.ConnectionError("Connection refused")
        with pytest.raises(ArticleDownloadException):
            scraper.download_with_retry("https://example.com/article", max_retries=3)
        assert mock_download.call_count == 4  # Initial + 3 retries
```

### 2. Integration Tests

Test the error handling in real-world scenarios:

```python
def test_real_world_error_handling():
    """Test error handling with real websites."""
    scraper = NewspaperScraper()
    
    # Test with non-existent domain
    with pytest.raises(ArticleDownloadException):
        scraper.download_with_retry("https://this-domain-does-not-exist-123456789.com/article")
    
    # Test with rate-limited domain (using a mock)
    with patch('newspaper.Article.download') as mock_download:
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_download.side_effect = requests.HTTPError("Too Many Requests", response=mock_response)
        
        with pytest.raises(ArticleDownloadException):
            scraper.download_with_retry("https://example.com/article", max_retries=2)
            
        # Verify rate limiter was updated
        assert scraper.rate_limiter.get_delay("example.com") > scraper.rate_limiter.default_delay
```

### 3. Stress Tests

Test the system under high load to ensure error handling scales:

```python
def test_error_handling_under_load():
    """Test error handling under high load."""
    scraper = NewspaperScraper()
    urls = ["https://example.com/article" + str(i) for i in range(100)]
    
    # Mock to simulate random errors
    with patch('newspaper.Article.download') as mock_download:
        def random_error(*args, **kwargs):
            r = random.random()
            if r < 0.2:  # 20% timeout
                raise requests.Timeout("Connection timed out")
            elif r < 0.3:  # 10% connection error
                raise requests.ConnectionError("Connection refused")
            elif r < 0.4:  # 10% HTTP error
                mock_response = MagicMock()
                mock_response.status_code = 500
                raise requests.HTTPError("Server Error", response=mock_response)
            # 60% success
                
        mock_download.side_effect = random_error
        
        # Process URLs in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(lambda url: scraper.download_with_retry(url, max_retries=2), urls))
            
        # Check circuit breaker state
        assert "example.com" in scraper.circuit_breaker.domains
        assert scraper.circuit_breaker.domains["example.com"]["failures"] > 0
```

## Recommendations

1. **Implement Comprehensive Retry Logic**: Add exponential backoff with jitter to all HTTP requests
2. **Add Circuit Breaker Pattern**: Prevent repeated requests to failing domains
3. **Enhance Error Classification**: Better categorize and respond to different error types
4. **Improve Logging and Monitoring**: Track error patterns for analysis and optimization
5. **Implement Fallback Mechanisms**: Use alternative scraping methods when Newspaper4k fails
6. **Add Domain-Specific Error Handling**: Customize retry strategies for different domains
7. **Create Error Recovery Procedures**: Automated and manual procedures for handling persistent errors

## Next Steps

1. Implement the enhanced error handling and retry mechanisms
2. Create comprehensive test suite for error scenarios
3. Develop monitoring dashboard for error tracking
4. Document common error patterns and solutions
5. Integrate with alerting system for critical errors 