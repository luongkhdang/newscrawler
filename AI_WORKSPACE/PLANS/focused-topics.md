list of news topic we care about:
   - Laws, regulations, global political shifts (that involved USA, Vietnam, China, Japan, Mexico, Germany, Singapore, and Taiwan).
   - Analysis of influential figures and decision-makers in The US, Vietnam and China.
   - Trade agreements and export-import dynamics (USA, Vietnam, China, Japan, Mexico, Germany, Singapore, and Taiwan).
   - Investment trends and economic indicators.
   - Currency interaction (USD, USD-EURO, USD-VND, USD-YEN, USD-YUAN).
   - Tracking of S&P 500, DJIA, QQQ.
   - Aggregation of research papers and official statistics.

NOTE THAT WE FILTER OUT UNRELATED INFORMATION TO OPTIMIZE STORAGE/DATA-BASE


DATA SOURCE API DETAILS:

1. MediaCloud API (MAIN URL SOURCE - GET YESTERDAY TO TODAY NEWS):
   - API: https://pypi.org/project/mediacloud/
   - Capabilities: Cross-platform news search and collection/source/feed directory browsing
   - Key Features:
     * Story list search with query parameters (e.g., 'modi AND biden')
     * Collection-based filtering (e.g., US_NATIONAL_COLLECTION, INDIA_NATIONAL_COLLECTION)
     * Date range filtering with start_date and end_date parameters
     * Pagination support for retrieving large result sets
   - Implementation Strategy:
     * Use SearchApi for querying news articles with focused topics as keywords
     * Filter by collection_ids relevant to our countries of interest
     * Set date range from yesterday to today to avoid duplicates
     * Process and store articles matching our focused topics

2. NewsAPI (DIRECT NEWS SOURCE - ARTICLES HAVE A 24 HOUR DELAY - GET YESTERDAY NEWS):
   - API: https://newsapi.org/docs/
   - Capabilities: Simple REST API for searching and retrieving live articles worldwide
   - Key Features:
     * Everything endpoint (/v2/everything) for comprehensive article search
     * Top headlines endpoint (/v2/top-headlines) for breaking news by country/category
     * Powerful filtering by keywords, date, source, and language
     * Boolean operators for complex queries (AND, OR, NOT)
   - Implementation Strategy:
     * Use the Everything endpoint with specific queries for our focused topics
     * Filter by domains relevant to financial/political news
     * Set date parameters to yesterday to complement MediaCloud data
     * Implement language filtering for relevant countries (en, zh, vi, ja, de, es)

3. NewsFilter API (DIRECT NEWS STOCK SOURCE):
   - API: https://developers.newsfilter.io/
   - Capabilities: Specialized stock market news API with real-time updates
   - Key Features:
     * Query API for searching the entire news corpus
     * Stream API for real-time article notifications
     * Company/ticker identification through NLP
     * Sub-500ms indexing of new articles
   - Implementation Strategy:
     * Use Query API for historical stock market news related to our focused topics
     * Implement Stream API for real-time monitoring of market-moving news
     * Filter by companies/tickers relevant to our countries of interest
     * Focus on financial news that impacts trade agreements and currency interactions

4. Alpha Vantage API (MARKET NEWS & SENTIMENT, COMPANY OVERVIEW, INSIDER TRANSACTIONS, HISTORICAL OPTIONS):
   - API: https://www.alphavantage.co/documentation/
   - Capabilities: Comprehensive financial data API with news, fundamentals, and market data
   - Key Features:
     * Market News & Sentiment API with topic filtering
     * Company Overview for fundamental data
     * Insider Transactions for tracking significant moves
     * Historical Options data for derivatives analysis
   - Implementation Strategy:
     * Use NEWS_SENTIMENT function with tickers relevant to our focused countries
     * Filter by topics like economy_fiscal, economy_monetary, financial_markets
     * Track specific companies influential in international trade
     * Monitor currency pairs (USD/EUR, USD/VND, USD/JPY, USD/CNY)
     * Set time_from and time_to parameters to get one day of data

INTEGRATION APPROACH:

1. Daily Data Collection Pipeline:
   - Schedule API calls to run once daily
   - Set date parameters to fetch only yesterday's news (avoid duplicates)
   - Implement error handling and retry mechanisms for API failures
   - Log all API responses for debugging and auditing

2. Content Filtering Strategy:
   - Apply NLP techniques to identify articles matching our focused topics
   - Implement keyword and entity extraction to categorize content
   - Filter out unrelated information to optimize storage
   - Prioritize articles mentioning multiple countries of interest

3. Storage Optimization:
   - Store only essential article metadata and content
   - Implement deduplication to avoid storing the same news from different sources
   - Use efficient indexing for quick retrieval by topic, country, or entity
   - Implement data retention policies based on relevance and age

4. Analysis Integration:
   - Connect collected data to analysis pipelines for trend identification
   - Generate daily summaries of key developments in focused topics
   - Track sentiment around key entities and countries
   - Correlate news events with market movements in relevant indices and currencies