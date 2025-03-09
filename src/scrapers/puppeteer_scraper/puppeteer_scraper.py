"""
Puppeteer scraper for JavaScript-heavy sites.
This module provides a scraper implementation that uses Puppeteer to render JavaScript.
"""

import logging
import json
import os
import subprocess
import tempfile
from typing import Dict, Any, Optional, List
import time
from urllib.parse import urlparse

from src.scrapers.base_scraper import BaseScraper
from src.models.article import Article, ArticleMetadata, ArticleImage
from src.scrapers.rate_limiter import get_crawler_rate_limiter

logger = logging.getLogger(__name__)


class PuppeteerScraper(BaseScraper):
    """
    Scraper implementation that uses Puppeteer to render JavaScript.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Puppeteer scraper.
        
        Args:
            config: Configuration options for the scraper
        """
        super().__init__(config)
        
        # Default configuration
        self.default_config = {
            "wait_time": 5,  # seconds to wait for page to load
            "scroll": True,  # whether to scroll the page
            "scroll_count": 3,  # number of times to scroll
            "scroll_delay": 1,  # seconds to wait between scrolls
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "timeout": 30,  # seconds to wait for Puppeteer to complete
            "puppeteer_path": "node_modules/.bin/puppeteer",  # path to Puppeteer executable
            "script_path": os.path.join(os.path.dirname(__file__), "puppeteer_script.js"),  # path to Puppeteer script
            "headless": True,  # whether to run in headless mode
            "respect_robots_txt": True,  # whether to respect robots.txt
            "rate_limit": True,  # whether to apply rate limiting
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}
        
        # Check if Puppeteer script exists
        if not os.path.exists(self.config["script_path"]):
            self._create_puppeteer_script()
        
        # Get rate limiter
        if self.config["rate_limit"] or self.config["respect_robots_txt"]:
            self.rate_limiter = get_crawler_rate_limiter(
                user_agent=self.config["user_agent"]
            )
        else:
            self.rate_limiter = None
    
    def can_fetch(self, url: str) -> bool:
        """
        Check if the URL can be fetched according to robots.txt.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the URL can be fetched, False otherwise
        """
        if not self.config["respect_robots_txt"] or not self.rate_limiter:
            return True
        
        return self.rate_limiter.can_fetch(url)
    
    def scrape(self, url: str) -> Optional[Article]:
        """
        Scrape an article from the given URL using Puppeteer.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Standardized Article object or None if scraping failed
            
        Raises:
            ScraperError: For scraping errors
        """
        try:
            # Check if URL can be fetched
            if not self.can_fetch(url):
                logger.warning(f"URL {url} cannot be fetched according to robots.txt")
                return None
            
            # Apply rate limiting if enabled
            if self.config["rate_limit"] and self.rate_limiter:
                self.rate_limiter.wait_if_needed(url)
            
            # Create temporary file for output
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
                output_path = temp_file.name
            
            # Build command
            command = [
                "node",
                self.config["script_path"],
                url,
                output_path,
                str(self.config["wait_time"]),
                str(self.config["scroll"]).lower(),
                str(self.config["scroll_count"]),
                str(self.config["scroll_delay"]),
                self.config["user_agent"],
                str(self.config["headless"]).lower()
            ]
            
            # Run Puppeteer
            logger.info(f"Running Puppeteer for URL: {url}")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for process to complete with timeout
            try:
                stdout, stderr = process.communicate(timeout=self.config["timeout"])
                
                if process.returncode != 0:
                    logger.error(f"Puppeteer failed for URL {url}: {stderr}")
                    return None
                
                # Read output file
                with open(output_path, "r", encoding="utf-8") as f:
                    result = json.load(f)
                
                # Clean up
                os.unlink(output_path)
                
                # Convert to Article object
                return self._convert_to_article(result, url)
            
            except subprocess.TimeoutExpired:
                process.kill()
                logger.error(f"Puppeteer timed out for URL {url}")
                return None
        
        except Exception as e:
            logger.error(f"Error scraping URL {url} with Puppeteer: {e}")
            return None
    
    def get_strategy_name(self) -> str:
        """
        Get the name of this scraper strategy.
        
        Returns:
            The strategy name as a string
        """
        return "puppeteer"
    
    def _convert_to_article(self, result: Dict[str, Any], url: str) -> Optional[Article]:
        """
        Convert Puppeteer result to Article object.
        
        Args:
            result: Puppeteer result
            url: Original URL
            
        Returns:
            Article object or None if conversion failed
        """
        try:
            # Extract basic fields
            title = result.get("title", "")
            content = result.get("content", "")
            
            # Skip if no title or content
            if not title or not content:
                logger.warning(f"No title or content found for URL {url}")
                return None
            
            # Extract metadata
            metadata = ArticleMetadata(
                authors=result.get("authors", []),
                published_date=result.get("published_date"),
                modified_date=result.get("modified_date"),
                section=result.get("section"),
                tags=result.get("tags", []),
                summary=result.get("summary", "")
            )
            
            # Extract images
            images = []
            for img in result.get("images", []):
                images.append(ArticleImage(
                    url=img.get("url", ""),
                    alt=img.get("alt", ""),
                    caption=img.get("caption", "")
                ))
            
            # Extract domain
            domain = urlparse(url).netloc
            
            # Create Article object
            article = Article(
                url=url,
                title=title,
                content=content,
                metadata=metadata,
                images=images,
                domain=domain
            )
            
            # Calculate quality score
            article.quality_score = self.calculate_quality_score(article)
            
            return article
        
        except Exception as e:
            logger.error(f"Error converting Puppeteer result to Article: {e}")
            return None
    
    def _create_puppeteer_script(self):
        """Create the Puppeteer script if it doesn't exist."""
        script_dir = os.path.dirname(self.config["script_path"])
        os.makedirs(script_dir, exist_ok=True)
        
        script_content = """
const puppeteer = require('puppeteer');
const fs = require('fs');

async function scrape(url, outputPath, waitTime, scroll, scrollCount, scrollDelay, userAgent, headless) {
    // Convert string parameters to appropriate types
    waitTime = parseInt(waitTime);
    scroll = scroll === 'true';
    scrollCount = parseInt(scrollCount);
    scrollDelay = parseInt(scrollDelay);
    headless = headless === 'true';
    
    const browser = await puppeteer.launch({
        headless: headless ? 'new' : false,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    
    try {
        const page = await browser.newPage();
        
        // Set user agent
        await page.setUserAgent(userAgent);
        
        // Set viewport
        await page.setViewport({ width: 1280, height: 800 });
        
        // Enable JavaScript
        await page.setJavaScriptEnabled(true);
        
        // Navigate to URL
        await page.goto(url, { waitUntil: 'networkidle2', timeout: waitTime * 1000 });
        
        // Scroll if enabled
        if (scroll) {
            for (let i = 0; i < scrollCount; i++) {
                await page.evaluate(() => {
                    window.scrollBy(0, window.innerHeight);
                });
                await page.waitForTimeout(scrollDelay * 1000);
            }
        }
        
        // Extract article data
        const result = await page.evaluate(() => {
            // Helper function to clean text
            const cleanText = (text) => {
                if (!text) return '';
                return text.trim()
                    .replace(/\\s+/g, ' ')
                    .replace(/\\n+/g, '\\n');
            };
            
            // Extract title
            const title = document.title || '';
            
            // Try to find article content
            let content = '';
            
            // Look for article or main content
            const articleSelectors = [
                'article',
                '[role="article"]',
                '.article',
                '.post',
                '.content',
                'main',
                '#main',
                '.main',
                '.post-content',
                '.entry-content'
            ];
            
            let contentElement = null;
            
            for (const selector of articleSelectors) {
                const elements = document.querySelectorAll(selector);
                if (elements.length > 0) {
                    // Find the largest element by content length
                    let maxLength = 0;
                    let maxElement = null;
                    
                    for (const element of elements) {
                        const text = element.textContent || '';
                        if (text.length > maxLength) {
                            maxLength = text.length;
                            maxElement = element;
                        }
                    }
                    
                    if (maxElement && maxLength > 200) {
                        contentElement = maxElement;
                        break;
                    }
                }
            }
            
            // If no content element found, use body
            if (!contentElement) {
                contentElement = document.body;
            }
            
            // Extract content
            content = cleanText(contentElement.textContent || '');
            
            // Extract authors
            const authors = [];
            const authorSelectors = [
                '[rel="author"]',
                '.author',
                '.byline',
                '[itemprop="author"]',
                '.entry-author'
            ];
            
            for (const selector of authorSelectors) {
                const authorElements = document.querySelectorAll(selector);
                for (const element of authorElements) {
                    const author = cleanText(element.textContent);
                    if (author && !authors.includes(author)) {
                        authors.push(author);
                    }
                }
            }
            
            // Extract published date
            let publishedDate = null;
            const dateSelectors = [
                '[itemprop="datePublished"]',
                '[property="article:published_time"]',
                'time',
                '.date',
                '.published',
                '.entry-date',
                '.post-date'
            ];
            
            for (const selector of dateSelectors) {
                const dateElements = document.querySelectorAll(selector);
                for (const element of dateElements) {
                    // Try to get date from datetime attribute
                    const datetime = element.getAttribute('datetime');
                    if (datetime) {
                        publishedDate = datetime;
                        break;
                    }
                    
                    // Try to get date from content
                    const dateText = cleanText(element.textContent);
                    if (dateText) {
                        publishedDate = dateText;
                        break;
                    }
                }
                
                if (publishedDate) break;
            }
            
            // Extract modified date
            let modifiedDate = null;
            const modifiedSelectors = [
                '[itemprop="dateModified"]',
                '[property="article:modified_time"]'
            ];
            
            for (const selector of modifiedSelectors) {
                const elements = document.querySelectorAll(selector);
                if (elements.length > 0) {
                    const element = elements[0];
                    modifiedDate = element.getAttribute('datetime') || element.getAttribute('content');
                    if (modifiedDate) break;
                }
            }
            
            // Extract section/category
            let section = null;
            const sectionSelectors = [
                '[itemprop="articleSection"]',
                '[property="article:section"]',
                '.category',
                '.section'
            ];
            
            for (const selector of sectionSelectors) {
                const elements = document.querySelectorAll(selector);
                if (elements.length > 0) {
                    section = cleanText(elements[0].textContent);
                    if (section) break;
                }
            }
            
            // Extract tags
            const tags = [];
            const tagSelectors = [
                '[rel="tag"]',
                '.tag',
                '.tags a',
                '[property="article:tag"]'
            ];
            
            for (const selector of tagSelectors) {
                const tagElements = document.querySelectorAll(selector);
                for (const element of tagElements) {
                    const tag = cleanText(element.textContent);
                    if (tag && !tags.includes(tag)) {
                        tags.push(tag);
                    }
                }
            }
            
            // Extract summary/description
            let summary = '';
            const summarySelectors = [
                '[name="description"]',
                '[property="og:description"]',
                '[itemprop="description"]',
                '.summary',
                '.description',
                '.excerpt'
            ];
            
            for (const selector of summarySelectors) {
                const elements = document.querySelectorAll(selector);
                if (elements.length > 0) {
                    const element = elements[0];
                    summary = element.getAttribute('content') || cleanText(element.textContent);
                    if (summary) break;
                }
            }
            
            // Extract images
            const images = [];
            const imageElements = document.querySelectorAll('img');
            
            for (const element of imageElements) {
                const src = element.getAttribute('src');
                if (!src) continue;
                
                // Skip small images and icons
                const width = parseInt(element.getAttribute('width') || '0');
                const height = parseInt(element.getAttribute('height') || '0');
                
                if (width > 0 && height > 0 && (width < 100 || height < 100)) {
                    continue;
                }
                
                // Get caption
                let caption = '';
                const figcaption = element.closest('figure')?.querySelector('figcaption');
                if (figcaption) {
                    caption = cleanText(figcaption.textContent);
                }
                
                images.push({
                    url: src,
                    alt: element.getAttribute('alt') || '',
                    caption: caption
                });
            }
            
            return {
                title,
                content,
                authors,
                published_date: publishedDate,
                modified_date: modifiedDate,
                section,
                tags,
                summary,
                images
            };
        });
        
        // Write result to output file
        fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
        
        return result;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    } finally {
        await browser.close();
    }
}

// Get command line arguments
const [url, outputPath, waitTime, scroll, scrollCount, scrollDelay, userAgent, headless] = process.argv.slice(2);

// Run scraper
scrape(url, outputPath, waitTime, scroll, scrollCount, scrollDelay, userAgent, headless)
    .then(() => {
        console.log('Scraping completed successfully');
        process.exit(0);
    })
    .catch((error) => {
        console.error('Scraping failed:', error);
        process.exit(1);
    });
"""
        
        with open(self.config["script_path"], "w", encoding="utf-8") as f:
            f.write(script_content)
        
        logger.info(f"Created Puppeteer script at {self.config['script_path']}") 