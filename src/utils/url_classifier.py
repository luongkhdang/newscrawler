"""
URL classification system for determining the best scraper strategy for each URL.
"""

import re
import logging
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple, Set
import sqlite3
from datetime import datetime
import json
import threading

from src.utils.exceptions import ConfigurationError, NetworkError


class URLClassifier:
    """
    Classifies URLs to determine the best scraper strategy.
    """
    
    # Common patterns for different site types
    PATTERNS = {
        'feed': [
            r'rss\.xml$',
            r'feed\.xml$',
            r'/rss/',
            r'/feed/',
            r'/atom/',
        ],
        'js_heavy': [
            r'react-',
            r'angular',
            r'vue',
            r'spa',
            r'single-page',
            r'next\.js',
            r'gatsby',
            r'nuxt',
            r'svelte',
            r'ember',
            r'backbone',
            r'knockout',
            r'meteor',
            r'mithril',
            r'preact',
            r'stimulus',
            r'webcomponents',
            r'shadow-dom',
            r'dynamic',
            r'interactive',
            r'ajax',
            r'webpack',
            r'parcel',
            r'rollup',
            r'vite',
            r'esbuild',
            r'snowpack',
            r'turbopack',
        ],
        'paywall': [
            r'subscriber',
            r'subscription',
            r'premium',
            r'paywall',
            r'member',
        ]
    }
    
    def __init__(self, db_path: str = 'domains.db'):
        """
        Initialize the URL classifier.
        
        Args:
            db_path: Path to the SQLite database for domain classification
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize the database
        self._init_db()
        
        # Cache for domain classifications
        self.domain_cache = {}
        
        # Thread-local storage for database connections
        self.local = threading.local()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self.local, 'conn'):
            self.local.conn = sqlite3.connect(self.db_path)
        return self.local.conn
    
    def _init_db(self):
        """Initialize the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS domains (
            domain TEXT PRIMARY KEY,
            strategy TEXT,
            features TEXT,
            rate_limit REAL,
            success_rate REAL,
            last_updated TIMESTAMP
        )
        ''')
        conn.commit()
        conn.close()
    
    def classify_url(self, url: str) -> str:
        """
        Classify a URL to determine the best scraper strategy.
        
        Args:
            url: The URL to classify
            
        Returns:
            The name of the best strategy ('newspaper', 'feed', 'bs4', or 'puppeteer')
        """
        domain = urllib.parse.urlparse(url).netloc
        
        # Check cache first
        if domain in self.domain_cache:
            self.logger.debug(f"Using cached classification for {domain}: {self.domain_cache[domain]}")
            return self.domain_cache[domain]
        
        # Check database
        strategy = self.get_domain_strategy(domain)
        if strategy:
            self.domain_cache[domain] = strategy
            return strategy
        
        # Analyze the URL and domain
        strategy = self._analyze_url(url)
        
        # Store in database and cache
        self.set_domain_strategy(domain, strategy)
        self.domain_cache[domain] = strategy
        
        return strategy
    
    def _analyze_url(self, url: str) -> str:
        """
        Analyze a URL to determine its characteristics and best strategy.
        
        Args:
            url: The URL to analyze
            
        Returns:
            The name of the best strategy
        """
        features = self._extract_features(url)
        
        # Decision logic based on features
        if features.get('has_feed', False):
            return 'feed'
        elif features.get('js_heavy', False):
            return 'puppeteer'
        elif features.get('complex_structure', False):
            return 'bs4'
        else:
            return 'newspaper'  # Default strategy
    
    def _extract_features(self, url: str) -> Dict[str, bool]:
        """
        Extract features from a URL to aid in classification.
        
        Args:
            url: The URL to analyze
            
        Returns:
            Dictionary of features
        """
        features = {
            'has_feed': False,
            'js_heavy': False,
            'paywall': False,
            'complex_structure': False
        }
        
        # Check URL patterns
        for pattern in self.PATTERNS['feed']:
            if re.search(pattern, url, re.IGNORECASE):
                features['has_feed'] = True
                break
        
        for pattern in self.PATTERNS['js_heavy']:
            if re.search(pattern, url, re.IGNORECASE):
                features['js_heavy'] = True
                break
        
        for pattern in self.PATTERNS['paywall']:
            if re.search(pattern, url, re.IGNORECASE):
                features['paywall'] = True
                break
        
        # Check for RSS feed availability
        domain = urllib.parse.urlparse(url).netloc
        if self._check_for_feed(domain):
            features['has_feed'] = True
        
        # Check for JavaScript-heavy site
        if not features['js_heavy']:
            features['js_heavy'] = self._check_for_js_heavy(url)
        
        # Determine if the site has a complex structure
        # This is a simplified heuristic - in a real system, we would do more analysis
        if features['js_heavy'] or features['paywall']:
            features['complex_structure'] = True
        
        return features
    
    def _check_for_feed(self, domain: str) -> bool:
        """
        Check if a domain has an RSS feed.
        
        Args:
            domain: The domain to check
            
        Returns:
            True if an RSS feed is found, False otherwise
        """
        common_feed_paths = [
            '/rss',
            '/feed',
            '/rss.xml',
            '/feed.xml',
            '/atom.xml',
            '/feeds/posts/default',
            '/rss/all.xml',
        ]
        
        for path in common_feed_paths:
            feed_url = f"https://{domain}{path}"
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                req = urllib.request.Request(feed_url, headers=headers)
                with urllib.request.urlopen(req, timeout=5) as response:
                    content = response.read().decode('utf-8', errors='ignore')
                    # Simple check for RSS/Atom feed markers
                    if '<rss' in content or '<feed' in content or '<atom' in content:
                        self.logger.info(f"Found feed at {feed_url}")
                        return True
            except Exception as e:
                self.logger.debug(f"Error checking feed at {feed_url}: {str(e)}")
                continue
        
        return False
    
    def _check_for_js_heavy(self, url: str) -> bool:
        """
        Check if a URL points to a JavaScript-heavy site.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the site is JavaScript-heavy, False otherwise
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                content = response.read().decode('utf-8', errors='ignore')
                
                # Check for common JavaScript frameworks and libraries
                js_frameworks = [
                    'react', 'angular', 'vue', 'next', 'nuxt', 'gatsby', 'svelte',
                    'ember', 'backbone', 'knockout', 'meteor', 'mithril', 'preact',
                    'stimulus', 'webcomponents', 'shadow-dom'
                ]
                
                for framework in js_frameworks:
                    if f'{framework}.js' in content or f'{framework}.min.js' in content:
                        self.logger.info(f"Detected JavaScript framework: {framework}")
                        return True
                
                # Check for SPA indicators
                spa_indicators = [
                    '<div id="root">', '<div id="app">', '<div id="__next">',
                    'window.__INITIAL_STATE__', 'window.__PRELOADED_STATE__',
                    'ReactDOM.render', 'createApp', 'createRoot',
                    'data-reactroot', 'ng-app', 'v-app', 'ember-app'
                ]
                
                for indicator in spa_indicators:
                    if indicator in content:
                        self.logger.info(f"Detected SPA indicator: {indicator}")
                        return True
                
                # Check for high JavaScript usage
                js_tags = content.count('<script')
                if js_tags > 15:  # Arbitrary threshold for "many" script tags
                    self.logger.info(f"Detected high JavaScript usage: {js_tags} script tags")
                    return True
                
                # Check for dynamic content loading
                dynamic_indicators = [
                    'fetch(', 'axios.', 'ajax(', 'XMLHttpRequest',
                    'IntersectionObserver', 'lazyload', 'infinite-scroll',
                    'data-src=', 'data-lazy', 'loading="lazy"'
                ]
                
                for indicator in dynamic_indicators:
                    if indicator in content:
                        self.logger.info(f"Detected dynamic content loading: {indicator}")
                        return True
        
        except Exception as e:
            self.logger.debug(f"Error checking for JavaScript-heavy site at {url}: {str(e)}")
        
        return False
    
    def get_domain_strategy(self, domain: str) -> Optional[str]:
        """
        Get the stored strategy for a domain.
        
        Args:
            domain: The domain to look up
            
        Returns:
            The strategy name or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT strategy FROM domains WHERE domain = ?", (domain,))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def set_domain_strategy(self, domain: str, strategy: str, features: Dict[str, bool] = None, rate_limit: float = 1.0):
        """
        Store the strategy for a domain.
        
        Args:
            domain: The domain to store
            strategy: The strategy to use
            features: Features extracted from the domain
            rate_limit: Rate limit in seconds
        """
        features_json = json.dumps(features) if features else '{}'
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        INSERT OR REPLACE INTO domains (domain, strategy, features, rate_limit, success_rate, last_updated)
        VALUES (?, ?, ?, ?, 0.0, ?)
        ''', (domain, strategy, features_json, rate_limit, datetime.now().isoformat()))
        conn.commit()
    
    def update_success_rate(self, domain: str, success: bool):
        """
        Update the success rate for a domain.
        
        Args:
            domain: The domain to update
            success: Whether the scraping was successful
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        UPDATE domains
        SET success_rate = (success_rate * 0.9) + (? * 0.1),
            last_updated = ?
        WHERE domain = ?
        ''', (1.0 if success else 0.0, datetime.now().isoformat(), domain))
        conn.commit()
    
    def get_domain_stats(self) -> Dict[str, Dict]:
        """
        Get statistics for all domains.
        
        Returns:
            Dictionary of domain statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT domain, strategy, features, rate_limit, success_rate, last_updated FROM domains")
        results = cursor.fetchall()
        
        stats = {}
        for row in results:
            domain, strategy, features_json, rate_limit, success_rate, last_updated = row
            stats[domain] = {
                'strategy': strategy,
                'features': json.loads(features_json),
                'rate_limit': rate_limit,
                'success_rate': success_rate,
                'last_updated': last_updated
            }
        
        return stats 