"""
Simple test script to fetch and analyze the first article from url.csv.
"""

import csv
import json
import urllib.request
import urllib.parse
from datetime import datetime
import re
import logging
from urllib.error import URLError, HTTPError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_first_article_url():
    """Read the first article URL from url.csv"""
    try:
        with open('url.csv', 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                if 'url' in row and row['url'].startswith('http'):
                    logger.info(f"Found URL: {row['url']}")
                    logger.info(f"Title from CSV: {row.get('title', 'No title in CSV')}")
                    return row['url']
        logger.error("No valid URL found in url.csv")
        return None
    except Exception as e:
        logger.error(f"Error reading url.csv: {e}")
        return None


def fetch_url(url):
    """Fetch the content of a URL with proper error handling"""
    logger.info(f"Fetching URL: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.read().decode('utf-8', errors='replace')
    except HTTPError as e:
        logger.error(f"HTTP Error: {e.code} - {url}")
        return None
    except URLError as e:
        logger.error(f"URL Error: {e.reason} - {url}")
        return None
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return None


def extract_title(html, url=None):
    """Extract the title from HTML using multiple methods"""
    if not html:
        return "Unknown Title"
    
    # First, try to extract from the CSV title if available
    if url:
        try:
            with open('url.csv', 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    if row.get('url') == url and row.get('title'):
                        logger.info(f"Using title from CSV: {row['title']}")
                        return row['title']
        except Exception as e:
            logger.warning(f"Could not get title from CSV: {e}")
    
    # Try og:title meta tag first (often most reliable)
    og_title_match = re.search(r'<meta\s+property=["\']og:title["\']\s+content=["\'](.*?)["\']', html)
    if og_title_match:
        title = og_title_match.group(1)
        logger.info(f"Found title in og:title: {title}")
        return title
    
    # Try twitter:title meta tag
    twitter_title_match = re.search(r'<meta\s+name=["\']twitter:title["\']\s+content=["\'](.*?)["\']', html)
    if twitter_title_match:
        title = twitter_title_match.group(1)
        logger.info(f"Found title in twitter:title: {title}")
        return title
    
    # Try standard title tag
    title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = title_match.group(1)
        # Clean up the title (remove site name, etc.)
        title = re.sub(r'\s*\|.*$', '', title)  # Remove pipe and everything after
        title = re.sub(r'\s*-.*$', '', title)   # Remove dash and everything after
        logger.info(f"Found title in title tag: {title}")
        return title.strip()
    
    # Try h1 tag as last resort
    h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.IGNORECASE | re.DOTALL)
    if h1_match:
        title = h1_match.group(1)
        # Clean up any HTML tags
        title = re.sub(r'<.*?>', '', title)
        logger.info(f"Found title in h1 tag: {title}")
        return title.strip()
    
    # ABC News specific pattern - look for the headline in the content
    abc_headline_match = re.search(r'<span[^>]*class=["\'].*?["\'][^>]*>(.*?)</span>', html, re.IGNORECASE | re.DOTALL)
    if abc_headline_match:
        title = abc_headline_match.group(1)
        # Clean up any HTML tags
        title = re.sub(r'<.*?>', '', title)
        logger.info(f"Found title in span tag: {title}")
        return title.strip()
    
    # Try to extract from URL for ABC News
    if url:
        url_parts = urllib.parse.urlparse(url).path.split('/')
        if url_parts and len(url_parts) > 1:
            # Get the last part of the URL path and replace hyphens with spaces
            potential_title = url_parts[-1].replace('-', ' ')
            # Remove any numbers or file extensions
            potential_title = re.sub(r'\d+$', '', potential_title)
            potential_title = re.sub(r'\..*$', '', potential_title)
            if len(potential_title) > 10:  # Only use if it seems like a reasonable title length
                logger.info(f"Using title from URL: {potential_title.title()}")
                return potential_title.title()  # Convert to title case
    
    return "Unknown Title"


def clean_html(html):
    """Clean HTML by removing scripts, styles, and other non-content elements"""
    if not html:
        return ""
    
    # Remove script tags and their contents
    html = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove style tags and their contents
    html = re.sub(r'<style.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove comments
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
    
    # Remove navigation, header, footer elements
    html = re.sub(r'<nav.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<header.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<footer.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove other common non-content elements
    html = re.sub(r'<aside.*?</aside>', '', html, flags=re.DOTALL | re.IGNORECASE)
    
    return html


def extract_content(html):
    """Extract the main content from HTML"""
    if not html:
        return ""
    
    # Clean the HTML first
    cleaned_html = clean_html(html)
    
    # Try to find the main content container
    main_content = None
    
    # Look for common content containers
    content_containers = [
        r'<article[^>]*>(.*?)</article>',
        r'<div[^>]*class=["\'].*?(?:content|article|story|entry).*?["\'][^>]*>(.*?)</div>',
        r'<main[^>]*>(.*?)</main>',
        r'<div[^>]*id=["\'].*?(?:content|article|story|entry).*?["\'][^>]*>(.*?)</div>'
    ]
    
    for pattern in content_containers:
        matches = re.findall(pattern, cleaned_html, re.DOTALL | re.IGNORECASE)
        if matches:
            main_content = max(matches, key=len)
            break
    
    # If no content container found, use the whole cleaned HTML
    if not main_content:
        main_content = cleaned_html
    
    # Extract paragraphs
    paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', main_content, re.DOTALL | re.IGNORECASE)
    
    # If no paragraphs found, try a more general approach
    if not paragraphs:
        # Split by line breaks and filter
        paragraphs = re.split(r'<br\s*/?>|</div>|</p>', main_content)
    
    # Clean paragraphs and filter out short ones (likely navigation or other non-content)
    cleaned_paragraphs = []
    for p in paragraphs:
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', p)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Filter out short paragraphs
        if len(text) > 20:
            cleaned_paragraphs.append(text)
    
    # Join paragraphs with newlines
    content = '\n\n'.join(cleaned_paragraphs)
    
    # If still no content, try a last resort approach
    if not content:
        # Remove all HTML tags and keep text
        text = re.sub(r'<.*?>', ' ', cleaned_html)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Split into sentences and filter
        sentences = re.split(r'[.!?]+', text)
        content = ' '.join([s.strip() for s in sentences if len(s.strip()) > 30])
    
    return content


def extract_metadata(html, url):
    """Extract metadata from HTML"""
    if not html:
        return {}
    
    metadata = {
        'url': url,
        'extraction_date': datetime.now().isoformat(),
        'source_domain': urllib.parse.urlparse(url).netloc,
    }
    
    # Extract author
    author_patterns = [
        r'<meta\s+name=["\']author["\']\s+content=["\'](.*?)["\']',
        r'<meta\s+property=["\']article:author["\']\s+content=["\'](.*?)["\']',
        r'<span[^>]*class=["\'].*?author.*?["\'][^>]*>(.*?)</span>',
        r'<div[^>]*class=["\'].*?author.*?["\'][^>]*>(.*?)</div>',
        r'<a[^>]*rel=["\']author["\'](.*?)</a>'
    ]
    
    for pattern in author_patterns:
        author_match = re.search(pattern, html, re.IGNORECASE)
        if author_match:
            author = author_match.group(1)
            # Clean up author text
            author = re.sub(r'<.*?>', '', author)
            author = re.sub(r'\s+', ' ', author).strip()
            if author:
                metadata['author'] = author
                break
    
    # If no author found, use domain as fallback
    if 'author' not in metadata:
        metadata['author'] = metadata['source_domain']
    
    # Extract publication date
    date_patterns = [
        r'<meta\s+property=["\']article:published_time["\']\s+content=["\'](.*?)["\']',
        r'<meta\s+name=["\']pubdate["\']\s+content=["\'](.*?)["\']',
        r'<time[^>]*datetime=["\']([^"\']+)["\'][^>]*>',
        r'<meta\s+name=["\']date["\']\s+content=["\'](.*?)["\']'
    ]
    
    for pattern in date_patterns:
        date_match = re.search(pattern, html, re.IGNORECASE)
        if date_match:
            metadata['publication_date'] = date_match.group(1)
            break
    
    return metadata


def analyze_url(url):
    """Analyze a URL and extract relevant information"""
    if not url:
        logger.error("No URL provided")
        return None
    
    html = fetch_url(url)
    if not html:
        logger.error(f"Failed to fetch content from {url}")
        return None
    
    title = extract_title(html, url)
    content = extract_content(html)
    metadata = extract_metadata(html, url)
    
    logger.info(f"Title: {title}")
    logger.info(f"Content length: {len(content)} characters")
    logger.info(f"Source domain: {metadata.get('source_domain')}")
    
    result = {
        'url': url,
        'title': title,
        'content_preview': content[:200] + '...' if len(content) > 200 else content,
        'content_length': len(content),
        'metadata': metadata
    }
    
    return result


def save_to_file(result, filename=None):
    """Save the extraction result to a file"""
    if not result:
        logger.error("No result to save")
        return False
    
    if not filename:
        # Generate filename based on URL
        domain = result['metadata']['source_domain']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"extracted_{domain}_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(result, file, indent=2, ensure_ascii=False)
        logger.info(f"Saved extraction result to {filename}")
        
        # Also save the content to a separate text file for easier reading
        content_filename = filename.replace('.json', '.txt')
        with open(content_filename, 'w', encoding='utf-8') as file:
            file.write(f"Title: {result['title']}\n\n")
            file.write(f"URL: {result['url']}\n\n")
            file.write(f"Extracted on: {result['metadata']['extraction_date']}\n\n")
            file.write("Content:\n\n")
            file.write(result['content_full'] if 'content_full' in result else result.get('content', ''))
        logger.info(f"Saved content to {content_filename}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving to file: {e}")
        return False


def main():
    """Main function to run the script"""
    url = get_first_article_url()
    if not url:
        logger.error("Failed to get a URL to analyze")
        return
    
    logger.info(f"Analyzing URL: {url}")
    result = analyze_url(url)
    
    if result:
        logger.info(f"Analysis complete. Title: {result['title']}")
        logger.info(f"Content length: {result['content_length']} characters")
        
        # Add the full content to the result
        content = extract_content(fetch_url(url))
        result['content_full'] = content
        
        # Print the result to console
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Save to file
        save_to_file(result)
    else:
        logger.error("Analysis failed")


if __name__ == "__main__":
    main() 