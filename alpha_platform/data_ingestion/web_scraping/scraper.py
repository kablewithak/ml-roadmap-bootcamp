"""
Web scraping with anti-detection measures.

Implements rotating proxies, user-agent rotation, and rate limiting.
"""

import random
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from alpha_platform.utils.config import get_config
from alpha_platform.utils.logger import get_logger

logger = get_logger(__name__)


class WebScraper:
    """
    Web scraper with anti-detection capabilities.

    Features:
    - Rotating user agents
    - Proxy support
    - Rate limiting
    - Retry logic
    """

    def __init__(
        self,
        proxies: Optional[List[str]] = None,
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
    ):
        """
        Initialize web scraper.

        Args:
            proxies: List of proxy URLs
            rate_limit_delay: Delay between requests in seconds
            max_retries: Maximum number of retries per request
        """
        self.config = get_config()
        self.proxies = proxies or []
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.ua = UserAgent()

        self.session = requests.Session()
        self.last_request_time = 0

        logger.info(f"Web scraper initialized with {len(self.proxies)} proxies")

    def scrape(
        self, url: str, parse_html: bool = True
    ) -> Optional[BeautifulSoup]:
        """
        Scrape a URL with anti-detection measures.

        Args:
            url: URL to scrape
            parse_html: Whether to parse HTML with BeautifulSoup

        Returns:
            BeautifulSoup object or None on failure
        """
        # Rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)

        # Try with retries
        for attempt in range(self.max_retries):
            try:
                # Random user agent
                headers = {"User-Agent": self.ua.random}

                # Random proxy if available
                proxy = None
                if self.proxies:
                    proxy_url = random.choice(self.proxies)
                    proxy = {"http": proxy_url, "https": proxy_url}

                # Make request
                response = self.session.get(
                    url, headers=headers, proxies=proxy, timeout=30
                )
                response.raise_for_status()

                self.last_request_time = time.time()

                if parse_html:
                    return BeautifulSoup(response.content, "html.parser")
                else:
                    return response.content

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        logger.error(f"Failed to scrape {url} after {self.max_retries} attempts")
        return None
