"""
SEC filing parser for 10-K, 10-Q, and 8-K filings.

Extracts risk factors, MD&A sections, and key financial data.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from alpha_platform.utils.config import get_config
from alpha_platform.utils.logger import get_logger

logger = get_logger(__name__)


class SECFilingParser:
    """Parse and analyze SEC filings."""

    def __init__(self):
        """Initialize SEC filing parser."""
        self.config = get_config()
        logger.info("SEC filing parser initialized")

    def parse_10k(self, filing_text: str, ticker: str) -> Dict[str, Any]:
        """
        Parse 10-K annual report.

        Args:
            filing_text: Full text of 10-K filing
            ticker: Stock ticker symbol

        Returns:
            Parsed filing data
        """
        logger.info(f"Parsing 10-K for {ticker}")

        sections = {
            "business": self._extract_section(filing_text, "item 1", "item 1a"),
            "risk_factors": self._extract_section(filing_text, "item 1a", "item 1b"),
            "mda": self._extract_section(filing_text, "item 7", "item 8"),
            "financial_statements": self._extract_section(
                filing_text, "item 8", "item 9"
            ),
        }

        return {
            "ticker": ticker,
            "filing_type": "10-K",
            "sections": sections,
            "parse_timestamp": datetime.now().isoformat(),
        }

    def _extract_section(
        self, text: str, start_marker: str, end_marker: str
    ) -> str:
        """Extract section between two markers."""
        start_pattern = re.compile(start_marker, re.IGNORECASE)
        end_pattern = re.compile(end_marker, re.IGNORECASE)

        start_match = start_pattern.search(text)
        end_match = end_pattern.search(text)

        if start_match and end_match:
            return text[start_match.end() : end_match.start()].strip()

        return ""
