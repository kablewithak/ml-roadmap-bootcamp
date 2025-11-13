"""
Alternative Data Alpha Generation Platform with Explainable Deep Learning

A production-grade quantitative trading platform that discovers and trades
market inefficiencies using non-traditional data sources.

Components:
- Data Ingestion: Satellite imagery, NLP, web scraping, credit card data
- Feature Engineering: Multi-modal deep learning, graph neural networks
- Alpha Generation: Ensemble models with explainable AI
- Backtesting: Event-driven simulation with walk-forward validation
- Trading: Real-time execution with risk management
- Monitoring: Model governance and performance tracking
"""

__version__ = "0.1.0"
__author__ = "ML Roadmap Bootcamp"

from alpha_platform.utils.config import load_config
from alpha_platform.utils.logger import get_logger

# Initialize platform
config = load_config()
logger = get_logger(__name__)

logger.info(f"Alpha Platform v{__version__} initialized")
