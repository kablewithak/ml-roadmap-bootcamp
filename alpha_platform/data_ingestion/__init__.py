"""Data ingestion modules for alternative data sources."""

from alpha_platform.data_ingestion.satellite.car_counter import SatelliteCarCounter
from alpha_platform.data_ingestion.satellite.oil_tank_analyzer import OilTankAnalyzer
from alpha_platform.data_ingestion.nlp.earnings_analyzer import EarningsCallAnalyzer
from alpha_platform.data_ingestion.nlp.sec_parser import SECFilingParser

__all__ = [
    "SatelliteCarCounter",
    "OilTankAnalyzer",
    "EarningsCallAnalyzer",
    "SECFilingParser",
]
