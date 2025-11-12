"""
Adversarial Fraud Detection System

A production-grade ML system for detecting and preventing fraud through
adversarial learning, graph analytics, and robust defense mechanisms.
"""

__version__ = "0.1.0"
__author__ = "Adversarial Fraud Detection Team"

from fraud_detection.attacks import AttackSimulator
from fraud_detection.defense import DefenseSystem
from fraud_detection.graph import GraphFraudDetector
from fraud_detection.adversarial import AdversarialLearner
from fraud_detection.evaluation import EvaluationFramework

__all__ = [
    "AttackSimulator",
    "DefenseSystem",
    "GraphFraudDetector",
    "AdversarialLearner",
    "EvaluationFramework",
]
