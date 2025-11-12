from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="adversarial-fraud-detection",
    version="0.1.0",
    author="Adversarial Fraud Detection Team",
    description="Production-grade adversarial ML system for fraud detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kablewithak/ml-roadmap-bootcamp",
    packages=find_packages(exclude=["tests", "docs", "scripts"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "networkx>=3.1",
        "stable-baselines3>=2.0.0",
        "fastapi>=0.100.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.16.0",
            "streamlit>=1.25.0",
        ],
        "monitoring": [
            "prometheus-client>=0.17.0",
            "opentelemetry-api>=1.20.0",
            "evidently>=0.4.0",
        ],
    },
)
