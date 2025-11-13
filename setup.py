"""Setup script for Alpha Generation Platform."""

from setuptools import setup, find_packages

setup(
    name="alpha_platform",
    version="0.1.0",
    description="Alternative Data Alpha Generation Platform with Explainable Deep Learning",
    author="ML Roadmap Bootcamp",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#") and not line.startswith("git+")
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pytest-cov>=4.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "alpha-platform=alpha_platform.cli:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
