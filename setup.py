from setuptools import setup, find_packages

setup(
    name="bigquerycostopt",
    version="0.1.0",
    description="BigQuery Cost Intelligence Engine",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "google-cloud-bigquery>=2.30.0",
        "google-cloud-pubsub>=2.8.0",
        "google-cloud-storage>=2.0.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "flask>=2.0.0",
        "gunicorn>=20.1.0",
        "requests>=2.25.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.1.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
        "prophet>=1.1.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
            "isort>=5.9.0"
        ],
        "test": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0"
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
)
