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
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "flask>=2.0.0",
        "gunicorn>=20.1.0",
        "requests>=2.25.0"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
)
