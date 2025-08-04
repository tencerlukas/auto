"""Setup configuration for auto package"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="auto",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python automation framework for streamlining repetitive tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tencerlukas/auto",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.0.0",
        ],
        "optional": [
            "requests>=2.27.0",
            "schedule>=1.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "auto=auto.cli:main",
        ],
    },
)