"""
Setup script for AstraPro Multi-Target Tracking System
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="astraPro",
    version="1.0.0",
    author="AstraPro Development Team",
    description="A real-time multi-sensor tracking system with Kalman filtering and data fusion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/astraPro",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "viz": ["matplotlib", "seaborn"],
        "analysis": ["pandas", "matplotlib", "seaborn", "scipy"],
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "astraPro=scripts.main:main",
            "astraPro-analyze=scripts.analyze_tracking:main",
        ],
    },
    scripts=[
        "scripts/main.py",
        "scripts/analyze_tracking.py",
    ],
    include_package_data=True,
    package_data={
        "astraPro": ["*.yaml", "*.yml"],
    },
    zip_safe=False,
)