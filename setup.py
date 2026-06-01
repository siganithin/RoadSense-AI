"""
setup.py — Package setup for RoadSense AI
Allows: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="roadsense-ai",
    version="2.0.0",
    author="RoadSense AI Team",
    description="AI-Powered Road Damage Detection using EfficientNetB0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/CNN_road_damage",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    entry_points={
        "console_scripts": [
            "roadsense-train=train:main",
            "roadsense-predict=src.predict:main",
            "roadsense-evaluate=src.evaluate:evaluate_model",
        ],
    },
)
