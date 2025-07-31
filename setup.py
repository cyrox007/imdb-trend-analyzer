from setuptools import setup, find_packages

setup(
    name="kinovanga",
    version="0.1.0",
    description="КиноВанга XD — нейросеть-кинокритик, предсказывающая рейтинг фильма",
    author="JSInteractive",
    author_email="email@example.com",
    url="https://github.com/cyrox007/imdb-trend-analyzer",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)