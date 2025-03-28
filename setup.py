from setuptools import setup, find_packages

setup(
    name="stock_prediction_STA410",
    version="0.1.4",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "yfinance",
        "scikit-learn",
        "statsmodels",
        "xgboost",
        "lightgbm",
        "catboost",
        "matplotlib",
        "seaborn",
        "pandas-market-calendars",
        "pandas_market_calendars",
        "mplcursors",
        "scikit-optimize",
        "hmmlearn",
        "pmdarima"




    ],
)


