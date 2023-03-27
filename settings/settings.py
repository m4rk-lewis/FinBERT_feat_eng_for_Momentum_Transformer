import os

DATASET_OUTPUT_FOLDER = os.path.join(f"dataset")

SPARSE_PORTFOLIO = [
    # Europe
    "E6_continuous_adjusted_",  # Euro Fx
    "FX_continuous_adjusted_",  # Euro STOXX 50
    # UK
    "B6_continuous_adjusted_", # British Pound
    "X_continuous_adjusted_",  # FTSE 100
    # US
    "ES_continuous_adjusted_",  # E-Mini S&P 500
    "NQ_continuous_adjusted_",  # E-Mini Nasdaq-100
    # Com
    "CA_continuous_adjusted_",  # Cocoa Futures
    "CL_continuous_adjusted_"  # Crude Oil WTI
]


SINGLE_TICKER = [
    # Stocks Indices:
    ##--> USA
    "ES_continuous_adjusted_",  # E-Mini S&P 500
    # "CL_continuous_adjusted_"  # Crude Oil WTI
]



DUAL_TICKER = [
    # Stocks Indices:
    ##--> USA
    "ES_continuous_adjusted_",  # E-Mini S&P 500
    "NQ_continuous_adjusted_",  # E-Mini Nasdaq-100
]



H1_TICKERS = [
    # FX:
    "A6_continuous_adjusted_",  # Australian Dollar
    "N6_continuous_adjusted_",  # Australian Dollar
    "E6_continuous_adjusted_",  # Euro Fx
    "B6_continuous_adjusted_",  # British Pound
    "AD_continuous_adjusted_",  # Canadian Dollar
    "E1_continuous_adjusted_",  # Swiss Franc
    "J1_continuous_adjusted_",  # Japanese Yen
    "DX_continuous_adjusted_",  # Us Dollar Index Future

    # Metals
    "GC_continuous_adjusted_",  # Gold Future
    "SI_continuous_adjusted_",  # Silver Future

    # Stocks Indices:
    ##--> USA
    "ES_continuous_adjusted_",  # E-Mini S&P 500
    "NQ_continuous_adjusted_",  # E-Mini Nasdaq-100
    "RTY_continuous_adjusted_", # E-Mini Russell 2000
    "YM_continuous_adjusted_",  # E-Mini Dow Mini
    ##--> Europe
    "DY_continuous_adjusted_",  # Dax
    "FX_continuous_adjusted_",  # Euro STOXX 50
    ##--> Japan
    "NKD_continuous_adjusted_",  # Nikkei 225 Dollar
    ##--> UK
    "X_continuous_adjusted_",  # FTSE 100

    # Softs :
    "CA_continuous_adjusted_",  # Cocoa Futures
    "KC_continuous_adjusted_",  # Coffee Futures
    "SB_continuous_adjusted_",  # Sugar #11
    "XC_continuous_adjusted_",  # Corn Mini

    #  Coms :
    "CL_continuous_adjusted_",  # Crude Oil WTI
    "HH_continuous_adjusted_",  # Natural Gas
]


