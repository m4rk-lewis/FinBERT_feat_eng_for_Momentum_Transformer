# FinBERT Feature Engineering for TFT and LSTM Momentum Transformer


## Creating the FinBERT TFT dataset and running it on the Momentum Transformer trading model
1. Open 'forexlive_scraper.ipynb' within Google Colab. Run All Cells for all desired years (some are subdivided into months)
2. place sent_forexlive_2021.csv type labelled files into 'ForexLive' '4_Finbert' folders within the repo
3. Place Futures Data into the Futures folder (1hour)
4. Place a downloaded CSV of the economic callendar into teh callendar folder
5. Run feature_engineer.py using the daily timeframe option
6. Run  export_dataset_to_mom_trans.py
7. Use command line arguments shown in the excel spreadsheet Experiemnt log to recreate the experiments




===========================================================================




## The files and folder within this repository can be placced inside the [Momentum Transforer](https://github.com/kieranjwood/trading-momentum-transformer) repository by Keiran Wood to allow comparison of feature engineering performance. Overwrite folders with the same names to transfer modifications over.
1. 'forexlive_scraper.ipynb'
2. 'feature_engineer.py'
3. 'export_dataset_to_mom_trans.py'
4. 'settings.py' within the 'settings' folder
5. folder: callendar (all items)
6. folder: futures (this contains purchased intraday futures data from Firstrate.com)
7. folder: dataset (all items)
8. folder: forexlive (all items)
9. folder: twitter  (this contains data scraped using Optimized-Modified-GetOldTweets3-OMGOT)



## RUNNING MOMENTUM TRANSFORMER EXPIRENTS AS PER WOOD's ORIGINAL IMPLIMENTATION
Note: 
- the sharpe ratio settings must be set to 252 periods
- the original expanding window settings must be used that utilise year, rather than rolloing quarters (which are misslabelled as year to avoid further modifications). simple unblock the orignal lines of code and block out the Finbert_TFT lines of code.

1. Create a Nasdaq Data Link account to access the [free Quandl dataset](https://data.nasdaq.com/data/CHRIS-wiki-continuous-futures/documentation). This dataset provides continuous contracts for 600+ futures, built on top of raw data from CME, ICE, LIFFE etc.
2. Download the Quandl data with: `python -m data.download_quandl_data <<API_KEY>>`
3. Create Momentum Transformer input features with: `python -m examples.create_features_quandl`. In this example we use the 100 futures tickers which have i) the longest history ii) more than 90% of trading days have data iii) data up until at least Dec 2021.
4. Optionally, run the changepoint detection module: `python -m examples.concurent_cpd_quandl <<CPD_WINDOW_LENGTH>>`, for example `python -m examples.concurent_cpd_quandl 21` and `python -m examples.concurent_cpd_quandl 126`
5. Create Momentum Transformer input features, including CPD module features with: `python -m examples.create_features_quandl 21` after the changepoint detection module has completed.
6. To create a features file with multiple changepoint detection lookback windows: `python -m examples.create_features_quandl 126 21` after the 126 day LBW changepoint detection module has completed and a features file for the 21 day LBW exists.
7. Run one of the Momentum Transformer or Slow Momentum with Fast Reversion experiments with `python -m examples.run_dmn_experiment <<EXPERIMENT_NAME>>`




## TESTING THE FinBERT TFT dataset on the Momentum Transformer backtesting engine
1. The twitter data was scraped using the Optimized-Modified-GetOldTweets3-OMGOT repository by 
2. Download the Quandl data with: `python -m data.download_quandl_data <<API_KEY>>`
3. Create Momentum Transformer input features with: `python -m examples.create_features_quandl`. In this example we use the 100 futures tickers which have i) the longest history ii) more than 90% of trading days have data iii) data up until at least Dec 2021.
4. Optionally, run the changepoint detection module: `python -m examples.concurent_cpd_quandl <<CPD_WINDOW_LENGTH>>`, for example `python -m examples.concurent_cpd_quandl 21` and `python -m examples.concurent_cpd_quandl 126`
5. Create Momentum Transformer input features, including CPD module features with: `python -m examples.create_features_quandl 21` after the changepoint detection module has completed.
6. To create a features file with multiple changepoint detection lookback windows: `python -m examples.create_features_quandl 126 21` after the 126 day LBW changepoint detection module has completed and a features file for the 21 day LBW exists.
7. Run one of the Momentum Transformer or Slow Momentum with Fast Reversion experiments with `python -m examples.run_dmn_experiment <<EXPERIMENT_NAME>>`




## References
The FinBERT TFT uses a number of components from the Momentum Transformer. The code for the Momentum Transformer can be found [here](https://github.com/google-research/google-research/tree/master/tft).

The FinBERT TFT uses scraped data from multiple sources. Since 2009, Forexlive have posted a link to every breaking news article  can be found [here](https://github.com/google-research/google-research/tree/master/tft).

The Momentum Transformer uses a number of components from the Temporal Fusion Transformer (TFT). The code for the TFT can be found [here](https://github.com/kieranjwood/trading-momentum-transformer).

