# Financial Bidirectional Encoder Representations from Transformers (FinBERT) and Temporal Fusion Transformer (TFT) for Stock Market Prediction

#### The files and folder within this repository can be placed inside the [Momentum Transforer](https://github.com/kieranjwood/trading-momentum-transformer) repository by Keiran Wood to allow comparison of feature engineering performance. Overwrite folders with the same names to transfer modifications over.
1. 'forexlive_scraper.ipynb'
2. 'feature_engineer.py'
3. 'export_dataset_to_mom_trans.py'
4. overwrite the 'settings' folder from this repo into the mom trans repo
5. overwrite the 'mom_trans' folder from this repo into the mom trans repo
6. overwrite the 'examples' folder from this repo into the mom trans repo
7. 'futures' folder - add purchased intraday futures data (continuous 1hour) from [firstrate](https://firstratedata.com/it/futures)
8. 'dataset' folder - this is where export_dataset_to_mom_trans.py will export the new dataset to 
9. 'forexlive' folder - this contains the sentiment data for every breaking news article needed for feature_engineer.py
10. 'twitter' folder - this contains data scraped from twitter using Optimized-Modified-GetOldTweets3-OMGOT. forexlive_scraper.ipynb uses this data to extract the URL of every web page containing the news articles


===========================================================================
## Modified code from [Momentum Transforer](https://github.com/kieranjwood/trading-momentum-transformer) repo
The 'examples', 'settings' and the 'mom_trans' folders contains code from the original momentum transformer repo by Wood. They are largely unmodified, with the exception that run_dmn_experiment.py, default.py, settings.py, run_classical_strategies.py and model_inputs.py which have been minimally modified to accept new experiments for the new feature set and to run for a reduced training perdiod to match the start of the alternative features. 

The experiments are also no longer multi-instrument portfolio tests, and rather focus only on $ES_F (S&P 500) futures.

===========================================================================
## Creating the FinBERT TFT dataset and running it on the Momentum Transformer trading model
1. Place this entire repo inside the momentum transforer repo, overwriting folders with the same names
2. Open 'forexlive_scraper.ipynb' within Google Colab. Run All Cells for all desired years (some are subdivided into months)
3. Place sent_forexlive_2021.csv type labelled files into 'ForexLive', then '4_Finbert' folders within the repo
4. Place Futures Data into the Futures folder (1hour)
5. Run feature_engineer.py using the daily timeframe option
6. Run  export_dataset_to_mom_trans.py
7. Use command line arguments shown in the excel spreadsheet Experiemnt log to recreate the experiments. eg. `python -m examples.run_dmn_experiment FinBERT-LSTM-63-1`				

===========================================================================

<p align="center">
  <img src="https://github.com/m4rk-lewis/FinBERT_Feat_Eng_for_MOMENTUM_TRANFORMER/blob/main/pics/MomTransArch.png" width="350" title="Momentum Transformer Architechture [1]">
    <img src="https://github.com/m4rk-lewis/FinBERT_Feat_Eng_for_MOMENTUM_TRANFORMER/blob/main/pics/Experiment Flow.png" width="350" title="Experiment Flow Chart">
</p>

===========================================================================
## RUNNING MOMENTUM TRANSFORMER EXPIRENTS AS PER WOOD's ORIGINAL IMPLIMENTATION
1. Create a Nasdaq Data Link account to access the [free Quandl dataset](https://data.nasdaq.com/data/CHRIS-wiki-continuous-futures/documentation). This dataset provides continuous contracts for 600+ futures, built on top of raw data from CME, ICE, LIFFE etc.
2. Download the Quandl data with: `python -m data.download_quandl_data <<API_KEY>>`
3. Create Momentum Transformer input features with: `python -m examples.create_features_quandl`. In this example we use the 100 futures tickers which have i) the longest history ii) more than 90% of trading days have data iii) data up until at least Dec 2021.
4. Optionally, run the changepoint detection module: `python -m examples.concurent_cpd_quandl <<CPD_WINDOW_LENGTH>>`, for example `python -m examples.concurent_cpd_quandl 21` and `python -m examples.concurent_cpd_quandl 126`
5. Create Momentum Transformer input features, including CPD module features with: `python -m examples.create_features_quandl 21` after the changepoint detection module has completed.
6. To create a features file with multiple changepoint detection lookback windows: `python -m examples.create_features_quandl 126 21` after the 126 day LBW changepoint detection module has completed and a features file for the 21 day LBW exists.
7. Run one of the Momentum Transformer or Slow Momentum with Fast Reversion experiments with `python -m examples.run_dmn_experiment <<EXPERIMENT_NAME>>`



===========================================================================
## New Features
Fig 3. VWAP Proximity Oscilators - Daily timeframe:
<p align="center">
  <img src="https://github.com/m4rk-lewis/FinBERT_Feat_Eng_for_MOMENTUM_TRANFORMER/blob/main/pics/VWAP Dist Oscilators.png" width="1000" title="VWAP proximity Oscilators">
</p>



Fig 4. Pin Bar Oscilators - Daily timeframe:
<p align="center">
  <img src="https://github.com/m4rk-lewis/FinBERT_Feat_Eng_for_MOMENTUM_TRANFORMER/blob/main/pics/Pin Bar Oscliators.png" width="1000" title="Pin Bar Oscliators">
</p>



Fig 5. Finbert Sentiment Oscilators - Daily timeframe:
<p align="center">
  <img src="https://github.com/m4rk-lewis/FinBERT_Feat_Eng_for_MOMENTUM_TRANFORMER/blob/main/pics/FinBERT oscliators.png" width="1000" title="Finbert Sentiment Oscilators 1">
</p>



Fig 6. Finbert Sentiment Oscilators - 1 Hour timeframe:
<p align="center">
  <img src="https://github.com/m4rk-lewis/FinBERT_Feat_Eng_for_MOMENTUM_TRANFORMER/blob/main/pics/FinBERT oscliators 2.png" width="1000" title="Finbert Sentiment Oscilators 2">
</p>


Fig 7. Finbert Sentiment Oscilators - 1 hour timeframe:
<p align="center">
  <img src="https://github.com/m4rk-lewis/FinBERT_Feat_Eng_for_MOMENTUM_TRANFORMER/blob/main/pics/FinBERT oscliators 3.png" width="1000" title="Finbert Sentiment Oscilators 3 - hourly">
</p>


Fig 8. New Feature Map (per experiment):
<p align="center">
  <img src="https://github.com/m4rk-lewis/FinBERT_Feat_Eng_for_MOMENTUM_TRANFORMER/blob/main/pics/Proposed Improved Feature Set.jpg" width="700" title="Proposed Improved Feature Set">
</p>

===========================================================================
## Training
The experiments were conducted by shifting the intraday data to daily timeframe data. News callendar data, futures price data and breaking news data were all synchronised to Helsinki time (GMT+2/3), to adhere to 'New York Close' charting principles and DST shift schedule. This ensures only five daily candles per week rather than including a small Sunday candle.

Fig 9. Optimiised Hyperparaeters per year:
<p align="center">
  <img src="https://github.com/m4rk-lewis/FinBERT_Feat_Eng_for_MOMENTUM_TRANFORMER/blob/main/pics/optiised parameter.png" width="700" title="optiised parameters" >
</p>



Fig 10. Expanding Windown Training Regime:
<p align="center">
  <img src="https://github.com/m4rk-lewis/FinBERT_Feat_Eng_for_MOMENTUM_TRANFORMER/blob/main/pics/train test split.jpg" width="700" title="train test split" >
</p>


===========================================================================
## Results
Fig 11. Original Momentum Transformer Feature Set Performance:
<p align="center">
  <img src="https://github.com/m4rk-lewis/FinBERT_Feat_Eng_for_MOMENTUM_TRANFORMER/blob/main/pics/Original Features.jpg" width="1000" title="Original Features" >
</p>

Fig 12. Improved Feature Set Performance:
<p align="center">
  <img src="https://github.com/m4rk-lewis/FinBERT_Feat_Eng_for_MOMENTUM_TRANFORMER/blob/main/pics/results.jpg" width="1000" title="Improved Feature Set Performance" >
</p>

Fig 13. LSTM vs TFT Momentum Transformer Performance Comparison:
<p align="center">
  <img src="https://github.com/m4rk-lewis/FinBERT_Feat_Eng_for_MOMENTUM_TRANFORMER/blob/main/pics/LSTM vs TFT.jpg" width="600" title="LSTM vs TFT" >
</p>
Visualizing our key performance metric, SoAS ratio for each experiment and ranking them each, we are able to see the relative performance of experiments with each model type, then take a mean average of rank and SoAS ratio per model type.
 
LSTM DMN has outperformed TFT DMN with a mean rank of 9.1 vs 12.6 (lower indicating a better rank). The mean SoAS ratio is also higher for LSTM models at 0.66 vs 0.43 (higher is better). Visually we can also see that the red TFT experiments show correlation with the darker colored worse rank numbers, to confirm the result.



===========================================================================
## Conclusion
Sizeable improvements have been seen in annualized Sharpe ratio, cumulative percentage return and Sharpe of Annualized Sharpe ratios following the supplementation of a new feature set. While inconclusive, it could suggest that the reduction in performance in recent years of LSTM based trading strategies is partly due to model complexity, in particular the modelling of long-term dependencies, it also suggests that there was simply a regime change in terms of feature set use within the institutional trading industry, with more of a focus shifting towards alternative sentiment based features, rather than pure price based features.

Decreasing model training time does decrease model performance, but this was more than compensated for by the improvements from using a new feature set. 

The proposed non-sentiment-based features outperformed the original features, indicating that VWAP distance is a valuable metric for timeseries prediction of stock index futures, and the inclusion of known future covariates in the form of a label encoded economic calendar data has also improved the performance of the trading models.

One unusual outcome of these experiments is the unexpected outperformance of LSTM vs TFT. TFT can more accurately model longer-range dependencies in the data because of its attention mechanism, but the inclusion of non-lagging features such as pin_bar oscillators and vwap_distance oscillators may have highlighted the LSTM model’s greater ability to learn near term dependencies, which may have proved more valuable to the prediction algorithms. 



===========================================================================
## References

[1] The FinBERT TFT uses a number of components from the Momentum Transformer. The code for the Momentum Transformer can be found [here](https://github.com/kieranjwood/trading-momentum-transformer).

[2] The Momentum Transformer uses a number of components from the Temporal Fusion Transformer (TFT). The code for the TFT can be found [here](https://github.com/google-research/google-research/tree/master/tft).


@article{wood2021trading,
  title={Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture},
  author={Wood, Kieran and Giegerich, Sven and Roberts, Stephen and Zohren, Stefan},
  journal={arXiv preprint arXiv:2112.08534},
  year={2021}
}






