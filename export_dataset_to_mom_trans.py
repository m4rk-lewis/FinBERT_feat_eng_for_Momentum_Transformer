import os
import numpy as np
import pandas as pd
from sklearn import preprocessing

# set pandas display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)



def import_data(folder: str, subfolder: str, ticker: str, new_ticker: str,  options: str, options2: str, options3: str, timeframe: str):
    """ This funhctions inports the FinBERT dataset, and converts the names of each column to names that the
    Momentum Transformer Model expected to see. This prevents modification of the model and allows testing.
    the only change is that.

    :param folder:
    :param subfolder:
    :param ticker:
    :param test_q:
    :return:
    """
    directory = os.path.join(f"{folder}", f"{subfolder}", str(ticker)+'.csv')
    df = pd.read_csv(os.path.join(directory), low_memory=False)[[
                                                                'timestamp_5day',
                                                                # 'timestamp',
                                                                # 'open',
                                                                # 'high',
                                                                # 'low',
                                                                'close',
                                                                'volume',
                                                                'ticker',
                                                                # 'ema_50',
                                                                # 'vwap_D',
                                                                # 'vwap_W',
                                                                # 'vwap_M',
                                                                'ny_hour',
                                                                'day_of_week',
                                                                'day_of_month',
                                                                'month_of_year',
                                                                'year',
                                                                'quarter',
                                                                # 'q',
                                                                'quarter_roll',
                                                                'bar_return',
                                                                'rsi_10',
                                                                'rsi_15',
                                                                'rsi_30',
                                                                'macd_8_24_16',
                                                                'macd_16_48_32',
                                                                'macd_32_96_64',
                                                                'vwap_D_dist',
                                                                'ema_50_dist',
                                                                'vwap_W_dist',
                                                                'vwap_M_dist',
                                                                'pin_1',
                                                                'twobar_pin_1',
                                                                'threebar_pin_1',
                                                                'fourbar_pin_1',
                                                                'sixbar_pin_1',
                                                                'target_return',
                                                                'target_return_2',
                                                                'target_return_3',
                                                                # 'target_return_4',
                                                                # 'target_return_5',
                                                                # 'target_return_6',
                                                                # 'target_return_7',
                                                                # 'target_return_8',
                                                                'pos_sent',
                                                                'neg_sent',
                                                                'neu_sent',
                                                                'finbert_sent_osc',
                                                                'finbert_sent_osc_day',
                                                                'ALL_econ_cal',
                                                                'ALL_econ_cal_lab'
                                                                ]] # date is placed onto the 'index' column when made

    df['ticker'] = new_ticker

    # slice the whole dataset to just 2012 to 2022
    if timeframe == 'daily':
        start_date = '2011-09-05'  # this gives one quarter of lead in before the training starts
        start_index = df.loc[df['timestamp_5day'] == start_date].index.values[0]
        end_date = '2022-06-30'
        end_index = df.loc[df['timestamp_5day'] == end_date].index.values[0]
        df = df[start_index : end_index+1].copy()
    else:
        start_date = '2012-01-03 13:00:00+02:00'
        start_index = df.loc[df['timestamp_5day'] == start_date].index.values[0]
        end_date = '2022-06-30 23:00:00+03:00'
        end_index = df.loc[df['timestamp_5day'] == end_date].index.values[0]
        df = df[start_index : end_index+1].copy()

    # print('start_date', start_date, 'end_date', end_date)
    # print('start',start_index,'end', end_index + 1)

    # this should set the index to 0 on the first bar of 2011, then it will increase by one every bar
    # df.reset_index(inplace=True)
    # try:
    #     df["index"] = df["index"] - df.loc[df['timestamp_5day'] == '2011-01-03 13:00:00+02:00', 'index'].item()
    # except:
    #     '2012-01-03 13:00:00+02:00 is not contained within the dataset. time_idx not created'



    """ Convert News Calendar Data into encoded labels:
    We do not need to worry about transforming the whole dataset and leaking data from the future because we have 
    pre-selected a fixed list of economic events to be included. These do not change throughout the whole dataset
    therefore we can fit and transform the whole thing now
    """
    df["ALL_econ_cal"] = df.ALL_econ_cal.astype(str)
    df["ALL_econ_cal"] = df.ALL_econ_cal.replace("0", np.nan)
    df["ALL_econ_cal"] = df.ALL_econ_cal.replace('nan', np.nan)
    df["ALL_econ_cal"] = df.ALL_econ_cal.replace('', np.nan)

    # LABEL ENCODING FOR NEWS CALLENDAR
    le = preprocessing.LabelEncoder()
    categoricals_ = le.fit(df["ALL_econ_cal"].values)
    df["ALL_econ_cal_lab"] = le.transform(df["ALL_econ_cal"].values)
    # print(df)



    """
    THIS IS NEEDED FOR THE VOLATILITY SCALING THAT IS KEY TO THE DEEP MOMENTUM NETWORK MODEL
    WE WILL SACRIFICE THE TRADE VOLUME METRIC FOR IT
    """
    df["volume"] = df['bar_return'].ewm(span=60, min_periods=60).std().fillna(method="bfill")



    """ this is to tell the momentum transformer that the our features are named the same as the ones the mom transformer 
    expects to see in quandl_cpd_0lbw.csv / quandl_cpd_1lbw.csv / quandl_cpd_2lbw.csv
    By specifying these CPD numbers (0, 1, 2, OR ANY COMBINATION OF THOSE) this tells the model where to find the 
    correct csv to get the dataset from to run the model. This can be verified by checking the fixed_params json file
    in the results folders. eg: "features_file_path": "data\\quandl_cpd_1lbw.csv" = finbert +news dataset with the 
    sentiment oscilators. "data\\quandl_cpd_26lbw.csv" and "data\\quandl_cpd_126lbw.csv" are both the original momentum 
    transformer dataset
    """
    if timeframe != 'daily':
        df['index'] = range(0, len(df))
        df.rename(columns={'index': 'date'}, inplace=True)
        df.rename(columns={'ny_hour': 'week_of_year'}, inplace=True)
        df.drop(['year'], axis=1, inplace=True)
        df.rename(columns={'quarter_roll': 'year'}, inplace=True)
    else:
        df.rename(columns={'ny_hour': 'date'}, inplace=True)
        df.rename(columns={'quarter': 'week_of_year'}, inplace=True)
    df.rename(columns={'timestamp_5day': 'Date'}, inplace=True)
    ##########################################################################
    # df.rename(columns={'close': 'close'}, inplace=True)
    df['srs'] = df['close']
    df.rename(columns={'volume': 'daily_vol'}, inplace=True)
    # df.rename(columns={'ticker': 'ticker'}, inplace=True)
    # df.rename(columns={'day_of_week': 'day_of_week'}, inplace=True)
    # df.rename(columns={'day_of_month': 'day_of_month'}, inplace=True)
    df.rename(columns={'bar_return': 'daily_returns'}, inplace=True)
    ##########################################################################
    df.rename(columns={'macd_8_24_16': 'macd_8_24'}, inplace=True)
    if options == 'macd':
        df.rename(columns={'macd_16_48_32': 'macd_16_48'}, inplace=True)
        df.rename(columns={'macd_32_96_64': 'macd_32_96'}, inplace=True)
    elif options == 'rsi':
        df.rename(columns={'rsi_15': 'macd_16_48'}, inplace=True)
        df.rename(columns={'twobar_pin_1': 'macd_32_96'}, inplace=True)
    ##########################################################################
    df.rename(columns={'vwap_D_dist': 'norm_daily_return'}, inplace=True)
    df.rename(columns={'ema_50_dist': 'norm_monthly_return'}, inplace=True)
    df.rename(columns={'vwap_W_dist': 'norm_quarterly_return'}, inplace=True)
    df.rename(columns={'vwap_M_dist': 'norm_biannual_return'}, inplace=True)
    ##########################################################################
    df.rename(columns={'pin_1': 'norm_annual_return'}, inplace=True)
    ##########################################################################
    # if options2 == 'pin_1':
    #     )
    # elif options2 == 'twobar_pin_1':
    #     df.rename(columns={'twobar_pin_1': 'norm_annual_return'}, inplace=True)
    # elif options2 == 'threebar_pin_1':
    #     df.rename(columns={'threebar_pin_1': 'norm_annual_return'}, inplace=True)
    # elif options2 == 'fourbar_pin_1':
    #     df.rename(columns={'fourbar_pin_1': 'norm_annual_return'}, inplace=True)
    # elif options2 == 'sixbar_pin_1':
    #     df.rename(columns={'sixbar_pin_1': 'norm_annual_return'}, inplace=True)
    ##########################################################################
    df.rename(columns={'pos_sent': 'cp_rl_0'}, inplace=True)
    df.rename(columns={'neg_sent': 'cp_score_0'}, inplace=True)
    df.rename(columns={'finbert_sent_osc': 'cp_rl_1'}, inplace=True)
    df.rename(columns={'finbert_sent_osc_day': 'cp_score_1'}, inplace=True)
    ##########################################################################
    # df['month_of_year'] = df['ALL_econ_cal_lab'].astype(str)
    df.rename(columns={'ALL_econ_cal_lab': 'month_of_year'}, inplace=True)
    ##########################################################################
    df.rename(columns={'target_return': 'target_returns'}, inplace=True)
    if options3 == 'rsi':
        df['cp_rl_2'] = df['rsi_10']
        df['cp_score_2'] = df['rsi_30']
    elif options3 == 'pins':
        df['cp_rl_2'] = df['twobar_pin_1']
        df['cp_score_2'] = df['fourbar_pin_1']
    elif options3 == 'both':
        df['cp_rl_2'] = df['twobar_pin_1']
        df['cp_score_2'] = df['rsi_30']


    df = df[[                       # mapped to the following finbert dataset features:
        'Date',                     # Timestamp_5day
        'date',                     # index / ny_hour [depending on if daily or hourly]
        'ticker',                   # ticker
        'close',                    # close (this is redundant but likely still needed for calculating returns)
        'srs',                      # close
        'daily_returns',            # bar_return
        'daily_vol',                # volume - note this was volatility not volume
        'target_returns',           # target_return
        'norm_daily_return',        # vwap_D_dist
        'norm_monthly_return',      # ema_50_dist
        'norm_quarterly_return',    # vwap_W_dist
        'norm_biannual_return',     # vwap_M_dist
        'norm_annual_return',       # pin_1
        'macd_8_24',                # macd_8_24_16
        'macd_16_48',               # macd_16_48_32 / rsi_15                                                   [Option1]
        'macd_32_96',               # macd_32_96_64 / twobar_pin_1                                             [Option1]
        'day_of_week',              # day_of_week - not included when TIME_FEATURES = False
        'day_of_month',             # day_of_month - not included when TIME_FEATURES = False
        'week_of_year',             # ny_hour - not included when TIME_FEATURES = False
        'month_of_year',            # ALL_econ_cal_lab  - not included when TIME_FEATURES = False   <<<<<<<<<<<<<<<[NewsCal]
        'year',                     # year [daily] / quarter_roll [1hour]  - not included when TIME_FEATURES = False
        'cp_rl_0',                  # pos_sent <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[FinBERT]
        'cp_score_0',               # neg_sent <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[FinBERT]
        'cp_rl_1',                  # finbert_sent_osc <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[FinBERT]
        'cp_score_1',               # finbert_sent_osc_day <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<[FinBERT]
        'cp_rl_2',                  # rsi_10 /  twobar_pin_1                                                   [Option3]
        'cp_score_2',               # rsi_30 /  fourbar_pin_1                                                  [Option3]
    ]]
    df.set_index('Date', inplace=True)

    # create folder within 'dataset' to place dataset. csv
    folder = 'data'
    directory = os.path.join(f"{folder}")
    df.to_csv(f"{directory}/quandl_cpd_0lbw.csv")
    df.to_csv(f"{directory}/quandl_cpd_1lbw.csv")
    df.to_csv(f"{directory}/quandl_cpd_2lbw.csv")

    print(df.head(25))
    print(df.sample(25))
    print(df.tail(25))
    return(df)



########################################################################################################################
def main():
    """ OPTIONS #######################
    option 1:
        macd
        rsi
    option 3:
        rsi           # rsi_10 and rsi_30
        pins          # twobar_pin_1 and fourbar_pin_1
        both          # twobar_pin_1 and rsi_30
    """
    # feature_settings = '1hour_50_10_15_30_8_24_16_16_48_32_32_96_64_1_1_1_1_1_H_24'
    feature_settings = 'daily_50_10_15_30_8_24_16_16_48_32_32_96_64_1_1_1_1_1_D_5'

    import_data('dataset', feature_settings,
                'ES',                           # dataset Ticker
                'CME_ES',                       # Ticker expected by Model
                'macd',                         # Option 1 : macd = macd_16_48_32 and macd_32_96_64 / rsi = rsi_15 and twobar_pin_1
                'pin_1',                        # Option 2 : removed
                'both',                         # Option 3 : rsi = rsi_10 and rsi_30 / pins = twobar_pin_1 and fourbar_pin_1 / both = twobar_pin_1 and rsi_30
                feature_settings.split('_')[0]
                )

########################################################################################################################
if __name__ == "__main__":
    main()


