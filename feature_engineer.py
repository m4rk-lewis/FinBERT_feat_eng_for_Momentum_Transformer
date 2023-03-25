import pandas as pd
import numpy as np
import os
from settings.settings import DUAL_TICKER, SINGLE_TICKER, SPARSE_PORTFOLIO
import finplot as fplt
from dateutil import tz
from datetime import datetime
from tqdm import tqdm
from sklearn import preprocessing
import argparse

# set pandas display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)


def txt_to_csv(folder: str, timeframe: str):
    """ The futures data comes as .txt files, even though they are stored as .csv data, so we need to rename the files
    :param folder: the directory of the futures folder
    :param timeframe: directory of the timeframe folder
    :return: simple text confirmation
    """
    directory = os.path.join(f"{folder}", f"{timeframe}")
    for f in os.listdir(directory):
        if f.endswith('.txt'):
            fullpath = os.path.join(directory, f)
            print(fullpath, fullpath[:-4] + f"{timeframe}" + '.csv')
            os.rename(fullpath, fullpath[:-4] + f"{timeframe}" + '.csv')
    print('txt_to_csv ', f"{timeframe}", 'conversion is complete')



def create_dataset(folder: str, timeframe: str, ticker: str) -> pd.DataFrame:
    """ create dataset from raw futures data http://firstratedata.com/about/price_adjustment#futures
    - Timezone is US Eastern Time
    - Timestamps run from the start of the period (eg 1min bars stamped 09:30 run from 09:30.00 to 09:30.59)
    - Volume Numbers are in individual contracts
    - Times with zero volume are omitted (thus gaps in the data sequence are when there have been no trades)
    :param folder:
    :param timeframe:
    :param portfolio:
    :return: df
    """
    # for ticker in portfolio:
    directory = os.path.join(f"{folder}", f"{timeframe}", str(ticker) + f"{timeframe}" + '.csv')
    df = pd.read_csv(os.path.join(directory), parse_dates=[0])
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df['ticker'] = ticker.split('_')[0]
    df['timestamp_5day'] = pd.to_datetime(df['timestamp'], format="%d%m%Y%H%M%S")  # 06/09/2005  16:00:00
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%d%m%Y%H%M%S")  # 06/09/2005  16:00:00
    print('create_dataset ', f"{timeframe}", f"{ticker}", ' is complete')
    return (df)



def df_timezone_convert(df, native_tz: str, target_tz: str):
    """
        ################################################################################################################
        https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
        ################################################################################################################

        'America/New_York' =  EDT(GMT-5) and EST(GMT-4) switching second Sunday in March / first Sunday in November
            The futures price data is in this timezine
                EST: (winter)
                    Eastern Standard Time (EST), when observing standard time (autumn/winter)
                    five hours behind Coordinated Universal Time (UTC−05:00).
                EDT: (summer)
                    Eastern Daylight Time (EDT), when observing daylight saving time (spring/summer)
                    four hours behind Coordinated Universal Time (UTC−04:00).

        ################################################################################################################

        'Europe/London' = GMT and DST(GMT+1) switching last Sunday of March / last Sunday of October
            The news, FX and calendar data are in this timezone
                GMT: (winter)
                    Greenwich Mean Time is defined in law as standard time in the United Kingdom, where the summer
                    time is called British Summer Time (BST)
                BST: (summer)
                    BST begins at 01:00 GMT every year on the last Sunday of March and ends at 01:00 GMT (02:00 BST) on the
                    last Sunday of October. The starting and finishing times of daylight saving were aligned across the European
                    Union on 22 October 1995, and the UK retained this alignment after it left the EU

        ################################################################################################################

        'Europe/Helsinki' = EET(GMT+2) and EST(GMT+3) switching last Sunday of March / last Sunday of October
            We need everything to be in this timezone so that there are only 5 daily candles per week (no Sunday candles / London DST switching)
                EET: (winter)
                    Eastern European Time (EET) is one of the names of UTC+02:00 time zone, 2 hours ahead of Coordinated
                    Universal Time. The zone uses daylight saving time, so that it uses UTC+03:00 during the summer.
                EEST: (summer)
                    Since 1996, Eastern European Summer Time has been observed from the last Sunday in March to the last Sunday in October.
                    Finland, regular EEST since 1981

        ################################################################################################################

        CME Globex : Sunday 6:00 p.m. - Friday - 5:00 p.m. ET with a daily maintenance period from 5:00 p.m. – 6:00 p.m.
        File Format : {DateTime, Open, High, Low, Close, Volume}  Timezone is US Eastern Time

        // CHECK DST SWITCHING //
        US Eastern time 2022 DST switch: 	March 13 / November 6
        16 march 2022 should have NY in DST and london / helsinki not in DST

        INPUT:
            i = 99400    # 2022-03-08 16:00:00 (America/New_York)   = < 2nd Sun, < last Sun = US: -5, LDN: +0, Fin: +2
            i = 99525    # 2022-03-16 03:00:00 (America/New_York)   = > 2nd Sun, < last Sun = US: -4, LDN: +0, Fin: +2
            i = 99759    # 2022-03-30 07:00:00 (America/New_York)   = > 2nd Sun, > last Sun = US: -4, LDN: +1, Fin: +3

        OUTPUT:
            nyc_tz:  2022-03-08 16:00:00-05:00 = winter
            ldn_tz:  2022-03-08 21:00:00+00:00 = winter
            fin_tz:  2022-03-08 23:00:00+02:00 = winter

            nyc_tz:  2022-03-16 03:00:00-04:00 = summer
            ldn_tz:  2022-03-16 07:00:00+00:00 = winter
            fin_tz:  2022-03-16 09:00:00+02:00 = winter

            nyc_tz:  2022-03-30 07:00:00-04:00 = summer
            ldn_tz:  2022-03-30 12:00:00+01:00 = summer
            fin_tz:  2022-03-30 14:00:00+03:00 = summer
    """
    dt_native = df['timestamp_5day'].copy()
    for i in tqdm(range(len(df)), desc="df_timezone_convert"):
        """ iterate through df and replace timestamp_5day with corrected NY_Close timezone
        """
        # nyc_tz = tz.gettz('America/New_York')
        # ldn_tz = tz.gettz('Europe/London')
        # fin_tz = tz.gettz('Europe/Helsinki')

        nat_tz = tz.gettz(native_tz)
        chart_tz = tz.gettz(target_tz)

        dt = datetime(dt_native[i].year,
                      dt_native[i].month,
                      dt_native[i].day,
                      dt_native[i].hour,
                      dt_native[i].minute,
                      dt_native[i].second,
                      tzinfo=nat_tz)
        df.at[i, 'timestamp_5day'] = dt.astimezone(chart_tz)
    df.set_index('timestamp_5day', inplace=True)
    print('df_timezone_convert ', native_tz, ' to ', target_tz, ' is complete')
    return (df)



def resample_dataset_timeframe(df, freq: str):
    df.reset_index(inplace=True)
    df2 = pd.DataFrame()
    df2['high'] = df[['timestamp_5day',  'high']].groupby(pd.Grouper(key='timestamp_5day', axis=0, freq=freq)).max()
    df2['low'] = df[['timestamp_5day',  'low']].groupby(pd.Grouper(key='timestamp_5day', axis=0, freq=freq)).min()
    df2['open'] = df[['timestamp_5day',  'open']].groupby(pd.Grouper(key='timestamp_5day', axis=0, freq=freq)).first()
    df2['close'] = df[['timestamp_5day',  'close']].groupby(pd.Grouper(key='timestamp_5day', axis=0, freq=freq)).last()
    df2['volume'] = df[['timestamp_5day',  'volume']].groupby(pd.Grouper(key='timestamp_5day', axis=0, freq=freq)).sum()
    df2['ticker'] = df[['timestamp_5day',  'ticker']].groupby(pd.Grouper(key='timestamp_5day', axis=0, freq=freq)).first()
    df = df2.dropna()
    return(df)



def plot_candles(df, ax):
    """ plot candlestick chart in main window
    :param df: input dataframe
    :param ax: set chart window
    :return: none
    """
    df[['open', 'close', 'high', 'low']].plot(ax=ax, kind='candle')



def plot_time_divs(df, new_timeframe: str):
    """
    DatetimeIndex(['2010-12-31', '2011-03-31', '2011-06-30', '2011-09-30', '2011-12-31', '2012-03-31', '2012-06-30',
    '2012-09-30', '2012-12-31', '2013-03-31', '2013-06-30', '2013-09-30', '2013-12-31', '2014-03-31', '2014-06-30',
    '2014-09-30', '2014-12-31', '2015-03-31', '2015-06-30', '2015-09-30', '2015-12-31', '2016-03-31', '2016-06-30',
    '2016-09-30', '2016-12-31', '2017-03-31', '2017-06-30', '2017-09-30', '2017-12-31', '2018-03-31', '2018-06-30',
    '2018-09-30', '2018-12-31', '2019-03-31', '2019-06-30', '2019-09-30', '2019-12-31', '2020-03-31', '2020-06-30',
    '2020-09-30', '2020-12-31', '2021-03-31'], dtype='datetime64[ns]', freq='Q-DEC')

    :param df:
    :return:
    """
    if new_timeframe == 'daily':
        df["ny_hour"] = df.index.strftime('%Y-%m-%d') # 1998-09-02 / 1998-09-02 / 1998-09-02
    else:
        df["ny_hour"] = df['timestamp'].dt.hour
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month_of_year"] = df.index.month
    df["year"] = df.index.year.astype(str)
    df['quarter'] = (df.index.month - 1) // 3 + 1
    df['q'] = df["year"].str.cat(df[["quarter"]].astype(str), sep="Q").astype(str)
    df['quarter_roll'] = 0

    pd.options.mode.chained_assignment = None  # default='warn'
    for i in tqdm(range(len(df)), desc="quarter_roll"):
        try:
            df['quarter_roll'][i] = int(
                (pd.to_datetime(df['q'][i]).to_period(freq='Q') - pd.to_datetime('1981Q1').to_period(freq='Q')).n)
        except:
            continue
    return (df)



def plot_volume(df, ax):
    """ simple bar volume indicator
    :param df: input dataframe
    :param ax: set chart window
    :return: none - save output to df
    """
    df[['open', 'close', 'volume']].plot(ax=ax, kind='volume')
    return (df)



def plot_ema(df, period, ax):
    """ exponential moving average
    :param df:
    :param period:
    :param ax:
    :return:
    """
    df.close.ewm(span=period).mean().plot(ax=ax, legend='EMA ' + str(period))
    df['ema_' + str(period)] = df.close.ewm(span=period).mean()
    print('ema ', period, ' is complete')
    return (df)



def bar_return(df, ax):
    """ % return for current close vs previous close
    :param df: input dataframe
    :param ax: set chart window
    :return: none - save output to df
    """
    df['bar_return'] = df.close.pct_change() # * 100 # do not use percent, use fraction
    fplt.volume_ocv(df[['open', 'close', 'bar_return']], ax=ax,
                    colorfunc=fplt.strength_colorfilter)
    """
    THIS IS NEEDED FOR THE VOLATILITY SCALING THAT IS KEY TO THE DEEP MOMENTUM NETWORK MODEL
    WE WILL SACRIFICE THE TRADE VOLUME METRIC FOR IT
    """
    df["daily_vol"] = df.close.pct_change().ewm(span=60, min_periods=60).std().fillna(method="bfill")
    print('bar_return is complete')
    return (df)



def target_return(df):
    """ bar_return for next bars (with lookahead bias!!!!!!)
    :param df: input dataframe
    :return: none - save output to df
    """
    df['target_return'] = df.bar_return.shift(-1)
    df['target_return_2'] = df.bar_return.shift(-2) + df.bar_return.shift(-1)
    df['target_return_3'] = df.bar_return.shift(-3) + df.bar_return.shift(-2) + df.bar_return.shift(-1)
    df['target_return_4'] = df.bar_return.shift(-4) + df.bar_return.shift(-3) + df.bar_return.shift(
        -2) + df.bar_return.shift(-1)
    df['target_return_5'] = df.bar_return.shift(-5) + df.bar_return.shift(-4) + df.bar_return.shift(
        -3) + df.bar_return.shift(-2) + df.bar_return.shift(-1)
    df['target_return_6'] = df.bar_return.shift(-6) + df.bar_return.shift(-5) + df.bar_return.shift(
        -4) + df.bar_return.shift(-3) + df.bar_return.shift(-2) + df.bar_return.shift(-1)
    df['target_return_7'] = df.bar_return.shift(-7) + df.bar_return.shift(-6) + df.bar_return.shift(
        -5) + df.bar_return.shift(-4) + df.bar_return.shift(-3) + df.bar_return.shift(-2) + df.bar_return.shift(-1)
    df['target_return_8'] = df.bar_return.shift(-8) + df.bar_return.shift(-7) + df.bar_return.shift(
        -6) + df.bar_return.shift(-5) + df.bar_return.shift(-4) + df.bar_return.shift(-3) + df.bar_return.shift(
        -2) + df.bar_return.shift(-1)
    print('bar_return is complete')
    return (df)



def plot_rsi(df, period, colour, ax):
    """ relative strength index modified from the finplot examples:
    https://github.com/highfestiva/finplot/blob/master/finplot/examples/analyze.py

    :param df:
    :param period:
    :param colour:
    :param ax:
    :return:
    """
    diff = df.close.diff().values
    gains = diff
    losses = -diff
    with np.errstate(invalid='ignore'):
        gains[(gains < 0) | np.isnan(gains)] = 0.0
        losses[(losses <= 0) | np.isnan(losses)] = 1e-10  # we don't want divide by zero/NaN
    # period = 24
    m = (period - 1) / period
    ni = 1 / period
    g = gains[period] = np.nanmean(gains[:period])
    l = losses[period] = np.nanmean(losses[:period])
    gains[:period] = losses[:period] = np.nan
    for i, v in enumerate(gains[period:], period):
        g = gains[i] = ni * v + m * g
    for i, v in enumerate(losses[period:], period):
        l = losses[i] = ni * v + m * l
    rs = gains / losses
    df['rsi'] = 100 - (100 / (1 + rs))
    df.rsi.plot(ax=ax, legend='RSI ' + str([period]), color=colour)
    fplt.set_y_range(10, 90, ax=ax)
    fplt.add_band(70, 100, ax=ax)
    fplt.add_band(0, 30, ax=ax)
    df.rename(columns={'rsi': 'rsi_' + str(period)}, inplace=True)
    print('rsi ', period, ' is complete')
    return (df)



def plot_pinbar(df, period, ax):
    """ measure the pin rejection
    This will be large when there has been a price reversal or continuation pattern and small when ranging or trending

    args:
        df: dataframe
        period: period of double smoothed EMA. set to 1 if smoothing not needed
        ax: for charting the output
    return:
        none: output saved to saved to df
    """
    lo_wicks = df[['open', 'close']].T.min() - df['low']
    hi_wicks = df['high'] - df[['open', 'close']].T.max()
    bar_range = abs(df['high'] - df['low'])
    df['pin'] = (lo_wicks.ewm(span=period).mean() - hi_wicks.ewm(span=period).mean()) * bar_range  # EMA
    if period > 0:
        df['pin'] = df['pin'].ewm(span=period).mean()  # double smoothed
    fplt.volume_ocv(df[['open', 'close', 'pin']], ax=ax, colorfunc=fplt.strength_colorfilter)
    df.rename(columns={'pin': 'pin_' + str(period)}, inplace=True)
    print('pin ', period, ' is complete')
    return (df)



def plot_twobar_pinbar(df, period, ax):
    """
    combine two candles into one candle, then measure the pin rejection of all bars
    this will be large when there has been a price reversal and small when ranging or trending
    it shows the equivalent of a onebar pin but on double the native candle timeframe

    args:
        df: dataframe
        period: period of double smoothed EMA. set to 1 if smoothing not needed
        ax: for charting the output
    return:
        none: output saved to saved to df
    """
    df['open_shft'] = df['open'].shift(1)
    df['high_max'] = pd.concat([df['high'], df['high'].shift(1)], axis=1).T.max().values
    df['low_min'] = pd.concat([df['low'], df['low'].shift(1)], axis=1).T.min().values
    lo_wicks = df[['open_shft', 'close']].T.min() - df['low_min']
    hi_wicks = df['high_max'] - df[['open_shft', 'close']].T.max()
    bar_range = abs(df['high_max'] - df['low_min'])
    df['twobar_pin'] = (lo_wicks.ewm(span=period).mean() - hi_wicks.ewm(span=period).mean()) * bar_range  # EMA
    if period > 0:
        df['twobar_pin'] = df['twobar_pin'].ewm(span=period).mean()  # double smoothed
    fplt.volume_ocv(df[['open_shft', 'close', 'twobar_pin']], ax=ax,
                    colorfunc=fplt.strength_colorfilter)  # , legend='pin_wicks_osc'
    df.rename(columns={'twobar_pin': 'twobar_pin_' + str(period)}, inplace=True)
    df.drop(['high_max'], axis=1, inplace=True)
    df.drop(['low_min'], axis=1, inplace=True)
    df.drop(['open_shft'], axis=1, inplace=True)
    print('twobar_pin ', period, ' is complete')
    return (df)



def plot_threebar_pinbar(df, period, ax):
    """
    combine three candles into one candle, then measure the pin rejection of all bars
    this will be large when there has been a price reversal and small when ranging or trending
    it shows the equivalent of a onebar pin but on triple the native candle timeframe

    args:
        df: dataframe
        period: period of double smoothed EMA. set to 1 if smoothing not needed
        ax: for charting the output
    return:
        none: output saved to saved to df
    """
    df['open_shft'] = df['open'].shift(2)
    df['high_max'] = pd.concat([df['high'], df['high'].shift(1), df['high'].shift(2)], axis=1).T.max().values
    df['low_min'] = pd.concat([df['low'], df['low'].shift(1), df['low'].shift(2)], axis=1).T.min().values
    lo_wicks = df[['open_shft', 'close']].T.min() - df['low_min']
    hi_wicks = df['high_max'] - df[['open_shft', 'close']].T.max()
    bar_range = abs(df['high_max'] - df['low_min'])
    df['threebar_pin'] = (lo_wicks.ewm(span=period).mean() - hi_wicks.ewm(span=period).mean()) * bar_range  # EMA
    if period > 0:
        df['threebar_pin'] = df['threebar_pin'].ewm(span=period).mean()  # double smoothed
    fplt.volume_ocv(df[['open_shft', 'close', 'threebar_pin']], ax=ax,
                    colorfunc=fplt.strength_colorfilter)  # , legend='pin_wicks_osc'
    df.rename(columns={'threebar_pin': 'threebar_pin_' + str(period)}, inplace=True)
    df.drop(['high_max'], axis=1, inplace=True)
    df.drop(['low_min'], axis=1, inplace=True)
    df.drop(['open_shft'], axis=1, inplace=True)
    print('threebar_pin ', period, ' is complete')
    return (df)



def plot_fourbar_pinbar(df, period, ax):
    """
    combine four candles into one candle, then measure the pin rejection of all bars
    this will be large when there has been a price reversal and small when ranging or trending
    it shows the equivalent of a onebar pin but on four times the native candle timeframe

    args:
        df: dataframe
        period: period of double smoothed EMA. set to 1 if smoothing not needed
        ax: for charting the output
    return:
        none: output saved to saved to df
    """
    df['open_shft'] = df['open'].shift(3)
    df['high_max'] = pd.concat([df['high'], df['high'].shift(1), df['high'].shift(2), df['high'].shift(3)],
                               axis=1).T.max().values
    df['low_min'] = pd.concat([df['low'], df['low'].shift(1), df['low'].shift(2), df['low'].shift(3)],
                              axis=1).T.min().values
    lo_wicks = df[['open_shft', 'close']].T.min() - df['low_min']
    hi_wicks = df['high_max'] - df[['open_shft', 'close']].T.max()
    bar_range = abs(df['high_max'] - df['low_min'])
    df['fourbar_pin'] = (lo_wicks.ewm(span=period).mean() - hi_wicks.ewm(span=period).mean()) * bar_range  # EMA
    if period > 0:
        df['fourbar_pin'] = df['fourbar_pin'].ewm(span=period).mean()  # double smoothed
    fplt.volume_ocv(df[['open_shft', 'close', 'fourbar_pin']], ax=ax,
                    colorfunc=fplt.strength_colorfilter)  # , legend='pin_wicks_osc'
    df.rename(columns={'fourbar_pin': 'fourbar_pin_' + str(period)}, inplace=True)
    df.drop(['high_max'], axis=1, inplace=True)
    df.drop(['low_min'], axis=1, inplace=True)
    df.drop(['open_shft'], axis=1, inplace=True)
    print('fourbar_pin ', period, ' is complete')
    return (df)



def plot_sixbar_pinbar(df, period, ax):
    """
    combine six candles into one candle, then measure the pin rejection of all bars
    this will be large when there has been a price reversal and small when ranging or trending
    it shows the equivalent of a onebar pin but on four times the native candle timeframe

    args:
        df: dataframe
        period: period of double smoothed EMA. set to 1 if smoothing not needed
        ax: for charting the output
    return:
        none: output saved to saved to df
    """
    df['open_shft'] = df['open'].shift(5)
    df['high_max'] = pd.concat(
        [df['high'], df['high'].shift(1), df['high'].shift(2), df['high'].shift(3), df['high'].shift(4),
         df['high'].shift(5)], axis=1).T.max().values
    df['low_min'] = pd.concat(
        [df['low'], df['low'].shift(1), df['low'].shift(2), df['low'].shift(3), df['low'].shift(4), df['low'].shift(5)],
        axis=1).T.min().values
    lo_wicks = df[['open_shft', 'close']].T.min() - df['low_min']
    hi_wicks = df['high_max'] - df[['open_shft', 'close']].T.max()
    bar_range = abs(df['high_max'] - df['low_min'])
    df['sixbar_pin'] = (lo_wicks.ewm(span=period).mean() - hi_wicks.ewm(span=period).mean()) * bar_range  # EMA
    if period > 0:
        df['sixbar_pin'] = df['sixbar_pin'].ewm(span=period).mean()  # double smoothed
    fplt.volume_ocv(df[['open_shft', 'close', 'sixbar_pin']], ax=ax,
                    colorfunc=fplt.strength_colorfilter)  # , legend='pin_wicks_osc'
    df.rename(columns={'sixbar_pin': 'sixbar_pin_' + str(period)}, inplace=True)
    df.drop(['high_max'], axis=1, inplace=True)
    df.drop(['low_min'], axis=1, inplace=True)
    df.drop(['open_shft'], axis=1, inplace=True)
    print('sixbar_pin ', period, ' is complete')
    return (df)



def Import_FinBERT_Sentiment(df, folder: str, sub_folder: str, freq: str, cand_per_day: int, ax: str, axb: str):
    """Import_FinBERT
    1 concatenate multiple yearly .csv files from the FinBERT folder into one complete pandas database
    2 create subset of columns with only what is needed
    3 drop duplicate rows and NaN
    4 make 'datePublished' column a DateTime object
    5 create empty column with datetime
    6 place 'Europe/Helsinki' timezone adjusted datetime into new column (native timezone is already encoded in datetime)
    7 groupby per 'freq' so that the spuriously timed breaking news fits to a the same grid as the chart timeframe
    8 drop unused columns and concat the new columns onto the features df (reindexing the finbert df using the df index to remove weekend data)
    9 fill NaN and zero's with small float to prevent div by zero
    10 add new column 'finbert_sent_osc' and fill with the difference between an EMA smoothed (4 bar), Least Squares fit Weekly Linear Regression Slope
    11 replace 'pos_sent' and 'neg_sent' Least Squares fit daily Linear Regression Slope to convert the metric from an absolute metric to a 'daily rate of change' style metric
    12 chart the output and check syncronisation

    :param folder: the directory of the futures folder
    :param sub_folder: the directory of the futures folder
    :param sub_folder: the directory of the futures folder
    :param freq: time divisions used byt he groupby function
    :param cand_per_day: number of bars per day for the Linear Regression Slope
    :param ax: for charting to a panel
    :param axb: for charting to a panel
    :return: none - simple text print confirmation
    """
    # 1 ---
    directory = os.path.join(f"{folder}", f"{sub_folder}")
    list_of_files = []
    for f in os.listdir(directory):
        if f.endswith('.csv'):
            fullpath = os.path.join(directory, f)
            list_of_files.append(pd.read_csv(fullpath))
    # 2 ---
    finbert_sentiment = pd.concat(list_of_files)[['datePublished', 'tweet', 'pos_sent', 'neg_sent', 'neu_sent']]
    finbert_sentiment.drop_duplicates(subset=['datePublished', 'tweet'], inplace=True)
    # 3 ---
    finbert_sentiment.dropna(inplace=True)
    finbert_sentiment.reset_index(inplace=True)
    # 4 ---
    finbert_sentiment['datePublished'] = pd.to_datetime(finbert_sentiment['datePublished'])  # change to datetime object
    # 5 ---
    finbert_sentiment['timestamp_5day'] = np.NaN
    # 6 ---
    dt_native = finbert_sentiment['datePublished'].copy()
    for i in tqdm(range(len(finbert_sentiment)), desc="finbert_sentiment"):
        """ iterate through dt_native and replace timestamp_5day with corrected timezone from datePublished
        # nyc_tz = tz.gettz('America/New_York')
        # ldn_tz = tz.gettz('Europe/London')
        # fin_tz = tz.gettz('Europe/Helsinki')
        """
        # nat_tz = tz.gettz(native_tz) # hopefully the timestamp is already time-zoned so this isn't needed
        chart_tz = tz.gettz('Europe/Helsinki')
        dt = datetime(dt_native[i].year,
                      dt_native[i].month,
                      dt_native[i].day,
                      dt_native[i].hour,
                      dt_native[i].minute,
                      dt_native[i].second
                      )
        finbert_sentiment.at[i, 'timestamp_5day'] = dt.astimezone(chart_tz)
    print('finbert_sentiment_timezone_convert to Europe/Helsinki is complete')
    # 7 ---
    grouped = finbert_sentiment.groupby(pd.Grouper(key='timestamp_5day', axis=0, freq=freq)).mean()  # use mean or sum
    # 8 ---
    grouped.drop(['index'], axis=1, inplace=True)
    df = pd.concat([df, grouped.reindex(df.index)], axis=1)
    # 9 ---
    df['neu_sent'] = df['neu_sent'].fillna(0.00000001)  # prevent div by 0
    df['neu_sent'] = df['neu_sent'].replace(0, 0.00000001)  # prevent div by 0
    df['pos_sent'] = df['pos_sent'].fillna(0.00000001)  # prevent div by 0
    df['pos_sent'] = df['pos_sent'].replace(0, 0.00000001)  # prevent div by 0
    df['neg_sent'] = df['neg_sent'].fillna(0.00000001)  # prevent div by 0
    df['neg_sent'] = df['neg_sent'].replace(0, 0.00000001)  # prevent div by 0
    # 10 ---
    cand_per_week = 5 * cand_per_day
    df['finbert_sent_osc'] = df['pos_sent'].ewm(span=2).mean().rolling(cand_per_week).apply(
        lambda x: np.polyfit(np.array(range(0, cand_per_week)), x, 1)[0], raw=True) - df['neg_sent'].ewm(
        span=2).mean().rolling(cand_per_week).apply(lambda x: np.polyfit(np.array(range(0, cand_per_week)), x, 1)[0],
                                                    raw=True)
    # 11 ---
    df['pos_sent'] = df['pos_sent'].rolling(cand_per_day).apply(
        lambda x: np.polyfit(np.array(range(0, cand_per_day)), x, 1)[0], raw=True)
    df['neg_sent'] = df['neg_sent'].rolling(cand_per_day).apply(
        lambda x: np.polyfit(np.array(range(0, cand_per_day)), x, 1)[0], raw=True)
    df['finbert_sent_osc_day'] = df['pos_sent'] - df['neg_sent']
    # 12 --
    # df.pos_sent.plot(ax=ax, legend='pos_sent', color='lime')
    # df.neg_sent.plot(ax=ax, legend='neg_sent', color='red')
    # df.neu_sent.plot(ax=ax, legend='neu_sent', color='aqua')
    fplt.volume_ocv(df[['open', 'close', 'finbert_sent_osc_day']], ax=ax, colorfunc=fplt.strength_colorfilter)
    fplt.volume_ocv(df[['open', 'close', 'finbert_sent_osc']], ax=axb, colorfunc=fplt.strength_colorfilter)
    print('Import_FinBERT_Sentiment is complete')
    return (df)



def Import_Economic_Calendar(df, folder: str, freq: str):  # , ax: str
    """ Import_Economic_Calendar
    1 - import www.fxstreet.com .csv economic callendar (already downloaded)
    2 - split the calendar by country and filter the desired economic indicators
    3 - rejoin the five split callendars into one ad combine selected news into a combined column
    4 - apply timezone shift
    5 - groupby to same timeframe as the main price database
    6 - concat data to main price df (re-indexing to main price df)

    :param folder: the directory of the futures folder
    :param sub_folder: the directory of the callendar folder
    :param freq: time divisions used byt he groupby function (eg 'H' for hourly)
    :param ax: for charting to a panel
    :return: none - simple text print confirmation
    """
    # 1 ---
    directory = os.path.join(f"{folder}")
    fullpath = os.path.join(directory, 'calendar-event-list_2012_to_2022_EEST.csv')
    econ_cal = pd.read_csv(fullpath)[['Start', 'Name', 'Impact', 'Currency']]
    econ_cal['Start'] = pd.to_datetime(econ_cal['Start'], format='%m/%d/%Y %H:%M:%S')
    econ_cal.rename(columns={'Start': 'timestamp_5day'}, inplace=True)
    econ_cal.drop_duplicates(inplace=True)

    ####################################################################################################################
    # define a list of US news calendar items to include in the model
    """ US_econ_ind_list :
    - filter the full calendar dataframe for 'USA' and 'HIGH' impact
    - keep only those with unique 'Name' and reset index
    - select with the index which to keep
    - send to list

    OUTPUT:
        ['FOMC Minutes', 'Nonfarm Payrolls', 'Consumer Price Index (YoY)', 
        "Fed's Monetary Policy Statement and Press Conference", "Fed's Monetary Policy Statement", 
        'ISM Manufacturing Prices Paid', 'FOMC Press Conference', 'ADP Employment Change']
    """
    US_econ_ind_list = econ_cal[(
            (econ_cal['Currency'] == 'USD')
            & (econ_cal['Impact'] == 'HIGH')
    )].drop_duplicates('Name', keep='first')['Name'].reset_index(drop=True)[[0, 1, 4, 8, 13, 34, 38, 68]].tolist()
    # print(US_econ_ind_list)

    # create new dataframe with just that list of economic indicators
    US_econ_cal = econ_cal[((econ_cal['Currency'] == 'USD') * (econ_cal['Name'].isin(US_econ_ind_list)))][
        ['timestamp_5day', 'Name']]
    # print(US_econ_cal)

    # check that none are at the same time = will return df of clashing items (if any)
    # print(US_econ_cal[US_econ_cal.duplicated(subset=['timestamp_5day'], keep=False)])

    US_econ_cal.set_index('timestamp_5day', inplace=True)

    ####################################################################################################################
    # define a list of EU news calendar items to include in the model
    """ EU_econ_ind_list :
    - filter the full calendar dataframe for 'EUR' and 'HIGH' impact
    - keep only those with unique 'Name' and reset index
    - select with the index which to keep
    - send to list

    OUTPUT:
        ['ECB rate on main refinancing operations', 'ECB Monthly Report', 'Unemployment Rate', 
        'ECB Press Conference', 'European Monetary Union Press Conference']
    """
    EU_econ_ind_list = econ_cal[(
            (econ_cal['Currency'] == 'EUR')
            & (econ_cal['Impact'] == 'HIGH')
    )].drop_duplicates('Name', keep='first')['Name'].reset_index(drop=True)[[0, 3, 5, 10, 35]].tolist()
    # print(EU_econ_ind_list)

    # create new dataframe with just that list of economic indicators
    EU_econ_cal = econ_cal[((econ_cal['Currency'] == 'EUR') * (econ_cal['Name'].isin(EU_econ_ind_list)))][
        ['timestamp_5day', 'Name']]
    # print(EU_econ_cal)

    # check that none are at the same time = will return df of clashing items (if any)
    # print(EU_econ_cal[EU_econ_cal.duplicated(subset=['timestamp_5day'], keep=False)])
    EU_econ_cal.set_index('timestamp_5day', inplace=True)

    ####################################################################################################################
    # define a list of GB news calendar items to include in the model
    """ GB_econ_ind_list :
    - filter the full calendar dataframe for 'GBP' and 'HIGH' impact
    - keep only those with unique 'Name' and reset index
    - select with the index which to keep
    - send to list

    OUTPUT:
        ['BoE Interest Rate Decision', 'Claimant Count Change', 'Consumer Price Index (MoM)']
    """
    GB_econ_ind_list = econ_cal[(
            (econ_cal['Currency'] == 'GBP')
            & (econ_cal['Impact'] == 'HIGH')
    )].drop_duplicates('Name', keep='first')['Name'].reset_index(drop=True)[[1, 4, 11]].tolist()
    # print(GB_econ_ind_list)

    # create new dataframe with just that list of economic indicators
    GB_econ_cal = econ_cal[((econ_cal['Currency'] == 'GBP') * (econ_cal['Name'].isin(GB_econ_ind_list)))][
        ['timestamp_5day', 'Name']]
    # print(GB_econ_cal)

    # check that none are at the same time = will return df of clashing items (if any)
    # print(GB_econ_cal[GB_econ_cal.duplicated(subset=['timestamp_5day'], keep=False)])
    GB_econ_cal.set_index('timestamp_5day', inplace=True)

    ####################################################################################################################
    # define a list of JP news calendar items to include in the model
    """ JP_econ_ind_list :
    - filter the full calendar dataframe for 'JPY' and 'HIGH' impact
    - keep only those with unique 'Name' and reset index
    - select with the index which to keep
    - send to list

    OUTPUT:
        ['BoJ Monetary Policy Statement ', 'BoJ Monetary Policy Meeting Minutes', 'National Consumer Price Index (YoY)']
    """
    JP_econ_ind_list = econ_cal[(
            (econ_cal['Currency'] == 'JPY')
            & (econ_cal['Impact'] == 'HIGH')
    )].drop_duplicates('Name', keep='first')['Name'].reset_index(drop=True)[[0, 2, 5]].tolist()
    # print(JP_econ_ind_list)

    # create new dataframe with just that list of economic indicators
    JP_econ_cal = econ_cal[((econ_cal['Currency'] == 'JPY') * (econ_cal['Name'].isin(JP_econ_ind_list)))][
        ['timestamp_5day', 'Name']]
    # print(JP_econ_cal)

    # check that none are at the same time = will return df of clashing items (if any)
    # print(JP_econ_cal[JP_econ_cal.duplicated(subset=['timestamp_5day'], keep=False)])
    JP_econ_cal.set_index('timestamp_5day', inplace=True)

    ####################################################################################################################
    # define a list of CN news calendar items to include in the model
    """ CN_econ_ind_list :
    - filter the full calendar dataframe for 'JPY' and 'HIGH' impact
    - keep only those with unique 'Name' and reset index
    - select with the index which to keep
    - send to list

    OUTPUT:
        ['Producer Price Index (YoY)', 'PBoC Interest Rate Decision']
    """
    CN_econ_ind_list = econ_cal[(
            (econ_cal['Currency'] == 'CNY')
            & (econ_cal['Impact'] == 'HIGH')
    )].drop_duplicates('Name', keep='first')['Name'].reset_index(drop=True)[[1, 2]].tolist()
    # print(CN_econ_ind_list)

    # create new dataframe with just that list of economic indicators
    CN_econ_cal = econ_cal[((econ_cal['Currency'] == 'CNY') * (econ_cal['Name'].isin(CN_econ_ind_list)))][
        ['timestamp_5day', 'Name']]
    # print(CN_econ_cal)

    # check that none are at the same time = will return df of clashing items (if any)
    # print(CN_econ_cal[CN_econ_cal.duplicated(subset=['timestamp_5day'], keep=False)])
    CN_econ_cal.set_index('timestamp_5day', inplace=True)

    ####################################################################################################################
    """ combine all the above news calendars back into one calendar
    none of the items are at the same time as each other so can be a discrete categorical variable, rather than a list 

    NOTE:
        looking at this I can spot that although the calendar was meant to be Easter European Standard Time (EEST), 
        the downloaded csv is in fact GMT+0, so we need to do a timezone transformation on this also. 

    OUTPUT:    
        index remains numerical for the time being and 'timestamp_5day' a column of data
                  timestamp_5day            US_econ_cal                              EU_econ_cal                 GB_econ_cal                          JP_econ_cal                  CN_econ_cal
        0    2012-01-03 19:00:00           FOMC Minutes                                      NaN                         NaN                                  NaN                          NaN
        1    2012-01-06 13:30:00       Nonfarm Payrolls                                      NaN                         NaN                                  NaN                          NaN
        2    2012-01-12 12:00:00                    NaN                                      NaN  BoE Interest Rate Decision                                  NaN                          NaN
        3    2012-01-12 12:45:00                    NaN  ECB rate on main refinancing operations                         NaN                                  NaN                          NaN
        4    2012-01-18 09:30:00                    NaN                                      NaN       Claimant Count Change                                  NaN                          NaN
    """
    ALL_econ_cal = pd.concat([US_econ_cal, EU_econ_cal, GB_econ_cal, JP_econ_cal, CN_econ_cal], axis=1).reset_index()
    ALL_econ_cal.set_axis(['timestamp_5day', 'US_econ_cal', 'EU_econ_cal', 'GB_econ_cal', 'JP_econ_cal', 'CN_econ_cal'],
                          axis=1, inplace=True)

    cols = ['US_econ_cal', 'EU_econ_cal', 'GB_econ_cal', 'JP_econ_cal', 'CN_econ_cal']
    ALL_econ_cal.fillna('', inplace=True)
    ALL_econ_cal['ALL_econ_cal'] = ALL_econ_cal[cols].apply(lambda row: ''.join(row.values.astype('str')), axis=1)

    dt_native = ALL_econ_cal['timestamp_5day'].copy()
    for i in tqdm(range(len(ALL_econ_cal)), desc="ALL_econ_cal"):
        """ iterate through dt_native and replace timestamp_5day with corrected timezone 
        """
        nat_tz = tz.gettz('Etc/UTC')  # need to check this does not have native DST shift, otherwise use 'Europe/London'
        chart_tz = tz.gettz('Europe/Helsinki')
        dt = datetime(dt_native[i].year,
                      dt_native[i].month,
                      dt_native[i].day,
                      dt_native[i].hour,
                      dt_native[i].minute,
                      dt_native[i].second,
                      tzinfo=nat_tz)
        ALL_econ_cal.at[i, 'timestamp_5day'] = dt.astimezone(chart_tz)


    print('ALL_econ_cal UTC to Europe/Helsinki is complete')
    print(ALL_econ_cal)

    # check that none are at the same time = will return df of clashing items (if any)
    # print(ALL_econ_cal[ALL_econ_cal.duplicated(subset=['timestamp_5day'], keep=False)])

    ####################################################################################################################
    """ merge the 5 dataframes into one, then groupby to hourly resolution to match the chart and main price dataframe times
    """
    ALL_econ_cal = ALL_econ_cal.groupby(pd.Grouper(key='timestamp_5day', axis=0, freq=freq))
    ALL_econ_cal = ALL_econ_cal.aggregate(np.sum)['ALL_econ_cal']
    print(ALL_econ_cal.head(5))
    ####################################################################################################################
    """ merge the new combined dataframe into the main price dataframe to use as categorical variable (re-indexed)
    """
    df = pd.concat([df, ALL_econ_cal.reindex(df.index)], axis=1)  # THIS DOES NOT WORK <<<<<<<<<
    df['ALL_econ_cal'] = df['ALL_econ_cal'].replace(0, np.nan)

    """ Convert News Calendar Data into encoded labels:
    We do not need to worry about transforming the whole dataset and leaking data from the future because we have 
    pre-selected a fixed list of economic events to be included. These do not change throughout the whole dataset
    therefore we can fit and transform the whole thing now.
    """
    df["ALL_econ_cal"] = df.ALL_econ_cal.astype(str)
    df["ALL_econ_cal"] = df.ALL_econ_cal.replace("0", np.nan)
    df["ALL_econ_cal"] = df.ALL_econ_cal.replace('nan', np.nan)
    df["ALL_econ_cal"] = df.ALL_econ_cal.replace('', np.nan)

    # LABEL ENCODING FOR NEWS CALLENDAR
    le = preprocessing.LabelEncoder()
    categoricals_ = le.fit(df["ALL_econ_cal"].values)
    df["ALL_econ_cal_lab"] = le.transform(df["ALL_econ_cal"].values)
    print('Import_Economic_Calendar is complete:')
    return (df)


def plot_macd(df, fast, slow, sig, ax):
    """macd modified from the finplot examples:
    https://github.com/highfestiva/finplot/blob/master/finplot/examples/analyze.py
    :param df:
    :return:
    """
    macd_ = df.close.ewm(span=fast).mean() - df.close.ewm(span=slow).mean()
    signal = macd_.ewm(span=sig).mean()
    df['macd'] = (macd_ - signal).values
    fplt.volume_ocv(df[['open', 'close', 'macd']], ax=ax, colorfunc=fplt.strength_colorfilter)
    fplt.plot(macd_, ax=ax, color='lime', legend='MACD ' + str([fast, slow]))
    fplt.plot(signal, ax=ax, color='red', legend='Signal ' + str(sig))
    df.rename(columns={'macd': 'macd_' + str(fast) + '_' + str(slow) + '_' + str(sig)}, inplace=True)
    print('macd ', fast, slow, sig, ' is complete')
    return (df)


# def plot_multi_zigzag_binary(df, dev_1, dev_2, dev_3, ax):
#     """
#     The takes four percentage movement zigzags and encodes the output of the current leg as a 4 bit binary number
#     :param df:
#     :param dev_1:
#     :param dev_2:
#     :param dev_3:
#     :param ax:
#     :return:
#     """
#     def first_leg():
#         start =
#     pass


def plot_vwap(df, period, colour, ax):
    """
    :param df:
    :param period:
    :param colour:
    :param ax:
    :return:
    """
    df['hlc3v'] = ((df['high'] + df['low'] + df['close']) / 3) * df['volume']
    vwap = pd.Series([], dtype='float64')
    for _, g in df.groupby(pd.Grouper(level='timestamp_5day', freq=period)):
        try:
            i0, i1 = g.index[0], g.index[-1]
            vwap = pd.concat([vwap, g.hlc3v.loc[i0:i1].cumsum() / df.volume.loc[i0:i1].cumsum()], ignore_index=True)
        except:
            continue
    vwap.plot(ax=ax, color=colour)
    df['vwap_' + str(period)] = vwap.values
    df.drop(['hlc3v'], axis=1, inplace=True)
    print('vwap ', period, ' is complete')
    return (df)


def plot_vwap_dist(df, vwap, ax):
    """

    """
    df[vwap + '_dist'] = 0
    if vwap == 'vwap_D':
        df.loc[((df['high'] + df['low'] + df['close']) / 3) > df[vwap], vwap + '_dist'] = ((df['high'] + df['low'] + df[
            'close']) / 3) - df[vwap]
        df.loc[((df['high'] + df['low'] + df['close']) / 3) < df[vwap], vwap + '_dist'] = ((df['high'] + df['low'] + df[
            'close']) / 3) - df[vwap]
    else:
        df.loc[df['low'] > df[vwap], vwap + '_dist'] = df['low'] - df[vwap]
        df.loc[df['high'] < df[vwap], vwap + '_dist'] = df['high'] - df[vwap]
    fplt.volume_ocv(df[['open', 'close', vwap + '_dist']], ax=ax, colorfunc=fplt.strength_colorfilter)
    print('vwap_dist ', vwap, ' is complete')
    return (df)


def plot_ema_dist(df, ema, ax):
    """

    """
    df[ema + '_dist'] = 0
    df.loc[df['low'] > df[ema], ema + '_dist'] = df['low'] - df[ema]
    df.loc[df['high'] < df[ema], ema + '_dist'] = df['high'] - df[ema]
    fplt.volume_ocv(df[['open', 'close', ema + '_dist']], ax=ax, colorfunc=fplt.strength_colorfilter)
    print('ema_dist ', ema, ' is complete')
    return (df)


#######################################################################################################################
def chart_dataframe(df, ticker_name, new_timeframe):
    """

    """
    # convert native df Eastern Time to NY CLose (Helsinki time)

    df_timezone_convert(df, 'America/New_York', 'Europe/Helsinki')

    # convert
    if new_timeframe == 'daily':
        df = resample_dataset_timeframe(df, 'D')
    else:
        pass

    # without this it would default to displaying in UTC, we need 'NY Close' broker time
    fplt.display_timezone = tz.gettz('Europe/Helsinki')

    # # all predictors
    # ax, ax_vol, ax_rtn, ax_rsi, ax_macd_1, ax_macd_2, ax_macd_3, ax_vwap_D, ax_vwap_W, ax_vwap_M, ax_ema50, \
    # ax_pin, ax_pin2, ax_pin3, ax_pin4, ax_pin6, axfin, axfin2 = fplt.create_plot(ticker_name, rows=18)
    #
    # # # all except finbert and news cal
    # # ax, ax_vol, ax_rtn, ax_rsi, ax_macd_1, ax_macd_2, ax_macd_3, ax_vwap_W, ax_vwap_M, ax_ema50, \
    # # ax_pin, ax_pin2, ax_pin3, ax_pin4, ax_pin6 = fplt.create_plot(ticker_name, rows=15)
    #
    # all price, vol, rtn, rsi, macd
    # ax, ax_vol, ax_rtn, ax_rsi, ax_macd_1, ax_macd_2, ax_macd_3 = fplt.create_plot(ticker_name, rows=7)
    #
    # all price, vol, vwap dis, ema
    # ax, ax_vwap_D, ax_vwap_W, ax_vwap_M, ax_ema50 = fplt.create_plot(ticker_name, rows=5)
    # #
    # # # all price, vol, pin and FinBert
    # ax, ax_pin, ax_pin2, ax_pin3, ax_pin4, ax_pin6  = fplt.create_plot(ticker_name, rows=6) #axfin, axfin2

    # ax, axfin, axfin2  = fplt.create_plot(ticker_name, rows=3) #

    # all predictors
    ax, ax_vol, ax_rtn, ax_rsi , ax_vwap_D, ax_vwap_W, ax_vwap_M, ax_ema50 , ax_pin, ax_pin2, ax_pin3, ax_pin4, \
    ax_pin6, ax_macd_1, ax_macd_2, ax_macd_3, axfin, axfin2 = fplt.create_plot(ticker_name, rows=18)



    ax.set_visible(xgrid=True, ygrid=True)

    # main price window
    plot_candles(df, ax)  # price chart

    # averages indicators
    a = 50
    plot_ema(df, a, ax)
    plot_vwap(df, 'D', 'black', ax)
    plot_vwap(df, 'W', 'green', ax)
    plot_vwap(df, 'M', 'blue', ax)

    # volume window
    plot_volume(df, ax_vol)

    # main price window
    plot_time_divs(df, new_timeframe)  # price chart

    # volatility indicator
    bar_return(df, ax_rtn)
    #
    # trend extension indicators
    b, c, d = 10, 15, 30
    plot_rsi(df, b, 'lime', ax_rsi)
    plot_rsi(df, c, 'orange', ax_rsi)
    plot_rsi(df, d, 'red', ax_rsi)

    # trend indication
    e, f, g = 8, 24, 16
    plot_macd(df, e, f, g, ax_macd_1)
    h, i, j = 16, 48, 32
    plot_macd(df, h, i, j, ax_macd_2)
    k, l, m = 32, 96, 64
    plot_macd(df, k, l, m, ax_macd_3)

    # relative distance indicators
    plot_vwap_dist(df, 'vwap_D', ax_vwap_D)
    plot_ema_dist(df, 'ema_50', ax_ema50)
    plot_vwap_dist(df, 'vwap_W', ax_vwap_W)
    plot_vwap_dist(df, 'vwap_M', ax_vwap_M)

    # trend reversal indicators
    n, o, p, q, r = 1, 1, 1, 1, 1
    plot_pinbar(df, n, ax_pin)
    plot_twobar_pinbar(df, o, ax_pin2)
    plot_threebar_pinbar(df, p, ax_pin3)
    plot_fourbar_pinbar(df, q, ax_pin4)
    plot_sixbar_pinbar(df, r, ax_pin6)

    # create the look ahead target returns - CHECK FOR LOOK AHEAD BIAS WHEN USING!!!!!!!!!
    target_return(df)


    if new_timeframe == 'daily':
        s = 'D'
        t = 5  # candles per week
    else:
        s = 'H'  # group news data by what timeframe?
        t = 24  # candles per day


    # # import and synchronise FinBERT news sentiment data
    df = Import_FinBERT_Sentiment(df, 'forexlive', '4_FinBert', s, t, axfin, axfin2)

    # import and synchronise economic calendar data
    df = Import_Economic_Calendar(df, 'callendar', s)





    # save settings as a string for output filename
    if new_timeframe == 'daily':
        df.reset_index(inplace=True)
        df['timestamp_5day'] = df['ny_hour']
        df.reset_index(inplace=True)
        df.set_index('timestamp_5day', inplace=True)
        df.drop(['index'], axis=1, inplace=True)
        parameters = [ticker_name.split('_')[0], new_timeframe,
                      str(a), str(b), str(c), str(d), str(e), str(f), str(g),
                      str(h), str(i), str(j), str(k), str(l), str(m), str(n),
                      str(o), str(p), str(q), str(r), str(s), str(t)]
    else:
        parameters = [ticker_name.split('_')[0], ticker_name.split('_')[3],
                      str(a), str(b), str(c), str(d), str(e), str(f), str(g),
                      str(h), str(i), str(j), str(k), str(l), str(m), str(n),
                      str(o), str(p), str(q), str(r), str(s), str(t)]

    experiment_name = '_'.join(parameters)
    print('experiment_name', experiment_name)



    # create folder within 'dataset' to place dataset. csv
    folder = 'dataset'
    directory = os.path.join(f"{folder}")
    fullpath = os.path.join(directory, experiment_name.replace(ticker_name.split('_')[0], "")[1:])
    if not os.path.exists(fullpath):
        os.mkdir(fullpath)
    df.to_csv(f"{fullpath}/{ticker_name.split('_')[0]}.csv")


    # show charting
    print(df.sample(50))  # check output of dataset
    fplt.autoviewrestore()
    fplt.show()


#######################################################################################################################
# def main(portfolio):

def main():

    # this is the timeframe of the raw futures data.
    timeframe = "1hour" # <<<  do not change this parameter, it specifies the data native timescale


    # rename .txt files as .csv
    txt_to_csv("futures", timeframe)


    # Change Dateset Timeframe ?
    """ new_timeframe
    - if 'daily', timeframe is grouped from hourly to daily. 
    - if not 'daily' function is inactive and the dataset remains at '1hour' timeframe
    - must be a string, but can be empty
    """
    new_timeframe = 'daily' # daily / 1hour


    # portfolio = SPARSE_PORTFOLIO
    portfolio = SINGLE_TICKER
    # portfolio = DUAL_TICKER

    for ticker in portfolio:
        # create pandas dataframe from raw futures data

        df = create_dataset("futures", timeframe, ticker)

        ticker_name = ticker + timeframe
        print(ticker_name)
        chart_dataframe(df, ticker_name, new_timeframe)


#######################################################################################################################
if __name__ == "__main__":
    """ 
    """
    main()



