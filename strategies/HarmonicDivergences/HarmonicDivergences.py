from enum import Enum
import pandas as pd
from pandas import DataFrame, Series
from typing import Dict, List, Optional, Tuple, Union
import ta
import matplotlib.pyplot as plt
import numpy as np
from strategies.Indicators import *
from strategies.PlotConfig import *
import talib.abstract as ta
from collections import deque
from strategies.IStrategy import *

pd.options.mode.chained_assignment = None  # default='warn'


class PivotSource(Enum):
    HighLow = 0
    Close = 1
    
class HarmonicDivergences(IStrategy):
    plot_config = {}

    def process_indicators(self, df):
        informative = df

        # RSI
        informative['rsi'] = ta.RSI(informative)
        # ATR
        informative['atr'] = atr(informative, window=14, exp=False)
        # Stochastic Slow
        informative['stoch'] = ta.STOCH(informative)['slowk']
        # ROC
        informative['roc'] = ta.ROC(informative)
        # Ultimate Oscillator
        informative['uo'] = ta.ULTOSC(informative)
        # Awesome Oscillator
        informative['ao'] = awesome_oscillator(informative)
        # MACD
        informative['macd'] = ta.MACD(informative)['macd']
        # Commodity Channel Index
        informative['cci'] = ta.CCI(informative)
        # CMF
        informative['cmf'] = self.chaikin_money_flow(informative, 20)
        # OBV
        informative['obv'] = ta.OBV(informative)
        # MFI
        informative['mfi'] = ta.MFI(informative)
        # ADX
        informative['adx'] = ta.ADX(informative)

        # Keltner Channel
        # keltner = qtpylib.keltner_channel(dataframe, window=20, atrs=1)
        keltner = self.emaKeltner(informative)
        informative["kc_upperband"] = keltner["upper"]
        informative["kc_middleband"] = keltner["mid"]
        informative["kc_lowerband"] = keltner["lower"]

        # Bollinger Bands
        bollinger = bollinger_bands(typical_price(informative), window=20, stds=2)
        informative['bollinger_upperband'] = informative['upper']
        informative['bollinger_lowerband'] = informative['lower']

        # EMA - Exponential Moving Average
        informative['ema9'] = ta.EMA(informative, timeperiod=9)
        informative['ema20'] = ta.EMA(informative, timeperiod=20)
        informative['ema50'] = ta.EMA(informative, timeperiod=50)#50)
        informative['ema200'] = ta.EMA(informative, timeperiod=200)#200)

        pivots = self.pivot_points(informative)
        informative['pivot_lows'] = pivots['pivot_lows']
        informative['pivot_highs'] = pivots['pivot_highs']

        self.initialize_divergences_lists(informative)
        
        self.add_divergences(informative, 'rsi')
        self.add_divergences(informative, 'stoch')
        self.add_divergences(informative, 'roc')
        self.add_divergences(informative, 'uo')
        self.add_divergences(informative, 'ao')
        self.add_divergences(informative, 'macd')
        self.add_divergences(informative, 'cci')
        self.add_divergences(informative, 'cmf')
        self.add_divergences(informative, 'obv')
        self.add_divergences(informative, 'mfi')
        self.add_divergences(informative, 'adx')

        df = resampled_merge(df, informative)

        return df
    
    # -- Condition to open Market LONG --
    def openLongCondition(self,  row):
        if row['total_bullish_divergences'] > 0 and row['volume'] > 0 and self.two_bands_check_row(row):
            return True
        else:
            return False

    # -- Condition to close Market LONG --
    def closeLongCondition(self, row):
        if (row['volume'] > 0):
            return True
        else:
            return False

    # -- Condition to open Market SHORT --
    def openShortCondition(self, row):
        if row['total_bearish_divergences'] > 0 and row['volume'] > 0 and self.two_bands_check_row(row): # and dataframe['volume'] > 0:
            return True
        else:
            return False

    # -- Condition to close Market SHORT --
    def closeShortCondition( self, row ):
        if (row['volume'] > 0):
            return True
        else:
            return False
    
    def handle_indicators(self, df):
        # RSI
        df['rsi'] = ta.RSI(df)
        # ATR
        df['atr'] = atr(df, window=14, exp=False)

        # Stochastic Slow
        df['stoch'] = ta.STOCH(df)['slowk']
        # ROC
        df['roc'] = ta.ROC(df)
        # Ultimate Oscillator
        df['uo'] = ta.ULTOSC(df)
        # Awesome Oscillator
        df['ao'] = awesome_oscillator(df)
        # MACD
        df['macd'] = ta.MACD(df)['macd']
        # Commodity Channel Index
        df['cci'] = ta.CCI(df)
        # CMF
        df['cmf'] = self.chaikin_money_flow(df, 20)
        # OBV
        df['obv'] = ta.OBV(df)
        # MFI
        df['mfi'] = ta.MFI(df)
        # ADX
        df['adx'] = ta.ADX(df)


        # Keltner Channel
        # keltner = qtpylib.keltner_channel(dataframe, window=20, atrs=1)
        keltner = self.emaKeltner(df)
        df["kc_upperband"] = keltner["upper"]
        df["kc_middleband"] = keltner["mid"]
        df["kc_lowerband"] = keltner["lower"]

        # Bollinger Bands
        bollinger = bollinger_bands(typical_price(df), window=20, stds=2)
        df['bollinger_upperband'] = bollinger['upper']
        df['bollinger_lowerband'] = bollinger['lower']

        # EMA - Exponential Moving Average
        df['ema9'] = ta.EMA(df, timeperiod=9)
        df['ema20'] = ta.EMA(df, timeperiod=20)
        df['ema50'] = ta.EMA(df, timeperiod=50)#50)
        df['ema200'] = ta.EMA(df, timeperiod=200)#200)

        pivots = self.pivot_points(df)
        df['pivot_lows'] = pivots['pivot_lows']
        df['pivot_highs'] = pivots['pivot_highs']

        # Use the helper function merge_informative_pair to safely merge the pair
        # Automatically renames the columns and merges a shorter timeframe dataframe and a longer timeframe informative pair
        # use ffill to have the 1d value available in every row throughout the day.
        # Without this, comparisons between columns of the original and the informative pair would only work once per day.
        # Full documentation of this method, see below


        self.initialize_divergences_lists(df)
        self.add_divergences(df, 'rsi')
        self.add_divergences(df, 'stoch')
        self.add_divergences(df, 'roc')
        self.add_divergences(df, 'uo')
        self.add_divergences(df, 'ao')
        self.add_divergences(df, 'macd')
        self.add_divergences(df, 'cci')
        self.add_divergences(df, 'cmf')
        self.add_divergences(df, 'obv')
        self.add_divergences(df, 'mfi')
        self.add_divergences(df, 'adx')
        return df

    def plot_config(self, dataframe):
        plot_config = (
            PlotConfig()
            .add_pivots_in_config()
            .add_divergence_in_config('rsi')
            .add_divergence_in_config('stoch')
            .add_divergence_in_config('roc')
            .add_divergence_in_config('uo')
            .add_divergence_in_config('ao')
            .add_divergence_in_config('macd')
            .add_divergence_in_config('cci')
            .add_divergence_in_config('cmf')
            .add_divergence_in_config('obv')
            .add_divergence_in_config('mfi')
            .add_divergence_in_config('adx')
            .add_total_divergences_in_config(dataframe)
            .config
            )
        return plot_config

    def two_bands_check_row(self, row):
        check = (
        ((row['low'] < row['kc_lowerband']) & (row['high'] > row['kc_upperband']))  # 1
        )
        return ~check

    def two_bands_check(self, dataframe):
        check = (
        ((dataframe[resample('low')] < dataframe[resample('kc_lowerband')]) & (dataframe[resample('high')] > dataframe[resample('kc_upperband')])) # 1

        )
        return ~check

    def ema_cross_check(self, dataframe):
        dataframe['ema20_50_cross'] = crossed_below(dataframe[resample('ema20')],dataframe[resample('ema50')])
        dataframe['ema20_200_cross'] = crossed_below(dataframe[resample('ema20')],dataframe[resample('ema200')])
        dataframe['ema50_200_cross'] = crossed_below(dataframe[resample('ema50')],dataframe[resample('ema200')])
        return ~(
            dataframe['ema20_50_cross'] 
            | dataframe['ema20_200_cross'] 
            | dataframe['ema50_200_cross'] 
            )

    def green_candle(self, dataframe):
        return dataframe[resample('open')] < dataframe[resample('close')]

    def keltner_middleband_check(self, dataframe):
        return (dataframe[resample('low')] < dataframe[resample('kc_middleband')]) & (dataframe[resample('high')] > dataframe[resample('kc_middleband')])

    def keltner_lowerband_check(self, dataframe):
        return (dataframe[resample('low')] < dataframe[resample('kc_lowerband')]) & (dataframe[resample('high')] > dataframe[resample('kc_lowerband')])

    def bollinger_lowerband_check(self, dataframe):
        return (dataframe[resample('low')] < dataframe[resample('bollinger_lowerband')]) & (dataframe[resample('high')] > dataframe[resample('bollinger_lowerband')])

    def bollinger_keltner_check(self, dataframe):
        return (dataframe[resample('bollinger_lowerband')] < dataframe[resample('kc_lowerband')]) & (dataframe[resample('bollinger_upperband')] > dataframe[resample('kc_upperband')])

    def ema_check(self, dataframe):
        check = (
            (dataframe[resample('ema9')] < dataframe[resample('ema20')])
            & (dataframe[resample('ema20')] < dataframe[resample('ema50')])
            & (dataframe[resample('ema50')] < dataframe[resample('ema200')]))
        return ~check

    def initialize_divergences_lists(self, dataframe: DataFrame):
        dataframe["total_bullish_divergences"] = np.empty(len(dataframe['close'])) * np.nan
        dataframe["total_bullish_divergences_count"] = np.empty(len(dataframe['close'])) * np.nan
        dataframe["total_bullish_divergences_count"] = [0 if x != x else x for x in dataframe["total_bullish_divergences_count"]]
        dataframe["total_bullish_divergences_names"] = np.empty(len(dataframe['close'])) * np.nan
        dataframe["total_bullish_divergences_names"] = ['' if x != x else x for x in dataframe["total_bullish_divergences_names"]]
        dataframe["total_bearish_divergences"] = np.empty(len(dataframe['close'])) * np.nan
        dataframe["total_bearish_divergences_count"] = np.empty(len(dataframe['close'])) * np.nan
        dataframe["total_bearish_divergences_count"] = [0 if x != x else x for x in dataframe["total_bearish_divergences_count"]]
        dataframe["total_bearish_divergences_names"] = np.empty(len(dataframe['close'])) * np.nan
        dataframe["total_bearish_divergences_names"] = ['' if x != x else x for x in dataframe["total_bearish_divergences_names"]]
        dataframe["bearish_divergences_count"] = np.empty(len(dataframe['close'])) * np.nan
        dataframe["bearish_divergences_count"] = [0 if x != x else x for x in dataframe["bearish_divergences_count"]]
        dataframe["bullish_divergences_count"] = np.empty(len(dataframe['close'])) * np.nan
        dataframe["bullish_divergences_count"] = [0 if x != x else x for x in dataframe["bullish_divergences_count"]]

        return dataframe

    def add_divergences(self, dataframe: DataFrame, indicator: str):
            (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines) = self.divergence_finder_dataframe(dataframe, indicator)
            dataframe['bearish_divergence_' + indicator + '_occurence'] = bearish_divergences
            # for index, bearish_line in enumerate(bearish_lines):
            #     dataframe['bearish_divergence_' + indicator + '_line_'+ str(index)] = bearish_line
            dataframe['bullish_divergence_' + indicator + '_occurence'] = bullish_divergences
            # for index, bullish_line in enumerate(bullish_lines):
            #     dataframe['bullish_divergence_' + indicator + '_line_'+ str(index)] = bullish_line
            return dataframe

    def divergence_finder_dataframe(self, dataframe: DataFrame, indicator_source: str) -> Tuple[pd.Series, pd.Series]:
        bearish_lines = [np.empty(len(dataframe['close'])) * np.nan]
        bearish_divergences = np.empty(len(dataframe['close'])) * np.nan
        bullish_lines = [np.empty(len(dataframe['close'])) * np.nan]
        bullish_divergences = np.empty(len(dataframe['close'])) * np.nan
        low_iterator = []
        high_iterator = []

        for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
            if np.isnan(row.pivot_lows):
                low_iterator.append(0 if len(low_iterator) == 0 else low_iterator[-1])
            else:
                low_iterator.append(index)
            if np.isnan(row.pivot_highs):
                high_iterator.append(0 if len(high_iterator) == 0 else high_iterator[-1])
            else:
                high_iterator.append(index)

        for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):

            bearish_occurence = self.bearish_divergence_finder(dataframe,
                dataframe[indicator_source],
                high_iterator,
                index)

            if bearish_occurence != None:
                (prev_pivot , current_pivot) = bearish_occurence 
                bearish_prev_pivot = dataframe['close'][prev_pivot]
                bearish_current_pivot = dataframe['close'][current_pivot]
                bearish_ind_prev_pivot = dataframe[indicator_source][prev_pivot]
                bearish_ind_current_pivot = dataframe[indicator_source][current_pivot]
                length = current_pivot - prev_pivot
                bearish_lines_index = 0
                can_exist = True
                while(True):
                    can_draw = True
                    if bearish_lines_index <= len(bearish_lines):
                        bearish_lines.append(np.empty(len(dataframe['close'])) * np.nan)
                    actual_bearish_lines = bearish_lines[bearish_lines_index]
                    for i in range(length + 1):
                        point = bearish_prev_pivot + (bearish_current_pivot - bearish_prev_pivot) * i / length
                        indicator_point =  bearish_ind_prev_pivot + (bearish_ind_current_pivot - bearish_ind_prev_pivot) * i / length
                        if i != 0 and i != length:
                            if (point <= dataframe['close'][prev_pivot + i] 
                            or indicator_point <= dataframe[indicator_source][prev_pivot + i]):
                                can_exist = False
                        if not np.isnan(actual_bearish_lines[prev_pivot + i]):
                            can_draw = False
                    if not can_exist:
                        break
                    if can_draw:
                        for i in range(length + 1):
                            actual_bearish_lines[prev_pivot + i] = bearish_prev_pivot + (bearish_current_pivot - bearish_prev_pivot) * i / length
                        break
                    bearish_lines_index = bearish_lines_index + 1
                if can_exist:
                    bearish_divergences[index] = row.close
                    dataframe["total_bearish_divergences"][index] = row.close
                    dataframe["bearish_divergences_count"][index] = dataframe["bearish_divergences_count"][index] + 1
                    if index > 30:
                        dataframe["total_bearish_divergences_count"][index-30] = dataframe["total_bearish_divergences_count"][index-30] + 1
                        dataframe["total_bearish_divergences_names"][index-30] = dataframe["total_bearish_divergences_names"][index-30] + indicator_source.upper() + '<br>'

            bullish_occurence = self.bullish_divergence_finder(dataframe,
                dataframe[indicator_source],
                low_iterator,
                index)
            
            if bullish_occurence != None:
                (prev_pivot , current_pivot) = bullish_occurence
                bullish_prev_pivot = dataframe['close'][prev_pivot]
                bullish_current_pivot = dataframe['close'][current_pivot]
                bullish_ind_prev_pivot = dataframe[indicator_source][prev_pivot]
                bullish_ind_current_pivot = dataframe[indicator_source][current_pivot]
                length = current_pivot - prev_pivot
                bullish_lines_index = 0
                can_exist = True
                while(True):
                    can_draw = True
                    if bullish_lines_index <= len(bullish_lines):
                        bullish_lines.append(np.empty(len(dataframe['close'])) * np.nan)
                    actual_bullish_lines = bullish_lines[bullish_lines_index]
                    for i in range(length + 1):
                        point = bullish_prev_pivot + (bullish_current_pivot - bullish_prev_pivot) * i / length
                        indicator_point =  bullish_ind_prev_pivot + (bullish_ind_current_pivot - bullish_ind_prev_pivot) * i / length
                        if i != 0 and i != length:
                            if (point >= dataframe['close'][prev_pivot + i] 
                            or indicator_point >= dataframe[indicator_source][prev_pivot + i]):
                                can_exist = False
                        if not np.isnan(actual_bullish_lines[prev_pivot + i]):
                            can_draw = False
                    if not can_exist:
                        break
                    if can_draw:
                        for i in range(length + 1):
                            actual_bullish_lines[prev_pivot + i] = bullish_prev_pivot + (bullish_current_pivot - bullish_prev_pivot) * i / length
                        break
                    bullish_lines_index = bullish_lines_index + 1
                if can_exist:
                    bullish_divergences[index] = row.close
                    dataframe["total_bullish_divergences"][index] = row.close
                    dataframe["bullish_divergences_count"][index] = dataframe["bullish_divergences_count"][index] + 1
                    if index > 30:
                        dataframe["total_bullish_divergences_count"][index-30] = dataframe["total_bullish_divergences_count"][index-30] + 1
                        dataframe["total_bullish_divergences_names"][index-30] = dataframe["total_bullish_divergences_names"][index-30] + indicator_source.upper() + '<br>'
        
        return (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines)

    def bearish_divergence_finder(self, dataframe, indicator, high_iterator, index):
        if high_iterator[index] == index:
            current_pivot = high_iterator[index]
            occurences = list(dict.fromkeys(high_iterator))
            current_index = occurences.index(high_iterator[index])
            for i in range(current_index-1,current_index-6,-1):
                prev_pivot = occurences[i]
                if np.isnan(prev_pivot):
                    return
                if ((dataframe['pivot_highs'][current_pivot] < dataframe['pivot_highs'][prev_pivot] and indicator[current_pivot] > indicator[prev_pivot])
                or (dataframe['pivot_highs'][current_pivot] > dataframe['pivot_highs'][prev_pivot] and indicator[current_pivot] < indicator[prev_pivot])):
                    #print('bearish pivot high current pivot', dataframe['pivot_highs'][current_pivot] )
                    return (prev_pivot , current_pivot)
        return None

    def bullish_divergence_finder(self, dataframe, indicator, low_iterator, index):
        if low_iterator[index] == index:
            current_pivot = low_iterator[index]
            occurences = list(dict.fromkeys(low_iterator))
            current_index = occurences.index(low_iterator[index])
            for i in range(current_index-1,current_index-6,-1):
                prev_pivot = occurences[i]
                if np.isnan(prev_pivot):
                    return 
                if ((dataframe['pivot_lows'][current_pivot] < dataframe['pivot_lows'][prev_pivot] and indicator[current_pivot] > indicator[prev_pivot])
                or (dataframe['pivot_lows'][current_pivot] > dataframe['pivot_lows'][prev_pivot] and indicator[current_pivot] < indicator[prev_pivot])):
                    #print('prev pivot', prev_pivot)
                    #print('bullish pivot low current pivot', dataframe['pivot_lows'][current_pivot] )
                    #print('current pivot close', dataframe['close'])
                    return (prev_pivot, current_pivot)
        return None


    def pivot_points(self, dataframe: DataFrame, window: int = 5, pivot_source: PivotSource = PivotSource.Close) -> DataFrame:
        high_source = None
        low_source = None

        if pivot_source == PivotSource.Close:
            high_source = 'close'
            low_source = 'close'
        elif pivot_source == PivotSource.HighLow:
            high_source = 'high'
            low_source = 'low'

        pivot_points_lows = np.empty(len(dataframe['close'])) * np.nan
        pivot_points_highs = np.empty(len(dataframe['close'])) * np.nan
        last_values = deque()
        
        # find pivot points
        for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
            #if index > 88:
                #print('89')
            last_values.append(row)
            if len(last_values) >= window * 2 + 1:
                current_value = last_values[window]
                is_greater = True
                is_less = True
                for window_index in range(0, window):
                    left = last_values[window_index]
                    right = last_values[2 * window - window_index]
                    local_is_greater, local_is_less = self.check_if_pivot_is_greater_or_less(current_value, high_source, low_source, left, right)
                    is_greater &= local_is_greater
                    is_less &= local_is_less
                if is_greater:
                    pivot_points_highs[index - window] = getattr(current_value, high_source)
                if is_less:
                    pivot_points_lows[index - window] = getattr(current_value, low_source)
                last_values.popleft()

        #print('pivots highs before last', pivot_points_highs)
        #print('pivots highs last values', last_values)
        # find last one
        if len(last_values) >= window + 2:
            current_value = last_values[-2]
            is_greater = True
            is_less = True
            for window_index in range(0, window):
                left = last_values[-2 - window_index - 1]
                right = last_values[-1]
                local_is_greater, local_is_less = self.check_if_pivot_is_greater_or_less(current_value, high_source, low_source, left, right)
                is_greater &= local_is_greater
                is_less &= local_is_less
            if is_greater:
                pivot_points_highs[index-1] = getattr(current_value, high_source)
            if is_less:
                pivot_points_lows[index-1] = getattr(current_value, low_source)
        #print('pivots highs', pivot_points_highs)
        return pd.DataFrame(index=dataframe.index, data={
            'pivot_lows': pivot_points_lows,
            'pivot_highs': pivot_points_highs
        })

    def check_if_pivot_is_greater_or_less(self, current_value, high_source: str, low_source: str, left, right) -> Tuple[bool, bool]:
        is_greater = True
        is_less = True
        if (getattr(current_value, high_source) < getattr(left, high_source) or
            getattr(current_value, high_source) < getattr(right, high_source)):
            is_greater = False

        if (getattr(current_value, low_source) > getattr(left, low_source) or
            getattr(current_value, low_source) > getattr(right, low_source)):
            is_less = False
        return (is_greater, is_less)

    def emaKeltner(self, dataframe):
        keltner = {}
        _atr = atr(dataframe, window=10)
        ema20 = ta.EMA(dataframe, timeperiod=20)
        keltner['upper'] = ema20 + _atr
        keltner['mid'] = ema20
        keltner['lower'] = ema20 - _atr
        return keltner

    def chaikin_money_flow(self, dataframe, n=20, fillna=False) -> Series:
        _df = dataframe.copy()
        mfv = ((_df['close'] - _df['low']) - (_df['high'] - _df['close'])) / (_df['high'] - _df['low'])
        mfv = mfv.fillna(0.0)  # float division by zero
        mfv *= _df['volume']
        cmf = (mfv.rolling(n, min_periods=0).sum()
            / _df['volume'].rolling(n, min_periods=0).sum())
        if fillna:
            cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
        return Series(cmf, name='cmf')