"""
IStrategy interface
This module defines the interface to apply for strategies
"""
import logging
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import arrow
from pandas import DataFrame
from config.BotConfig import *
import pandas as pd

class IStrategy():
    """
    Interface for freqtrade strategies
    Defines the mandatory structure must follow any custom strategies

    Attributes you can use:
        minimal_roi -> Dict: Minimal ROI designed for the strategy
        stoploss -> float: optimal stoploss designed for the strategy
        timeframe -> str: value of the timeframe (ticker interval) to use with the strategy
    """
    # Strategy interface version
    # Default to version 2
    # Version 1 is the initial interface without metadata dict
    # Version 2 populate_* include metadata dict
    INTERFACE_VERSION: int = 2


    # Definition of plot_config. See plotting documentation for more details.
    plot_config: Dict = {}

    # -- Definition of dt, that will be the dataset to do your trades analyses --
    dt = {}
    dt = None
    dt = pd.DataFrame(columns=['date', 'position', 'reason',
                               'price', 'frais', 'wallet', 'drawBack'])

    # -- Get actual price --
    orderInProgress = ''

    # ------
    # -- You can change variables below --
    leverage = 50
    wallet = 20
    makerFee = 0.0002
    takerFee = 0.0007

    # -- Do not touch these values --
    initalWallet = wallet
    lastAth = wallet
    USDAMOUNT = 20
    stopLoss = 0
    takeProfit = 500000
    orderInProgress = ''
    longIniPrice = 0
    shortIniPrice = 0
    closePriceWithFee = 0
    longLiquidationPrice = 500000
    shortLiquidationPrice = 0
    dfTest = None

    exchange = {}

    def __init__(self, exchange) -> None:
        self.exchange = exchange

    def get_historical_data(self, symbol, timeframe, nb):
        print('get last historic', symbol)

        df = self.exchange.get_last_historical(symbol, timeframe, nb)

        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['open'] = pd.to_numeric(df['open'])
        df['volume'] = pd.to_numeric(df['volume'])

        # -- Set the date to index --
        # df = df.set_index(df['timestamp'])
        df.index = pd.to_datetime(df.index, unit='ms')

        return df
    def replay(self, exchange, df):
        return


    def run_init(self, symbol, timeframe, nb_candles):
        # -- Definition of dt, that will be the dataset to do your trades analyses --
        print("Run init", symbol, timeframe)
        self.dt = None
        self.dt = pd.DataFrame(columns=['date', 'position', 'reason',
                                   'price', 'frais', 'wallet', 'drawBack'])

        # -- Get actual price --
        self.orderInProgress = ''

        self.symbol = symbol
        self.pairName = symbol
        self.timeframe = timeframe

        # -----
        # -- You can change variables below --
        self.leverage = 50
        self.wallet = 20
        self.makerFee = 0.0002
        self.takerFee = 0.0007

        # -- Do not touch these values --
        self.initalWallet = wallet
        self.lastAth = wallet
        self.USDAMOUNT = 20
        self.stopLoss = 0
        self.takeProfit = 500000
        self.orderInProgress = ''
        self.longIniPrice = 0
        self.shortIniPrice = 0
        self.closePriceWithFee = 0
        self.longLiquidationPrice = 500000
        self.shortLiquidationPrice = 0
        print("Loading data", symbol, timeframe)

        self.df = self.get_historical_data(symbol, timeframe, nb_candles)
        self.dfTest = self.df.copy()
        self.dfTest = self.dfTest.iloc[0:0]

        # df.to_csv('2000.zip', index=False)
        return

    def run_result(self):
        # -- BackTest Analyses --
        self.dt = self.dt.set_index(self.dt['date'])
        self.dt.index = pd.to_datetime(self.dt.index)
        self.dt['resultat%'] = self.dt['wallet'].pct_change() * 100

        self.dt['tradeIs'] = ''
        self.dt.loc[self.dt['resultat%'] > 0, 'tradeIs'] = 'Good'
        self.dt.loc[self.dt['resultat%'] < 0, 'tradeIs'] = 'Bad'

        try:
            iniClose = dfTest.iloc[0]['close']
            lastClose = dfTest.iloc[len(dfTest) - 1]['close']
        except:
            iniClose = 0
            lastClose = 0
        try:
            holdPercentage = ((lastClose - iniClose) / iniClose)
            holdWallet = holdPercentage * leverage * initalWallet
            algoPercentage = ((self.wallet - self.initalWallet) / self.initalWallet)
            vsHoldPercentage = ((self.wallet - holdWallet) / holdWallet) * 100
        except:
            holdPercentage = 0
            holdWallet = 0
            algoPercentage = 0
            vsHoldPercentage = 0

        try:
            tradesPerformance = round(self.dt.loc[(dt['tradeIs'] == 'Good') | (self.dt['tradeIs'] == 'Bad'), 'resultat%'].sum()
                                      / self.dt.loc[
                                          (self.dt['tradeIs'] == 'Good') | (self.dt['tradeIs'] == 'Bad'), 'resultat%'].count(), 2)
        except:
            tradesPerformance = 0
            print("/!\ There is no Good or Bad Trades in your BackTest, maybe a problem...")

        try:
            TotalGoodTrades = self.dt.groupby('tradeIs')['date'].nunique()['Good']
            AveragePercentagePositivTrades = round(self.dt.loc[dt['tradeIs'] == 'Good', 'resultat%'].sum()
                                                   / self.dt.loc[dt['tradeIs'] == 'Good', 'resultat%'].count(), 2)
            idbest = self.dt.loc[dt['tradeIs'] == 'Good', 'resultat%'].idxmax()
            bestTrade = str(
                round(self.dt.loc[dt['tradeIs'] == 'Good', 'resultat%'].max(), 2))
        except:
            TotalGoodTrades = 0
            AveragePercentagePositivTrades = 0
            idbest = ''
            bestTrade = 0
            print("/!\ There is no Good Trades in your BackTest, maybe a problem...")

        try:
            TotalBadTrades = self.dt.groupby('tradeIs')['date'].nunique()['Bad']
            AveragePercentageNegativTrades = round(self.dt.loc[dt['tradeIs'] == 'Bad', 'resultat%'].sum()
                                                   / self.dt.loc[dt['tradeIs'] == 'Bad', 'resultat%'].count(), 2)
            idworst = self.dt.loc[dt['tradeIs'] == 'Bad', 'resultat%'].idxmin()
            worstTrade = round(self.dt.loc[dt['tradeIs'] == 'Bad', 'resultat%'].min(), 2)
        except:
            TotalBadTrades = 0
            AveragePercentageNegativTrades = 0
            idworst = ''
            worstTrade = 0
            print("/!\ There is no Bad Trades in your BackTest, maybe a problem...")

        totalTrades = TotalBadTrades + TotalGoodTrades

        try:
            TotalLongTrades = self.dt.groupby('position')['date'].nunique()['LONG']
            AverageLongTrades = round(self.dt.loc[dt['position'] == 'LONG', 'resultat%'].sum()
                                      / self.dt.loc[dt['position'] == 'LONG', 'resultat%'].count(), 2)
            idBestLong = self.dt.loc[dt['position'] == 'LONG', 'resultat%'].idxmax()
            bestLongTrade = str(
                round(self.dt.loc[dt['position'] == 'LONG', 'resultat%'].max(), 2))
            idWorstLong = self.dt.loc[dt['position'] == 'LONG', 'resultat%'].idxmin()
            worstLongTrade = str(
                round(self.dt.loc[dt['position'] == 'LONG', 'resultat%'].min(), 2))
        except:
            AverageLongTrades = 0
            TotalLongTrades = 0
            bestLongTrade = ''
            idBestLong = ''
            idWorstLong = ''
            worstLongTrade = ''
            print("/!\ There is no LONG Trades in your BackTest, maybe a problem...")

        try:
            TotalShortTrades = self.dt.groupby('position')['date'].nunique()['SHORT']
            AverageShortTrades = round(self.dt.loc[dt['position'] == 'SHORT', 'resultat%'].sum()
                                       / self.dt.loc[dt['position'] == 'SHORT', 'resultat%'].count(), 2)
            idBestShort = self.dt.loc[dt['position'] == 'SHORT', 'resultat%'].idxmax()
            bestShortTrade = str(
                round(self.dt.loc[dt['position'] == 'SHORT', 'resultat%'].max(), 2))
            idWorstShort = self.dt.loc[dt['position'] == 'SHORT', 'resultat%'].idxmin()
            worstShortTrade = str(
                round(self.dt.loc[dt['position'] == 'SHORT', 'resultat%'].min(), 2))
        except:
            AverageShortTrades = 0
            TotalShortTrades = 0
            bestShortTrade = ''
            idBestShort = ''
            idWorstShort = ''
            worstShortTrade = ''
            print("/!\ There is no SHORT Trades in your BackTest, maybe a problem...")

        try:
            totalGoodLongTrade = self.dt.groupby(['position', 'tradeIs']).size()['LONG']['Good']
        except:
            totalGoodLongTrade = 0
            print("/!\ There is no good LONG Trades in your BackTest, maybe a problem...")

        try:
            totalBadLongTrade = self.dt.groupby(['position', 'tradeIs']).size()['LONG']['Bad']
        except:
            totalBadLongTrade = 0
            print("/!\ There is no bad LONG Trades in your BackTest, maybe a problem...")

        try:
            totalGoodShortTrade = self.dt.groupby(['position', 'tradeIs']).size()['SHORT']['Good']
        except:
            totalGoodShortTrade = 0
            print("/!\ There is no good SHORT Trades in your BackTest, maybe a problem...")

        try:
            totalBadShortTrade = self.dt.groupby(['position', 'tradeIs']).size()['SHORT']['Bad']
        except:
            totalBadShortTrade = 0
            print("/!\ There is no bad SHORT Trades in your BackTest, maybe a problem...")

        try:
            TotalTrades = TotalGoodTrades + TotalBadTrades
            winRateRatio = (TotalGoodTrades / TotalTrades) * 100
        except:
            TotalTrades = 0
            winRateRatio = 0

        try:
            reasons = self.dt['reason'].unique()
        except:
            reasons = ''

        print("BackTest finished, final wallet :", wallet, "$")

        # <h1>Print Complete BackTest Analyses</h1>

        # In[112]:
        try:
            print("Pair Symbol :", self.pairName)
            print("Period : [" + str(self.dfTest.index[0]) + "] -> [" +
                  str(self.dfTest.index[len(self.dfTest) - 1]) + "]")
            print("Starting balance :", self.initalWallet, "$")

            print("\n----- General Informations -----")
            print("Final balance :", round(self.wallet, 2), "$")
            print("Performance vs US Dollar :", round(algoPercentage * 100, 2), "%")
            print("Buy and Hold Performence :", round(holdPercentage * 100, 2),
                  "% | with Leverage :", round(holdPercentage * 100, 2) * self.leverage, "%")
            print("Performance vs Buy and Hold :", round(vsHoldPercentage, 2), "%")
            print("Best trade : +" + bestTrade, "%, the ", idbest)
            print("Worst trade :", worstTrade, "%, the ", idworst)
            print("Worst drawBack :", str(100 * round(self.dt['drawBack'].min(), 2)), "%")
            print("Total fees : ", round(self.dt['frais'].sum(), 2), "$")

            print("\n----- Trades Informations -----")
            print("Total trades on period :", totalTrades)
            print("Number of positive trades :", TotalGoodTrades)
            print("Number of negative trades : ", TotalBadTrades)
            print("Trades win rate ratio :", round(winRateRatio, 2), '%')
            print("Average trades performance :", tradesPerformance, "%")
            print("Average positive trades :", AveragePercentagePositivTrades, "%")
            print("Average negative trades :", AveragePercentageNegativTrades, "%")

            print("\n----- LONG Trades Informations -----")
            print("Number of LONG trades :", TotalLongTrades)
            print("Average LONG trades performance :", AverageLongTrades, "%")
            print("Best  LONG trade +" + bestLongTrade, "%, the ", idBestLong)
            print("Worst LONG trade", worstLongTrade, "%, the ", idWorstLong)
            print("Number of positive LONG trades :", totalGoodLongTrade)
            print("Number of negative LONG trades :", totalBadLongTrade)
            print("LONG trade win rate ratio :", round(totalGoodLongTrade / TotalLongTrades * 100, 2), '%')

            print("\n----- SHORT Trades Informations -----")
            print("Number of SHORT trades :", TotalShortTrades)
            print("Average SHORT trades performance :", AverageShortTrades, "%")
            print("Best  SHORT trade +" + bestShortTrade, "%, the ", idBestShort)
            print("Worst SHORT trade", worstShortTrade, "%, the ", idWorstShort)
            print("Number of positive SHORT trades :", totalGoodShortTrade)
            print("Number of negative SHORT trades :", totalBadShortTrade)
            print("SHORT trade win rate ratio :", round(totalGoodShortTrade / TotalShortTrades * 100, 2), '%')

            print("\n----- Trades Reasons -----")
            reasons = self.dt['reason'].unique()
            for r in reasons:
                print(r + " number :", self.dt.groupby('reason')['date'].nunique()[r])
        except:
            print("Pair Symbol :", self.pairName)
            print("Starting balance :", self.initalWallet, "$")
            print("\n----- General Informations -----")
            print("Final balance :", round(self.wallet, 2), "$")

        return

    def process_candle(self, index, row, previousRow = None):
        try:
            if previousRow is None:
                previousRow = row
            if (self.orderInProgress != ''):
                # -- Check if there is a LONG order in progress --
                if self.orderInProgress == 'LONG':
                    # -- Check Liquidation --
                    if row['low'] < self.longLiquidationPrice:
                        print('/!\ YOUR LONG HAVE BEEN LIQUIDATED the',index)

                    # -- Check Stop Loss --
                    elif row['low'] < self.stopLoss:
                        self.orderInProgress = ''
                        self.closePrice = stopLoss
                        self.closePriceWithFee = self.closePrice - self.takerFee * self.closePrice
                        self.pr_change = (self.closePriceWithFee - self.longIniPrice) / self.longIniPrice
                        self.wallet = self.wallet + self.wallet*pr_change*self.leverage

                        # -- You can uncomment the line below if you want to see logs --
                        print('Close LONG at',self.closePrice,"the", index, '| wallet :', self.wallet,
                               '| result :', self.pr_change*100*self.leverage)


                        # -- Check if your wallet hit a new ATH to know the drawBack --
                        if self.wallet > self.lastAth:
                            self.lastAth = self.wallet

                        # -- Add the trade to DT to analyse it later --
                        myrow = {'date': index, 'position': "LONG", 'reason': 'Stop Loss Long', 'price': self.closePrice,
                                'frais': self.takerFee * self.wallet * self.leverage, 'wallet': self.wallet, 'drawBack': (self.wallet-self.lastAth)/self.lastAth}
                        self.dt = self.dt.append(myrow, ignore_index=True)
                    # -- Check If you have to close the LONG --
                    elif self.closeLongCondition(previousRow) == True:
                        self.orderInProgress = ''
                        self.closePrice = row['close']
                        self.closePriceWithFee = row['close'] - self.takerFee * row['close']
                        self.pr_change = (self.closePriceWithFee - self.longIniPrice) / self.longIniPrice
                        self.wallet = self.wallet + self.wallet*self.pr_change*self.leverage

                        # -- You can uncomment the line below if you want to see logs --
                        print('Close LONG at',self.closePrice,"the", index, '| wallet :', self.wallet,
                               '| result :', self.pr_change*100*self.leverage)


                        # -- Check if your wallet hit a new ATH to know the drawBack --
                        if self.wallet > self.lastAth:
                            self.lastAth = self.wallet

                        # -- Add the trade to DT to analyse it later --
                        myrow = {'date': index, 'position': "LONG", 'reason': 'Close Long Market', 'price': self.closePrice,
                                'frais': self.takerFee * self.wallet * self.leverage, 'wallet': self.wallet, 'drawBack': (self.wallet-self.lastAth)/self.lastAth}
                        self.dt = self.dt.append(myrow, ignore_index=True)

                # -- Check if there is a SHORT order in progress --
                elif self.orderInProgress == 'SHORT':
                    # -- Check Liquidation --
                    if row['high'] > self.shortLiquidationPrice:
                        print('/!\ YOUR SHORT HAVE BEEN LIQUIDATED the',index)

                    # -- Check stop loss --
                    elif row['high'] > self.stopLoss:
                        self.orderInProgress = ''
                        self.closePrice = self.stopLoss
                        self.closePriceWithFee = self.closePrice + self.takerFee * self.closePrice
                        self.pr_change = -(self.closePriceWithFee - self.shortIniPrice) / self.shortIniPrice
                        self.wallet = self.wallet + self.wallet*self.pr_change*self.leverage

                        # -- You can uncomment the line below if you want to see logs --
                        print('Close SHORT at',self.closePrice,"the", index, '| wallet :', self.wallet,
                               '| result :', self.pr_change*100*self.leverage)

                        # -- Check if your wallet hit a new ATH to know the drawBack --
                        if self.wallet > self.lastAth:
                            self.lastAth = self.wallet

                        # -- Add the trade to DT to analyse it later --
                        myrow = {'date': index, 'position': "SHORT", 'reason': 'Stop Loss Short', 'price': self.closePrice,
                                'frais': self.takerFee * self.wallet * self.leverage, 'wallet': self.wallet, 'drawBack': (self.wallet-self.lastAth)/self.lastAth}
                        self.dt = self.dt.append(myrow, ignore_index=True)


                    # -- Check If you have to close the SHORT --
                    elif self.closeShortCondition(previousRow) == True:
                        self.orderInProgress = ''
                        self.closePrice = row['close']
                        self.closePriceWithFee = row['close'] + self.takerFee * row['close']
                        self.pr_change = -(self.closePriceWithFee - self.shortIniPrice) / self.shortIniPrice
                        self.wallet = self.wallet + self.wallet*self.pr_change*self.leverage

                        # -- You can uncomment the line below if you want to see logs --
                        print('Close SHORT at',self.closePrice,"the", index, '| wallet :', self.wallet,
                               '| result :', self.pr_change*100*self.leverage)

                        # -- Check if your wallet hit a new ATH to know the drawBack --
                        if self.wallet > self.lastAth:
                            self.lastAth = self.wallet

                        # -- Add the trade to DT to analyse it later --
                        myrow = {'date': index, 'position': "SHORT", 'reason': 'Close Short Market', 'price': self.closePrice,
                                'frais': self.takerFee * self.wallet * self.leverage, 'wallet': self.wallet, 'drawBack': (self.wallet-self.lastAth)/self.lastAth}
                        self.dt = self.dt.append(myrow, ignore_index=True)

            # -- If there is NO order in progress --
            if self.orderInProgress == '':
                # -- Check If you have to open a LONG --
                if self.openLongCondition(previousRow) == True:
                    self.orderInProgress = 'LONG'
                    self.closePrice = row['close']
                    self.longIniPrice = row['close'] + self.takerFee * row['close']
                    self.tokenAmount = (self.wallet * self.leverage) / row['close']
                    self.longLiquidationPrice = self.longIniPrice - (self.wallet/self.tokenAmount)
                    self.stopLoss = self.closePrice - 0.03 * self.closePrice
                    # -- You can uncomment the line below if you want to see logs --
                    print('Open LONG at', self.closePrice, '$ the', index, '| Liquidation price :', self.longLiquidationPrice)

                    # -- Add the trade to DT to analyse it later --
                    myrow = {'date': index, 'position': "Open Long", 'reason': 'Open Long Market', 'price': self.closePrice,
                             'frais': self.takerFee * self.wallet * self.leverage, 'wallet': self.wallet, 'drawBack': (self.wallet-self.lastAth)/self.lastAth}
                    self.dt = self.dt.append(myrow, ignore_index=True)

                # -- Check If you have to open a SHORT --
                if self.openShortCondition(previousRow) == True:
                    self.orderInProgress = 'SHORT'
                    self.closePrice = row['close']
                    self.shortIniPrice = row['close'] - self.takerFee * row['close']
                    self.tokenAmount = (self.wallet * self.leverage) / row['close']
                    self.shortLiquidationPrice = self.shortIniPrice + (self.wallet/self.tokenAmount)
                    self.stopLoss = self.closePrice + 0.03 * self.closePrice
                    # -- You can uncomment the line below if you want to see logs --
                    print('Open SHORT', self.closePrice, '$ the', index, '| Liquidation price :', self.shortLiquidationPrice)

                    # -- Add the trade to DT to analyse it later --
                    myrow = {'date': index, 'position': "Open Short", 'reason': 'Open Short Market', 'price': self.closePrice,
                             'frais': self.takerFee * self.wallet * self.leverage, 'wallet': self.wallet, 'drawBack': (self.wallet-self.lastAth)/self.lastAth}
                    self.dt = self.dt.append(myrow, ignore_index=True)
        except BaseException as err:
            print(err)


    def backsteps(self, symbol, timeframe, nb_candles):
        self.run_init(symbol, timeframe, nb_candles)
        self.dfTest = self.df.copy()
        self.dfTest = self.dfTest.iloc[0:0]

        # -- Iteration on all your price dataset (df) --
        for index, candle in self.df.iterrows():
            self.dfTest = self.dfTest.append(candle, ignore_index=True)
            try:
                if len(self.dfTest) > 80:
                    self.dfTest = self.handle_indicators(self.dfTest)
                    row = self.dfTest.iloc[-1]
                    previousRow = self.dfTest.iloc[-2]
                    # -- If there is an order in progress --
                    self.process_candle(index, row, previousRow=previousRow)
            except BaseException as err:
                print(index, err)

        self.run_result()
        return
    def backtest(self, symbol, timeframe, nb_candles):
        self.run_init(symbol, timeframe, nb_candles)

        self.dfTest = self.df.copy()
        self.handle_indicators(self.dfTest)
        # -- Iteration on all your price dataset (df) --
        for index, row in self.dfTest.iterrows():
            # -- If there is an order in progress --
            self.process_candle(index, row, row)
        # -- Check if there is a LONG o
        self.run_result()
        return
    def live(self):
        return

    @abstractmethod
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators that will be used in the Buy and Sell strategy
        :param dataframe: DataFrame with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        return dataframe

    @abstractmethod
    def openLongCondition(self, row):
        return False

        # -- Condition to close Market LONG --
    @abstractmethod
    def closeLongCondition(self, row):
        return False

        # -- Condition to open Market SHORT --
    @abstractmethod
    def openShortCondition(self, row):
        return False

    # -- Condition to close Market SHORT --
    @abstractmethod
    def closeShortCondition(self, row):
        return False
