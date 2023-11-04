# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:12:36 2023

@author: saura
"""

import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np
import vectorbt as vbt
from AccountValueSummary import GetSummary
from datetime import time
import mplfinance as mpf
from data_reader import get_data
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 300


def AdjustPrices(mkts_data,mkts):
    if len(mkts)<2:
        mkts_adj_close_prices = mkts_data[['Adj Close']].ffill().copy()
        mkts_close_prices = mkts_data[['Close']].ffill().copy()
        mkts_open_prices = mkts_data[['Open']].ffill().copy()
        mkts_high_prices = mkts_data[['High']].ffill().copy()
        mkts_low_prices = mkts_data[['Low']].ffill().copy()
        mkts_volume = mkts_data[['Volume']].ffill().copy()
        
        mkts_adj_close_prices.columns = mkts
        mkts_close_prices.columns = mkts
        mkts_open_prices.columns = mkts
        mkts_high_prices.columns = mkts
        mkts_low_prices.columns = mkts
        mkts_volume.columns = mkts
    else:
        mkts_adj_close_prices = mkts_data['Adj Close'].ffill().copy()
        mkts_close_prices = mkts_data['Close'].ffill().copy()
        mkts_open_prices = mkts_data['Open'].ffill().copy()
        mkts_high_prices = mkts_data['High'].ffill().copy()
        mkts_low_prices = mkts_data['Low'].ffill().copy()
        mkts_volume =  mkts_data['Volume'].ffill().copy()

    #Adjust the Mkts Prices
    mkts_adj_factor = mkts_adj_close_prices / mkts_close_prices
    mkts_close_prices = mkts_close_prices * mkts_adj_factor
    mkts_open_prices = mkts_open_prices * mkts_adj_factor
    mkts_high_prices = mkts_high_prices * mkts_adj_factor
    mkts_low_prices = mkts_low_prices * mkts_adj_factor

    prices = pd.DataFrame(columns=mkts_data.columns)
    prices['Close'] = mkts_close_prices.copy()
    prices['Open'] = mkts_open_prices.copy()
    prices['High'] = mkts_high_prices.copy()
    prices['Low'] = mkts_low_prices.copy()
    prices['Volume'] = mkts_volume.copy()
    prices = prices.drop(['Adj Close'],axis=1)
    return prices

def get_perf_vals(signals,mkt_data,mkt):
    sigs_vbt = signals.copy()
    vbt_close_prices = mkt_data[['Close']].copy()
    vbt_open_prices = mkt_data[['Open']].copy()
    
    sigs_vbt.columns = [mkt]
    vbt_close_prices.columns = [mkt]
    vbt_open_prices.columns = [mkt] 
    
    sigs_vbt.columns.name = 'symbol'
    vbt_close_prices.columns.name = 'symbol'
    vbt_open_prices.columns.name = 'symbol'
    
    sigs_vbt.index.name = 'Date'
    vbt_close_prices.index.name = 'Date'
    vbt_open_prices.index.name = 'Date'
    
    vbt_pf = vbt.Portfolio.from_orders(
        vbt_close_prices,  # current close as reference price
        size=sigs_vbt.shift(1),  
        price= vbt_open_prices,  # current open as execution price
        size_type='targetpercent', 
        init_cash=100000,
        cash_sharing=True,  # share capital between assets in the same group
        call_seq='auto',  # sell before buying
        freq='d',  # index frequency for annualization
        size_granularity= 1,
        # raise_reject=True,
        lock_cash=True,
        # group_by = 'Grp',
        # min_size = 0,
        # fees=0.0025
        )
    
    AcctVal = pd.DataFrame({'Strategy':vbt_pf.value()})
    trades = vbt_pf.trades.records_readable
    
    return AcctVal,trades,vbt_pf

if __name__ == '__main__':
    
    mkts = ["Nifty"]#,"TCS.NS"]
    # data_15m = yf.download('^NSEI',interval='15m',period='60d')
    # data_5m = yf.download('^NSEI',interval='5m',period='60d')
    # daily_data = yf.download('^NSEI')
    sql_15m = """
    SELECT * from agastya_database.tblindexintraday15m
    Where vcSymbol = 'NIFTY'
    """
    mkts_data_15m = get_data(sql_15m)
    mkts_data_15m.set_index('dtDateTime',inplace=True)
    
    sql_5m = """
    SELECT * from agastya_database.tblindexintraday5m
    Where vcSymbol = 'NIFTY'
    """
    mkts_data_5m = get_data(sql_15m)
    mkts_data_5m.set_index('dtDateTime',inplace=True)
    
    sql_15m = """
    SELECT * from agastya_database.indexprice
    Where Symbol = 'NIFTY'
    """
    mkts_data_daily = get_data(sql_15m)
    mkts_data_daily.set_index('Date',inplace=True)
    
    # mkts_data_15m = AdjustPrices(data_15m,mkts)
    # mkts_data_5m = AdjustPrices(data_5m,mkts)
    # mkts_data_daily = AdjustPrices(daily_data,mkts)
    mkts_data_daily.index = pd.to_datetime(mkts_data_daily.index)
    kama = ta.kama(mkts_data_15m['fClose'])
    sma_daily = ta.sma(mkts_data_daily['Close'],10)
    # sma_daily.index = pd.to_datetime(sma_daily.index)
    
    
    dates = pd.DataFrame(mkts_data_15m.index, index=mkts_data_15m.index)
    dates.columns = ['Dates']
    dates = dates.loc[:mkts_data_daily.index.max()]
    signal = pd.DataFrame(0,index=mkts_data_15m.index, columns=['^NSEI'])
    
    last_trade = 0
    for dt in dates['Dates']:
        if last_trade == 0:
            if last_trade == -1 or last_trade == 0:
                if (kama.loc[dt] < mkts_data_15m.loc[dt,'fClose']) and (sma_daily.loc[str(dt.date())] < mkts_data_daily['Close'].loc[str(dt.date())]):
                    signal.loc[dt] = 1 
                    last_trade = 1
            if last_trade == 1 or last_trade == 0:
                if (kama.loc[dt] > mkts_data_15m.loc[dt,'fClose']) and (sma_daily.loc[str(dt.date())] > mkts_data_daily['Close'].loc[str(dt.date())]):
                    signal.loc[dt] = -1 
                    last_trade = -1
            
        else:
            if dates['Dates'].iloc[0].time() >= time(3, 15):
                signal.loc[dt] = 0
                last_trade = 0
    
    vbt_data = mkts_data_15m[['fOpen','fHigh','fLow','fClose']]
    vbt_data.columns = ['Open','High','Low','Close']
    act_val,trades,vbt_pf = get_perf_vals(signal,vbt_data,mkts)
    
    plt.boxplot(trades['PnL'])
    plt.show()
    
    
    perf_summary = GetSummary(act_val)
    
    strat = pd.DataFrame(columns=['Open','Close','High','Low'])
    strat['Close'] = strat['Open'] = strat['High'] = strat['Low'] = act_val['Strategy']
    mpf.plot(strat,type='line')
    mpf.plot(vbt_data,type='line')