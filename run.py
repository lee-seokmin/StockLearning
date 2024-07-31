import pandas as pd
# from agent import Agent
# from utils import *
#
# import time

df = pd.read_csv('datas/data_with_indicator.csv')
columns = ['Close', 'volatility_bbh', 'volatility_bbl', 'trend_macd', 'momentum_rsi']
df = df[columns]

print(df)
# stock_prices = df['Close'].values.tolist()
# WINDOW_SIZE = len(df.columns)
# BATCH_SIZE = 32
# action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
