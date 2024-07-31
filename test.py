import pandas as pd
from agent import Agent
from tqdm import tqdm
from utils import *

import time

df = pd.read_csv('datas/test_data_with_indicator.csv')

columns = ['Close', 'volatility_bbh', 'volatility_bbl', 'trend_macd', 'momentum_rsi']
df = df[columns]

stock_prices = df['Close'].values.tolist()
WINDOW_SIZE = 5
BATCH_SIZE = 32
num_episode = 3
action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

agent = Agent(state_dim=WINDOW_SIZE, is_eval=True)
agent.load('weights/weight2.keras')

start_time = time.time()
total_profit = 0

agent.reset()

state = getState(stock_prices, 0, WINDOW_SIZE + 1)
for t in range(len(stock_prices) - 1):
    if t % 100 == 0:
        print(f'\n-------------------Period: {t}/{len(stock_prices)}-------------------')
    reward = 0
    actions = agent.model.predict(state, verbose=0)[0]
    action = agent.act(state)

    next_state = getState(stock_prices, t + 1, WINDOW_SIZE + 1)

    if action == 1:
        agent.inventory.append(stock_prices[t])
        print('{}/{} | Buy: ₩{:,} | Number of Stocks: {}'.format(t, len(stock_prices) - 1, int(stock_prices[t]),
                                                                 len(agent.inventory)))
    elif action == 2 and len(agent.inventory) > 0:
        bought_price = agent.inventory.pop(0)
        profit = stock_prices[t] - bought_price
        reward = profit
        total_profit += profit
        print('{}/{} | Sell: ₩{:,} | 🎉Profit: ₩{:,}'.format(t,
                                                                  len(stock_prices) - 1,
                                                                  stock_prices[t],
                                                                  int(reward)))

    else:
        pass

    done = (t == len(stock_prices) - 2)
    agent.remember(state, action, reward, next_state, done)

    state = next_state
print('Total Profit: ₩{:,}'.format(total_profit))