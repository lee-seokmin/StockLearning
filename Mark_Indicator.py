import FinanceDataReader as fdr
df = fdr.DataReader('005930', '2022-01-01', '2022-01-31')

# Data 보조지수 입력
# MACD, BOLLINGER, STCK, RSI, VR, WR, MOK 등등
# 이 중에서 MACD, BOLLINGER, RSI 사용
from ta.utils import dropna
from ta import add_all_ta_features

df = dropna(df)
df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

df.to_csv('datas/samsung.csv')
