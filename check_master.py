import pandas as pd, sys
try:
    df = pd.read_parquet('ml_engine/USDT/master.parquet', columns=['timestamp'])
    print(len(df))
except Exception:
    print(0)
