# data_loader.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from datetime import datetime
import warnings
from config import pro

# ==================== 数据加载与预处理模块 ====================
class StockDataLoader:
    def __init__(self, symbol, start_date):
        self.symbol = self._format_symbol(symbol)
        self.start_date = start_date
        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        self.df = None

    def _format_symbol(self, symbol):
        return f"{symbol}.SH" if symbol.startswith('6') else f"{symbol}.SZ"

    def _fetch_data(self):
        for _ in range(3):
            try:
                df = pro.daily(
                    ts_code=self.symbol,
                    start_date=self.start_date,
                    end_date=datetime.now().strftime("%Y%m%d")
                )
                return df
            except Exception as e:
                print(f"Retrying... {str(e)}")
        return None

    def _process_features(self, df):
        df = (df.rename(columns={
            'trade_date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume'})
              .assign(date=lambda x: pd.to_datetime(x['date'], format='%Y%m%d'))
              .sort_values('date')
              .set_index('date')
              .loc[:, ['open', 'high', 'low', 'close', 'volume']])

        # 技术指标
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['Volatility'] = df['close'].rolling(20).std()
        df['ATR'] = (df['high'] - df['low']).rolling(14).mean()

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean().replace(0, 1e-10)
        df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))

        # 动量指标
        df['Momentum'] = df['close'] - df['close'].shift(4)
        df['Price_Velocity'] = df['close'].diff(3)
        df['Volume_Change'] = df['volume'].pct_change()

        return df.dropna().ffill().bfill()

    def _create_sequences(self, data, targets, window):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i - window:i])
            y.append(targets[i])
        return np.array(X), np.array(y)

    def load_dataset(self, window=30, test_size=0.3):
        raw_df = self._fetch_data()
        if raw_df is None:
            return None

        processed_df = self._process_features(raw_df)
        self.df = processed_df

        features = processed_df[['open', 'high', 'low', 'close', 'volume',
                                 'MA5', 'MA20', 'Volatility', 'ATR',
                                 'MACD', 'Signal', 'RSI',
                                 'Momentum', 'Price_Velocity']]
        targets = processed_df[['close']]

        X = self.feature_scaler.fit_transform(features)
        y = self.target_scaler.fit_transform(targets)

        X_seq, y_seq = self._create_sequences(X, y.flatten(), window)

        split = int(len(X_seq) * (1 - test_size))
        return (X_seq[:split], y_seq[:split],
                X_seq[split:], y_seq[split:],
                processed_df.index[window:][split:].tolist())
