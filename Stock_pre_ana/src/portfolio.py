# portfolio.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from config import pro

# ==================== 组合优化模块 ====================
class PortfolioOptimizer:
    def __init__(self, base_codes, compare_codes, start_date, end_date):
        self.base_codes = base_codes
        self.compare_codes = compare_codes
        self.start_date = start_date
        self.end_date = end_date

    def _get_prices(self, codes):
        prices = {}
        valid_codes = []
        for code in codes:
            code = code.replace('.SH', '').replace('.SZ', '')  # 增加代码格式转换
            symbol = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"

            try:
                df = pro.daily(
                    ts_code=symbol,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    fields='trade_date,close'
                )
                if df is None or df.empty or len(df) < 20:
                    continue

                # 数据预处理部分
                df = (df.sort_values('trade_date')
                      .assign(trade_date=lambda x: pd.to_datetime(x['trade_date']))
                      .set_index('trade_date')
                      .rename(columns={'close': code}))  # 使用统一代码格式

                prices[code] = df[code]  # 使用原始代码作列名
                valid_codes.append(code)

            except Exception as e:
                print(f"跳过无效代码 {code}: {str(e)}")
                continue

        return pd.concat(prices.values(), axis=1).sort_index().ffill().bfill(), valid_codes

    def optimize_weights(self, codes, returns):
        try:
            min_weight = 0.05
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'ineq', 'fun': lambda x: x - min_weight}
            )
            bounds = [(min_weight, 0.5) for _ in codes]

            result = minimize(
                self._objective_function,
                x0=np.ones(len(codes)) / len(codes),
                args=(returns.values, 0.9, 0.0005),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )


            if not result.success:
                raise OptimizationError(result.message)

            return np.round(result.x, 4)
        except Exception as e:
            print(f"优化失败，使用等权重替代: {str(e)}")
            return np.ones(len(codes)) / len(codes)  # 返回等权重作为保底（简单测试版）

    def _objective_function(self, weights, returns, lambda_, cost):
        port_returns = returns.dot(weights)
        mean_return = np.mean(port_returns)

        alpha = 0.95
        var = -np.percentile(port_returns, 100 * (1 - alpha))
        tail_losses = port_returns[port_returns <= -var]
        cvar = -tail_losses.mean() if len(tail_losses) > 0 else 0

        transaction_cost = cost * np.sum(np.abs(weights))
        return -(mean_return - lambda_ * cvar - transaction_cost)

    def analyze_portfolios(self):
        # 获取价格数据（返回原始代码格式）
        original_prices, original_valid = self._get_prices(
            [c.replace('.SH', '').replace('.SZ', '') for c in self.base_codes])
        new_prices, new_valid = self._get_prices([c.replace('.SH', '').replace('.SZ', '') for c in self.compare_codes])

        # 增加空数据检查
        if len(original_valid) < 2 or len(new_valid) < 2:
            raise ValueError("有效股票数量不足，无法进行优化")

        # 使用有效的原始代码列
        original_returns = original_prices[original_valid].pct_change().dropna()
        new_returns = new_prices[new_valid].pct_change().dropna()

        # 优化权重（传递有效代码列表）
        original_weights = self.optimize_weights(original_valid, original_returns)
        new_weights = self.optimize_weights(new_valid, new_returns)

        # 返回结果增加有效性检查防错
        return {
            "optimized_weights": original_weights,
            "optimized_codes": original_valid,
            "original_optimized": self.calculate_returns(original_weights, original_returns),
            "original_equal": self.calculate_returns(
                np.ones(len(original_valid)) / len(original_valid), original_returns),
            "new_optimized": self.calculate_returns(new_weights, new_returns),
            "new_equal": self.calculate_returns(
                np.ones(len(new_valid)) / len(new_valid), new_returns)
        }

    # 将收益计算封装为类
    def calculate_returns(self, weights, returns):
        cost = 0.0005
        transaction_cost = cost * np.sum(np.abs(weights))
        adjusted_initial = 1 - transaction_cost
        portfolio_returns = returns.dot(weights)
        return adjusted_initial * (1 + portfolio_returns).cumprod()

# ==================== 新增异常类 ====================
class OptimizationError(Exception):
    pass
