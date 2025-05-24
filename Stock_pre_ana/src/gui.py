# gui.py
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from threading import Thread
import queue
import gc
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from data_loader import StockDataLoader
from model import create_cnn_lstm
from evaluation import evaluate_model
from portfolio import PortfolioOptimizer, OptimizationError

# ==================== 图形界面系统 ====================
class InvestmentAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("测试版（理想条件）CNN-LSTM-CVaR理论分析对照")
        master.geometry("1400x800")

        # 初始化
        self.compare_codes = ['601390.SH', '600019.SH', '002334.SZ', '600406.SH',
             '600528.SH', '600312.SH', '002202.SZ']
        self.current_plots = []
        self.prediction_results = []

        # 创主容器
        main_frame = ttk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ==================== 左侧控制面板 ====================
        self.control_frame = ttk.LabelFrame(main_frame, text="控制面板", width=280)
        self.control_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)

        # 创建顶部输入面板
        self.input_panel = ttk.Frame(self.control_frame)
        self.input_panel.pack(side=tk.TOP, fill=tk.X)

        # 权重图
        self.weight_frame = ttk.Frame(self.control_frame)
        self.weight_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # ==================== 右侧主显示区 ====================
        self.right_panel = ttk.Frame(main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 初始化组件
        self._create_control_widgets()
        self._create_main_chart_area()
        self._create_weight_chart_area()
        self._start_task_consumer()

    def _create_main_chart_area(self):
        """创建主图表显示区"""
        # 结果表格
        result_frame = ttk.Frame(self.right_panel)
        result_frame.pack(fill=tk.X, padx=5, pady=5)

        self.result_tree = ttk.Treeview(result_frame,
                                        columns=('代码', '名称', '现价', '预测价', '收益率%', 'MAE', 'R²'),
                                        show='headings', height=8)
        for col, width in [('代码', 80), ('名称', 120), ('现价', 80), ('预测价', 80),
                           ('收益率%', 80), ('MAE', 80), ('R²', 80)]:
            self.result_tree.heading(col, text=col)
            self.result_tree.column(col, width=width, anchor=tk.CENTER)
        self.result_tree.pack(fill=tk.X)

        # 主图表
        self.main_figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.main_canvas = FigureCanvasTkAgg(self.main_figure, master=self.right_panel)
        self.main_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _create_weight_chart_area(self):
        self.weight_figure = plt.Figure(figsize=(4, 3), dpi=80)
        self.weight_canvas = FigureCanvasTkAgg(self.weight_figure, master=self.weight_frame)
        self.weight_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    def _update_status(self, message):
        self.master.after(0, lambda: self.status.config(
            text=message,
            wraplength=300  # 添加自动换行
        ))

    def _create_control_widgets(self):
        # 输入组件
        ttk.Label(self.input_panel, text="分析标的:").grid(row=0, column=0, sticky=tk.W)
        self.symbol_entry = ttk.Entry(self.input_panel, width=20)
        self.symbol_entry.grid(row=0, column=1, pady=5, sticky=tk.EW)
        self.symbol_entry.insert(0, "601766,600036,000001")

        # 参数设置
        params = [
            ("开始日期", "20150101"),
            ("窗口大小", "30"),
            ("测试集比例", "0.3"),
            ("训练轮次", "200")
        ]
        self.entries = {}
        for i, (label, default) in enumerate(params, 1):
            ttk.Label(self.input_panel, text=label + ":").grid(row=i, column=0, sticky=tk.W)
            entry = ttk.Entry(self.input_panel)
            entry.grid(row=i, column=1, pady=2, sticky=tk.EW)
            entry.insert(0, default)
            self.entries[label] = entry

        # 优化参数
        ttk.Label(self.input_panel, text="优化数量:").grid(row=5, column=0, sticky=tk.W)
        self.optim_num = tk.IntVar(value=3)
        self.optim_spin = ttk.Spinbox(self.input_panel, from_=2, to=10, width=5,
                                      textvariable=self.optim_num)
        self.optim_spin.grid(row=5, column=1, pady=5, sticky=tk.W)

        # R²阈值
        self.r2_threshold_var = tk.DoubleVar(value=0.3)
        ttk.Label(self.input_panel, text="最小R²阈值:").grid(row=6, column=0, sticky=tk.W)
        self.r2_spin = ttk.Spinbox(self.input_panel, from_=0.0, to=1.0, increment=0.05,
                                   width=5, textvariable=self.r2_threshold_var)
        self.r2_spin.grid(row=6, column=1, pady=2, sticky=tk.W)

        # 功能按钮
        btn_frame = ttk.Frame(self.input_panel)
        btn_frame.grid(row=7, column=0, columnspan=2, pady=10)

        self.predict_btn = ttk.Button(btn_frame, text="执行预测", command=self._start_prediction)
        self.predict_btn.pack(side=tk.LEFT, padx=5)

        self.optim_btn = ttk.Button(btn_frame, text="组合优化", command=self._start_secondary_optimization,
                                    state=tk.DISABLED)
        self.optim_btn.pack(side=tk.LEFT, padx=5)

        # 状态显示
        self.status = ttk.Label(self.input_panel, text="就绪")
        self.status.grid(row=8, column=0, columnspan=2, pady=5)

        # 配置网格布局权重
        self.input_panel.columnconfigure(1, weight=1)

    def _start_task_consumer(self):
        self.task_queue = queue.Queue()

        def consumer():
            while True:
                task_type, params = self.task_queue.get()
                try:
                    if task_type == "predict":
                        self._process_prediction(params)
                    elif task_type == "optimize":
                        self._process_optimization(params)
                except Exception as e:
                    self._update_status(f"错误: {str(e)}")
                finally:
                    self.task_queue.task_done()

        Thread(target=consumer, daemon=True).start()

    def _start_prediction(self):
        codes = [c.strip() for c in self.symbol_entry.get().split(',')]
        params = {
            "symbols": codes,
            "start_date": self.entries["开始日期"].get(),
            "window": int(self.entries["窗口大小"].get()),
            "test_ratio": float(self.entries["测试集比例"].get()),
            "epochs": int(self.entries["训练轮次"].get())
        }
        self.task_queue.put(("predict", params))
        self._update_status("开始预测任务...")

    def _process_prediction(self, params):
        try:
            predictions = []
            for idx, symbol in enumerate(params["symbols"]):
                self._update_status(f"处理中 {symbol} ({idx + 1}/{len(params['symbols'])})")

                loader = StockDataLoader(
                    symbol=symbol,
                    start_date=params["start_date"]
                )

                dataset = loader.load_dataset(
                    window=params["window"],
                    test_size=params["test_ratio"]
                )
                if not dataset:
                    continue

                X_train, y_train, X_test, y_test, test_dates = dataset

                model = create_cnn_lstm((params["window"], X_train.shape[2]))

                model.fit(
                    X_train, y_train,
                    validation_split=0.2,
                    epochs=params["epochs"],
                    batch_size=64,
                    callbacks=[
                        EarlyStopping(patience=20, min_delta=0.0001, restore_best_weights=True),
                        ReduceLROnPlateau(factor=0.5, patience=10, verbose=0),
                        TerminateOnNaN()
                    ],
                    verbose=0
                )

                metrics, pred = evaluate_model(model, X_train, y_train, X_test, y_test, loader.target_scaler)

                try:
                    stock_name = pro.stock_basic(ts_code=loader.symbol)['name'].values[0]
                except:
                    stock_name = loader.symbol.split('.')[0]

                y_test_true = loader.target_scaler.inverse_transform(y_test.reshape(-1, 1))
                start_price = y_test_true[0, 0]
                pred_end_price = pred[-1, 0]
                return_rate = ((pred_end_price - start_price) / start_price) * 100

                predictions.append({
                    "symbol": symbol,
                    "name": stock_name,
                    "current": start_price,
                    "predicted": pred_end_price,
                    "return_rate": return_rate,
                    "metrics": metrics,
                    "dates": loader.df.index[params["window"]:].tolist(),
                    "true_prices": loader.target_scaler.inverse_transform(
                        np.concatenate([y_train, y_test]).reshape(-1, 1)
                    ),
                    "predictions": pred,
                    "split_date": test_dates[0],
                    "metrics": metrics,  # 包含test部分R²值
                })

                del model
                gc.collect()

            self.prediction_results = sorted(predictions, key=lambda x: x["return_rate"], reverse=True)
            self.master.after(0, self._update_prediction_results)
            self.optim_btn.config(state=tk.NORMAL)
            self._update_status("预测完成")

        except Exception as e:
            self._update_status(f"预测出错: {str(e)}")
            raise

    def _update_prediction_results(self):
        """更新预测结果到界面"""
        self.result_tree.delete(*self.result_tree.get_children())
        for pred in self.prediction_results:
            values = (
                pred["symbol"],
                pred["name"],
                f"{pred['current']:.2f}",
                f"{pred['predicted']:.2f}",
                f"{pred['return_rate']:.1f}%",
                f"{pred['metrics']['test']['MAE']:.2f}",
                f"{pred['metrics']['test']['R2']:.2f}"
            )
            self.result_tree.insert('', 'end', values=values)

        if self.prediction_results:
            self._update_prediction_chart(self.prediction_results[0])

    def _update_prediction_chart(self, pred_data):
        self.main_figure.clear()
        ax = self.main_figure.add_subplot(111)

        ax.plot(pred_data["dates"], pred_data["true_prices"],
                label='实际价格', color='#1f77b4', alpha=0.7)
        ax.plot(pred_data["dates"][-len(pred_data["predictions"]):],
                pred_data["predictions"],
                label='预测价格', color='#d62728', linestyle='--')

        split_date = pd.to_datetime(pred_data["split_date"])
        ymin, ymax = ax.get_ylim()
        ax.axvline(x=split_date, color='#2ca02c', linestyle='--', linewidth=1)
        ax.fill_betweenx([ymin, ymax],
                         pred_data["dates"][0], split_date,
                         color='#bdc9e1', alpha=0.2, label='训练区间')
        ax.fill_betweenx([ymin, ymax],
                         split_date, pred_data["dates"][-1],
                         color='#fdcc8a', alpha=0.2, label='测试区间')

        ax.set_title(f"{pred_data['name']} 价格预测", fontsize=12)
        ax.set_xlabel("日期", fontsize=10)
        ax.set_ylabel("价格 (CNY)", fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        self.main_figure.tight_layout()
        self.main_canvas.draw()

    def _start_secondary_optimization(self):
        try:
            # 获取过滤参数
            selected_num = self.optim_num.get()
            r2_threshold = self.r2_threshold_var.get()

            # 执行过滤
            filtered = [p for p in self.prediction_results if p['metrics']['test']['R2'] >= r2_threshold]
            if len(filtered) < 2:
                messagebox.showerror("错误", f"过滤后剩余股票不足2只（当前{len(filtered)}）")
                return

            # 选择前N只
            selected_codes = [pred["symbol"] for pred in filtered[:selected_num]]

            # 记录筛选日志
            excluded = len(self.prediction_results) - len(filtered)
            self._update_status(f"排除{excluded}只低R²股票，优化前{len(selected_codes)}只"),
            selected_num = self.optim_num.get()
            # 检查selected_num是否超过过滤后数量
            if selected_num < 2 or selected_num > len(filtered):
                messagebox.showerror("错误", f"优化数量需在2~{len(filtered)}之间")
                return

            selected_codes = [pred["symbol"] for pred in filtered[:selected_num]]

            params = {
                "base_codes": selected_codes,
                "compare_codes": self.compare_codes,
                "start_date": self.entries["开始日期"].get(),
                "end_date": datetime.now().strftime("%Y%m%d")
            }
            self.task_queue.put(("optimize", params))
            self._update_status(f"优化前{selected_num}只股票...")
        except ValueError:
            messagebox.showerror("输入错误", "R²阈值需在0.0到1.0之间")



    def _process_optimization(self, params):
        try:
            optimizer = PortfolioOptimizer(
                params["base_codes"],
                params["compare_codes"],
                params["start_date"],
                params["end_date"]
            )
            results = optimizer.analyze_portfolios()
            self._show_optimization_results(results)
            self._update_status("优化完成")
        except Exception as e:
            self._update_status(f"优化失败: {str(e)}")
            raise

    def _show_optimization_results(self, results):
        # 清空图表
        self.main_figure.clf()
        self.weight_figure.clf()

        try:
            # ==================== 主收益图绘制 ====================
            ax_main = self.main_figure.add_subplot(111)

            import seaborn as sns
            sns.set_style("whitegrid")  # 设置 seaborn 样式

            plt.rcParams.update({
                'font.family': 'SimHei',
                'axes.labelpad': 4,
                'font.size': 8,
                'axes.titlesize': 9,
                'axes.labelsize': 8,
                'legend.fontsize': 7,
                'xtick.labelsize': 7,
                'ytick.labelsize': 7
            })

            # 绘制收益曲线
            line_config = {'lw': 1.0, 'alpha': 0.9, 'linestyle': '-'}
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            labels = ['Prediction+CVaR', 'Prediction+1/N', 'CVaROnly', 'None'
]

            # 绘制对比曲线
            for idx, key in enumerate(['original_optimized', 'original_equal', 'new_optimized', 'new_equal']):
                if key in results and not results[key].empty:
                    results[key].plot(
                        ax=ax_main,
                        color=colors[idx],
                        label=labels[idx],
                        **line_config
                    )

            # 主图格式
            ax_main.set_title("组合收益对比", fontsize=10, pad=12)
            ax_main.set_xlabel("")
            ax_main.set_ylabel("累计收益", labelpad=8)
            ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}x"))
            ax_main.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
            ax_main.legend(
                loc='upper left',
                fontsize=12,
                frameon=False,
                bbox_to_anchor=(0.02, 0.98)
            )

            # 添加收益标注
            stats_text = []
            for idx, key in enumerate(['original_optimized', 'original_equal', 'new_optimized', 'new_equal']):
                if key in results and not results[key].empty:
                    final_return = results[key].iloc[-1]
                    stats_text.append(f"{labels[idx]}: {final_return:.2f}x")

            ax_main.text(0.78, 0.18, "\n".join(stats_text),
                         transform=ax_main.transAxes,
                         fontsize=7,
                         linespacing=1.5,
                         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            self.main_canvas.draw()

            # ==================== 组合权重图绘制模块 ====================
            ax_weights = self.weight_figure.add_subplot(111)

            if 'optimized_weights' in results and len(results['optimized_weights']) > 0:
                weights = results['optimized_weights']
                codes = [c.split('.')[0] for c in results['optimized_codes']]
                colors = plt.cm.tab20(np.linspace(0, 1, len(codes)))
                y_pos = np.arange(len(codes))

                # 绘制水平条形图
                bars = ax_weights.barh(y_pos, weights,
                                       color=colors,
                                       edgecolor='white',
                                       height=0.7)

                # 添加数值标
                for i, (w, c) in enumerate(zip(weights, codes)):
                    ax_weights.text(w + 0.01, i,
                                    f"{w * 100:.1f}%",
                                    va='center',
                                    fontsize=7,
                                    color='#333333')

                # 动态调整x轴
                max_weight = max(weights)
                x_max = max(1.0, max_weight * 1.15)
                ax_weights.set_xlim(0, x_max)

                ax_weights.set_yticks(y_pos)
                ax_weights.set_yticklabels(codes, fontsize=7)
                ax_weights.set_title("组合权重分布", fontsize=8, pad=10)
                ax_weights.xaxis.set_visible(False)

                # 隐藏边框
                for spine in ['top', 'right', 'bottom']:
                    ax_weights.spines[spine].set_visible(False)
            else:
                ax_weights.text(0.5, 0.5, '无优化数据',
                                ha='center', va='center',
                                fontsize=9, color='gray')

            self.weight_canvas.draw()

        except Exception as e:
            print(f"绘图错误: {str(e)}")
            raise
