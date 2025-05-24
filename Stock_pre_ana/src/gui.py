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
