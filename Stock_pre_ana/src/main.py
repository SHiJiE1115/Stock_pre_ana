# main.py
import warnings
from gui import InvestmentAnalysisApp

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    root = tk.Tk()
    app = InvestmentAnalysisApp(root)
    root.mainloop()
