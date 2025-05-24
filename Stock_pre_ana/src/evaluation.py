# evaluation.py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==================== 评估函数 ====================
def directional_accuracy(y_true, y_pred):
    dir_true = np.sign(np.diff(y_true.flatten()))
    dir_pred = np.sign(np.diff(y_pred.flatten()))
    return np.mean(dir_true == dir_pred)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_model(model, X_train, y_train, X_test, y_test, scaler):
    train_pred = scaler.inverse_transform(model.predict(X_train))
    y_train_true = scaler.inverse_transform(y_train.reshape(-1, 1))

    test_pred = scaler.inverse_transform(model.predict(X_test))
    y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1))

    metrics = {
        'train': {
            'MAE': mean_absolute_error(y_train_true, train_pred),
            'RMSE': np.sqrt(mean_squared_error(y_train_true, train_pred)),
            'R2': r2_score(y_train_true, train_pred),
            'MAPE': mean_absolute_percentage_error(y_train_true, train_pred)
        },
        'test': {
            'MAE': mean_absolute_error(y_test_true, test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test_true, test_pred)),
            'R2': r2_score(y_test_true, test_pred),
            'MAPE': mean_absolute_percentage_error(y_test_true, test_pred),
            'DIR_ACC': directional_accuracy(y_test_true, test_pred)
        }
    }
    return metrics, test_pred
