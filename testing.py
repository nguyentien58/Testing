import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def generate_stock_data(ticker="VIN", days=2000):
    """Giả lập dữ liệu chứng khoán thực tế với độ nhiễu và xu hướng."""
    np.random.seed(42)
    dates = pd.date_range(start='2018-01-01', periods=days, freq='D')
    
    # Tạo giá khởi điểm và xu hướng đi lên
    price = 100
    prices = []
    for _ in range(days):
        change = np.random.normal(0.001, 0.02) 
        price *= (1 + change)
        prices.append(price)
        
    df = pd.DataFrame({'Date': dates, 'Close': prices})
    df['Open'] = df['Close'] * (1 + np.random.normal(0, 0.01, days))
    df['High'] = df[['Close', 'Open']].max(axis=1) * (1 + abs(np.random.normal(0, 0.005, days)))
    df['Low'] = df[['Close', 'Open']].min(axis=1) * (1 - abs(np.random.normal(0, 0.005, days)))
    df['Volume'] = np.random.randint(1000, 100000, days)
    
    return df.set_index('Date')

def apply_technical_indicators(df):
    """Tính toán các chỉ số kỹ thuật để máy có thể học xu hướng."""
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Volatility (Độ biến động)
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Xử lý lỗi chia cho 0 (Inf) trước khi dropna
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Lags (Dữ liệu quá khứ)
    for i in range(1, 6):
        df[f'Lag_{i}'] = df['Close'].shift(i)
        
    # Target: Giá của ngày mai
    df['Target'] = df['Close'].shift(-1)
    
    return df.dropna()

def prepare_datasets(df):
    """Chuẩn bị tập Train/Test và chuẩn hóa dữ liệu."""
    features = ['Open', 'High', 'Low', 'Volume', 'MA20', 'MA50', 'Volatility', 'RSI', 
                'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5']
    X = df[features]
    y = df['Target']
    
    # Vì là dữ liệu thời gian
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, features, scaler


class StockNN(nn.Module):
    def __init__(self, input_dim):
        super(StockNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.layer3(x)

class PyTorchRegressor:
    def __init__(self, input_dim, epochs=200, lr=0.01):
        self.model = StockNN(input_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        
    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        # Chuyển y sang tensor, đảm bảo kích thước (N, 1)
        y_vals = y.values if hasattr(y, 'values') else y
        y_tensor = torch.tensor(y_vals, dtype=torch.float32).view(-1, 1)
        
        self.model.train()
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
            
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()

class StockPredictor:
    def __init__(self):
        self.models = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
        }
        self.best_model = None
        self.results = {}

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        # Thêm mô hình PyTorch vào danh sách
        self.models['PyTorch_DeepLearning'] = PyTorchRegressor(input_dim=X_train.shape[1])
        
        for name, model in self.models.items():
            print(f"--- Training {name} ---")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            
            self.results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Predictions': preds}
            print(f"{name} Results: MAE={mae:.2f}, R2={r2:.4f}")

    def select_best(self):
        best_name = min(self.results, key=lambda x: self.results[x]['RMSE'])
        self.best_model = self.models[best_name]
        print(f"\n>>> Best Model: {best_name}")
        return best_name


def plot_results(y_test, predictions, model_name):
    plt.figure(figsize=(15, 7))
    dates = y_test.index
    plt.plot(dates, y_test.values, label='Giá Thực Tế', color='blue', alpha=0.7)
    plt.plot(dates, predictions, label='Giá Dự Báo', color='red', linestyle='--')
    plt.title(f'So Sánh Giá Thực Tế và Dự Báo - Mô hình: {model_name}')
    plt.xlabel('Thời gian (Ngày)')
    plt.ylabel('Giá Cổ Phiếu')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_feature_importance(model, features):
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        
    if importances is not None:
        indices = np.argsort(importances)
        
        plt.figure(figsize=(10, 6))
        plt.title('Các yếu tố ảnh hưởng nhất đến giá')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Độ quan trọng')
        plt.show()

# ==========================================
# 6. MAIN EXECUTION (Luồng chính)
# ==========================================
if __name__ == "__main__":
    print("Bắt đầu hệ thống phân tích Machine Learning...\n")
    
    # B1: Tải dữ liệu
    raw_data = generate_stock_data(days=1500)
    
    # B2: Xử lý dữ liệu
    processed_data = apply_technical_indicators(raw_data)
    
    # B3: Chuẩn bị Train/Test
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_datasets(processed_data)
    
    # B4: Huấn luyện và So sánh
    predictor = StockPredictor()
    predictor.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # B5: Tìm mô hình tốt nhất
    best_name = predictor.select_best()
    
    # B6: Trực quan hóa kết quả
    best_preds = predictor.results[best_name]['Predictions']
    plot_results(y_test, best_preds, best_name)
    plot_feature_importance(predictor.best_model, feature_names)
    
    # B7: Lưu mô hình để sử dụng sau này
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(predictor.best_model, 'models/best_stock_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("\n--- Hoàn tất! Mô hình đã được lưu vào thư mục 'models/' ---")
