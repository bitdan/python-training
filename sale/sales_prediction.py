import pymysql
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import KFold
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.db_config import get_connection, close_connection
import argparse

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 添加全局文件路径配置
DATA_DIR = 'data'
SALES_DATA_FILE = f'{DATA_DIR}/sales_data.csv'
PREDICTIONS_FILE = f'{DATA_DIR}/sales_predictions.csv'
PLOT_FILE = f'{DATA_DIR}/sales_prediction.png'

def get_data(start_date=None, end_date=None, batch_size=10000):
    """
    分批次从数据库获取数据
    """
    conn = None
    all_data = []
    try:
        conn = get_connection()
        print(f"开始查询数据: {datetime.now()}")
        
        # 首先获取数据总量
        count_query = """
        SELECT COUNT(*) as total 
        FROM sale_amazon_order 
        WHERE deleted = 0
        """
        if start_date and end_date:
            count_query += f" AND create_time BETWEEN '{start_date}' AND '{end_date}'"
            
        total_count = pd.read_sql(count_query, conn).iloc[0]['total']
        
        # 分批次查询
        for offset in range(0, total_count, batch_size):
            query = """
            SELECT 
                create_time,
                purchase_date,
                currency_code,
                amount,
                number_of_items_shipped,
                number_of_items_unshipped,
                marketplace_id,
                is_prime,
                site,
                order_status
            FROM sale_amazon_order 
            WHERE deleted = 0
            """
            if start_date and end_date:
                query += f" AND create_time BETWEEN '{start_date}' AND '{end_date}'"
            
            query += f" LIMIT {batch_size} OFFSET {offset}"
            
            df_batch = pd.read_sql(query, conn)
            all_data.append(df_batch)
            
            print(f"已获取 {min(offset + batch_size, total_count)}/{total_count} 条记录")
        
        final_df = pd.concat(all_data, ignore_index=True)
        print(f"查询完成，总记录数: {len(final_df)}")
        
        return final_df
        
    except Exception as e:
        print(f"数据查询失败: {e}")
        raise e
    finally:
        close_connection(conn)

def save_data(df, filename=SALES_DATA_FILE):
    """保存数据到本地"""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        print(f"数据已保存到: {filename}")
    except Exception as e:
        print(f"保存数据时出错: {e}")

def load_local_data(filename=SALES_DATA_FILE):
    """从本地加载数据"""
    try:
        if os.path.exists(filename):
            print(f"正在从本地加载数据: {filename}")
            df = pd.read_csv(filename)
            if len(df) > 0:  # 确保数据不为空
                return df
        print(f"找不到有效的本地数据文件: {filename}")
        return None
    except Exception as e:
        print(f"加载本地数据时出错: {e}")
        return None

def preprocess_data(df):
    """数据预处理"""
    # 转换时间列
    df['create_time'] = pd.to_datetime(df['create_time'])
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    
    # 转换金额为数值类型
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # 按天聚合销售额
    daily_sales = df.groupby(df['create_time'].dt.date)['amount'].sum().reset_index()
    daily_sales['create_time'] = pd.to_datetime(daily_sales['create_time'])
    
    return daily_sales

class SalesPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        
        # 注意力层
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        
        # 重塑输入以适应线性层
        x = x.view(-1, features)
        x = self.input_layer(x)
        x = x.view(batch_size, seq_len, self.hidden_dim)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 注意力机制
        # 调整维度以适应注意力层
        lstm_out = lstm_out.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.permute(1, 0, 2)  # [batch_size, seq_len, hidden_dim]
        
        # 输出层
        x = self.output_layer(attn_out[:, -1, :])  # 只使用最后一个时间步
        return x

def prepare_features(df):
    """准备特征数据"""
    # 时间特征
    df['year'] = df['create_time'].dt.year
    df['month'] = df['create_time'].dt.month
    df['day'] = df['create_time'].dt.day
    df['day_of_week'] = df['create_time'].dt.dayofweek
    
    # 移动平均特征
    df['sales_ma7'] = df['amount'].rolling(window=7, min_periods=1).mean()
    df['sales_ma30'] = df['amount'].rolling(window=30, min_periods=1).mean()
    
    # 滞后特征
    for i in [1, 7, 30]:
        df[f'sales_lag_{i}'] = df['amount'].shift(i)
    
    # 周期性编码
    def cyclical_encoding(data, col, max_val):
        data[f'{col}_sin'] = np.sin(2 * np.pi * data[col] / max_val)
        data[f'{col}_cos'] = np.cos(2 * np.pi * data[col] / max_val)
    
    cyclical_encoding(df, 'month', 12)
    cyclical_encoding(df, 'day', 31)
    cyclical_encoding(df, 'day_of_week', 7)
    
    return df

def train_model(df, prediction_days=30, sequence_length=60):
    """训练模型"""
    # 准备特征
    df = prepare_features(df)
    
    # 定义特征列
    feature_columns = [
        'sales_ma7', 'sales_ma30',
        'sales_lag_1', 'sales_lag_7', 'sales_lag_30',
        'month_sin', 'month_cos',
        'day_sin', 'day_cos',
        'day_of_week_sin', 'day_of_week_cos'
    ]
    
    # 删除包含NaN的行
    df = df.dropna()
    
    # 准备数据
    X = df[feature_columns].values
    y = df['amount'].values
    
    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 创建序列数据
    X_sequences = []
    y_sequences = []
    
    for i in range(len(df) - sequence_length):
        X_sequences.append(X[i:(i + sequence_length)])
        y_sequences.append(y[i:(i + sequence_length)])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    # 分割训练集和测试集
    train_size = int(len(X_sequences) * 0.8)
    X_train = X_sequences[:train_size]
    y_train = y_sequences[:train_size]
    X_test = X_sequences[train_size:]
    y_test = y_sequences[train_size:]
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # 初始化模型
    model = SalesPredictor(input_dim=len(feature_columns))
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # 训练模型
    epochs = 100
    best_loss = float('inf')
    patience = 10
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train[:, -1])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test[:, -1])
            scheduler.step(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'best_sales_model.pth')
                no_improve = 0
            else:
                no_improve += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        
        if no_improve >= patience:
            print("Early stopping!")
            break
    
    return model, scaler_X, scaler_y, feature_columns

def predict_future(model, df, scaler_X, scaler_y, feature_columns, days_to_predict=30):
    """预测未来销售额"""
    model.eval()
    last_sequence = df[feature_columns].iloc[-60:].values  # 使用最后60天的数据
    last_sequence = scaler_X.transform(last_sequence)
    
    predictions = []
    current_sequence = torch.FloatTensor(last_sequence).unsqueeze(0)
    
    for _ in range(days_to_predict):
        with torch.no_grad():
            next_pred = model(current_sequence).squeeze().item()
            predictions.append(next_pred)
            
            # 更新序列
            current_sequence = torch.roll(current_sequence, -1, dims=1)
            current_sequence[0, -1] = next_pred
    
    # 反转标准化
    predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # 生成预测日期
    last_date = df['create_time'].iloc[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
    
    return pd.DataFrame({
        'date': future_dates,
        'predicted_sales': predictions.flatten()
    })

def plot_predictions(historical_data, predictions):
    """绘制预测结果"""
    plt.figure(figsize=(15, 7))
    
    # 绘制历史数据
    plt.plot(historical_data['create_time'].iloc[-90:], 
             historical_data['amount'].iloc[-90:], 
             label='历史销售额', color='blue')
    
    # 绘制预测数据
    plt.plot(predictions['date'], predictions['predicted_sales'], 
             label='预测销售额', color='red', linestyle='--')
    
    plt.title('销售额预测')
    plt.xlabel('日期')
    plt.ylabel('销售额')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('sales_prediction.png')
    plt.close()

def main():
    # 直接设置参数，不使用命令行
    args = {
        'update': False,  # 是否更新数据源
        'start_date': None,  # 可以设置具体日期，如 '2024-01-01'
        'end_date': None,    # 可以设置具体日期，如 '2024-03-01'
        'predict_days': 30,   # 预测天数
    }

    # 获取数据
    if args['update']:
        print("正在更新数据源...")
        df = get_data(args['start_date'], args['end_date'])
        save_data(df)
    else:
        print(f"尝试加载本地数据文件: {SALES_DATA_FILE}")
        df = load_local_data()
        if df is None:
            print("本地数据不存在或无效，正在从数据库获取...")
            df = get_data(args['start_date'], args['end_date'])
            save_data(df)

    # 数据预处理
    daily_sales = preprocess_data(df)
    
    # 训练模型
    print("开始训练模型...")
    model, scaler_X, scaler_y, feature_columns = train_model(daily_sales)
    
    # 预测未来销售额
    print("生成预测...")
    predictions = predict_future(model, daily_sales, scaler_X, scaler_y, 
                               feature_columns, args['predict_days'])
    
    # 绘制预测结果
    plot_predictions(daily_sales, predictions)
    
    # 保存预测结果
    os.makedirs(DATA_DIR, exist_ok=True)
    predictions.to_csv(PREDICTIONS_FILE, index=False)
    plt.savefig(PLOT_FILE)
    print(f"预测完成！结果已保存到 {PREDICTIONS_FILE} 和 {PLOT_FILE}")

if __name__ == "__main__":
    main()
