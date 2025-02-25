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
import matplotlib
# matplotlib.use('Agg')  # 在导入 pyplot 之前设置

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 添加全局文件路径配置
DATA_DIR = 'data'
SALES_DATA_FILE = f'{DATA_DIR}/sales_data.csv'
PREDICTIONS_FILE = f'{DATA_DIR}/sales_predictions.csv'
PLOT_FILE = f'{DATA_DIR}/sales_prediction.png'
PTH_DIR = 'pth/sale'
MODEL_PATH = f'{PTH_DIR}/best_sales_model.pth'

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
            nn.Dropout(0.2)  # 增加dropout比例以减少过拟合
        )
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.2,  # 增加dropout比例
            batch_first=True,
            bidirectional=True  # 使用双向LSTM捕获更多时序信息
        )
        
        # 注意力层
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, dropout=0.2)  # 适应双向LSTM的输出维度
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # 适应双向LSTM的输出维度
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        
        # 重塑输入以适应线性层
        x = x.view(-1, features)
        x = self.input_layer(x)
        x = x.view(batch_size, seq_len, self.hidden_dim)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)  # lstm_out形状: [batch_size, seq_len, hidden_dim*2]
        
        # 注意力机制
        lstm_out = lstm_out.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim*2]
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.permute(1, 0, 2)  # [batch_size, seq_len, hidden_dim*2]
        
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
        y_sequences.append(y[i + sequence_length])  # 只预测序列后的下一个值
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    # 分割训练集和测试集
    train_size = int(len(X_sequences) * 0.8)
    X_train = X_sequences[:train_size]
    y_train = y_sequences[:train_size]
    X_test = X_sequences[train_size:]
    y_test = y_sequences[train_size:]
    
    # 转换为tensor
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)
    
    # 初始化模型
    model = SalesPredictor(input_dim=len(feature_columns))
    criterion = nn.HuberLoss(delta=0.5)  # 降低delta值使损失函数对异常值更加鲁棒
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)  # 减小weight_decay
    
    # 添加批次训练
    batch_size = 64  # 增加批次大小以加速训练
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    # 添加验证集数据加载器
    val_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size
    )

    # 修改训练参数
    epochs = 300  # 增加训练轮数
    patience = 30  # 增加早停耐心值
    best_loss = float('inf')
    no_improve = 0
    
    # 使用更稳定的学习率调度
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.002,  # 略微增加最大学习率
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1e4
    )

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        # 批次训练
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 增加裁剪阈值以允许更大的梯度
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
        
        # 验证
        model.eval()
        total_val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                val_outputs = model(batch_X)
                val_loss = criterion(val_outputs, batch_y)
                total_val_loss += val_loss.item()
                
                val_predictions.extend(val_outputs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        # 计算平均损失
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        # 计算验证集上的相对误差
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        # 避免除以零
        non_zero_targets = np.where(np.abs(val_targets) > 1e-6, val_targets, 1e-6)
        relative_error = np.mean(np.abs((val_predictions - val_targets) / non_zero_targets)) * 100
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            # 创建目录并保存模型
            os.makedirs(PTH_DIR, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'scaler_X': scaler_X,  # 保存标准化器以便预测时使用
                'scaler_y': scaler_y,
            }, MODEL_PATH)
            no_improve = 0
        else:
            no_improve += 1
        
        if epoch % 10 == 0:  # 减少日志输出频率
            print(f"Epoch {epoch}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Relative Error: {relative_error:.2f}% | "
                  f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        if no_improve >= patience and epoch > 100:  # 确保至少训练100轮
            print("Early stopping triggered!")
            break
    
    # 加载最佳模型
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, scaler_X, scaler_y, feature_columns

def predict_future(model, df, scaler_X, scaler_y, feature_columns, days_to_predict=30):
    """预测未来销售额"""
    model.eval()
    
    # 准备最新的特征数据
    latest_df = df.copy()
    
    # 预测未来销售额
    future_dates = []
    predictions = []
    last_date = df['create_time'].iloc[-1]
    
    # 使用滑动窗口进行预测
    window_size = 60
    
    # 获取初始序列
    current_features = latest_df[feature_columns].iloc[-window_size:].values
    current_features = scaler_X.transform(current_features)
    current_sequence = torch.FloatTensor(current_features).unsqueeze(0)
    
    for i in range(days_to_predict):
        # 预测下一天
        with torch.no_grad():
            next_pred_scaled = model(current_sequence).item()
        
        # 反标准化预测值
        next_pred = scaler_y.inverse_transform([[next_pred_scaled]])[0][0]
        
        # 计算下一天的日期
        next_date = last_date + timedelta(days=i+1)
        future_dates.append(next_date)
        predictions.append(next_pred)
        
        # 为下一次预测准备新的特征
        # 创建新的一行数据
        new_row = pd.DataFrame({
            'create_time': [next_date],
            'amount': [next_pred]
        })
        
        # 添加到原始数据中
        temp_df = pd.concat([latest_df, new_row], ignore_index=True)
        
        # 重新计算特征
        temp_df = prepare_features(temp_df)
        
        # 获取最新的特征向量
        new_features = temp_df[feature_columns].iloc[-1:].values
        new_features_scaled = scaler_X.transform(new_features)
        
        # 更新序列：移除最早的一天，添加新预测的一天
        current_sequence = torch.roll(current_sequence, -1, dims=1)
        current_sequence[0, -1] = torch.FloatTensor(new_features_scaled[0])
        
        # 更新latest_df为新的temp_df
        latest_df = temp_df
    
    return pd.DataFrame({
        'date': future_dates,
        'predicted_sales': predictions
    })

def plot_predictions(historical_data, predictions):
    """绘制预测结果"""
    plt.figure(figsize=(15, 7))
    
    # 计算移动平均线以平滑历史数据
    historical_data['amount_ma7'] = historical_data['amount'].rolling(window=7).mean()
    
    # 绘制历史数据
    plt.plot(historical_data['create_time'].iloc[-90:], 
             historical_data['amount'].iloc[-90:], 
             label='历史销售额', color='blue', alpha=0.5)
    
    # 绘制历史数据的移动平均线
    plt.plot(historical_data['create_time'].iloc[-90:], 
             historical_data['amount_ma7'].iloc[-90:], 
             label='历史销售额(7日均值)', color='darkblue', linewidth=2)
    
    # 绘制预测数据
    plt.plot(predictions['date'], predictions['predicted_sales'], 
             label='预测销售额', color='red', linestyle='--', linewidth=2)
    
    # 添加预测区间的背景色
    min_date = predictions['date'].min()
    max_date = predictions['date'].max()
    plt.axvspan(min_date, max_date, color='red', alpha=0.1)
    
    # 添加标题和标签
    plt.title('销售额预测 (未来30天)', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('销售额', fontsize=12)
    
    # 格式化x轴日期
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图表
    os.makedirs(os.path.dirname(PLOT_FILE), exist_ok=True)
    plt.savefig(PLOT_FILE, dpi=300)
    print(f"预测图表已保存到 {PLOT_FILE}")
    
    # 显示图表
    plt.show()
    plt.close()

def main():
    # 直接设置参数，不使用命令行
    args = {
        'update': False,  # 是否更新数据源
        'start_date': None,  # 可以设置具体日期，如 '2024-01-01'
        'end_date': None,    # 可以设置具体日期，如 '2024-03-01'
        'predict_days': 30,   # 预测天数
    }

    # 确保数据目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PTH_DIR, exist_ok=True)

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
    
    # 检查是否已有训练好的模型
    if os.path.exists(MODEL_PATH):
        print(f"加载已有模型: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH)
        model = SalesPredictor(input_dim=len(checkpoint.get('feature_columns', [])) 
                              if 'feature_columns' in checkpoint 
                              else 11)  # 默认特征数量
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 尝试从checkpoint加载scaler
        if 'scaler_X' in checkpoint and 'scaler_y' in checkpoint:
            scaler_X = checkpoint['scaler_X']
            scaler_y = checkpoint['scaler_y']
            print("从模型中加载标准化器")
        else:
            # 如果没有保存scaler，重新训练模型
            print("模型中没有保存标准化器，重新训练模型...")
            model, scaler_X, scaler_y, feature_columns = train_model(daily_sales)
    else:
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
    predictions.to_csv(PREDICTIONS_FILE, index=False)
    print(f"预测结果已保存到 {PREDICTIONS_FILE}")

if __name__ == "__main__":
    main()
