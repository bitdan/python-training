import sys
import os
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

# 设置 matplotlib 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config.db_config import get_connection, close_connection

# 在文件开头导入部分后添加目录创建逻辑
model_dir = os.path.join(project_root, 'pth', 'photo')
os.makedirs(model_dir, exist_ok=True)

def get_data():
    """
    从数据库获取数据
    """
    conn = None
    try:
        # 获取数据库连接
        conn = get_connection()
        
        # 打印查询开始时间
        print(f"开始查询数据: {datetime.now()}")
        
        # 执行查询
        query = """
        SELECT create_time, supplier_delivery_quantity, scan_quantity, box_count 
        FROM wms_delivery_photo
        """
        df = pd.read_sql(query, conn)
        
        # 打印数据统计信息
        print(f"查询到的总记录数: {len(df)}")
        print(f"数据时间范围: {df['create_time'].min()} 到 {df['create_time'].max()}")
        
        return df
        
    except Exception as e:
        print(f"数据查询失败: {e}")
        raise e
    
    finally:
        # 确保连接被关闭
        close_connection(conn)

# 使用函数加载数据
df = get_data()
print("原始数据预览：")
print(df.head())

# 转换 create_time 为 pandas datetime 类型
df['create_time'] = pd.to_datetime(df['create_time'])

# 获取原始数据量
original_count = len(df)

# 定义重要特征列表（这些特征无论缺失值比例如何都要保留）
IMPORTANT_FEATURES = ['create_time', 'key_feature1', 'key_feature2']  # 替换为你的重要特征

# 分析每个特征的缺失值比例
missing_ratio = df.isnull().sum() / len(df) * 100
print("\n各特征缺失值比例（%）:")
print(missing_ratio)

# 设定缺失值比例阈值
MISSING_THRESHOLD = 50.0
high_missing_features = missing_ratio[
    (missing_ratio > MISSING_THRESHOLD) & 
    (~missing_ratio.index.isin(IMPORTANT_FEATURES))
].index.tolist()

if high_missing_features:
    print(f"\n缺失值比例超过{MISSING_THRESHOLD}%的特征:")
    for feature in high_missing_features:
        print(f"- {feature}: {missing_ratio[feature]:.2f}%")
    
    # 删除高缺失值的特征
    df = df.drop(columns=high_missing_features)
    print(f"\n已删除以上特征，剩余特征数量: {len(df.columns)}")

# 排除当天创建的数据
today = pd.Timestamp.now().normalize()
df = df[df['create_time'].dt.normalize() < today]

# 获取去除当天数据后的数量
after_date_filter = len(df)

# 处理剩余的缺失值
df = df.dropna()
final_count = len(df)

print(f"\n数据处理统计:")
print(f"原始数据总数: {original_count}")
print(f"去除当天数据后的记录数: {after_date_filter}")
print(f"过滤掉的当天记录数: {original_count - after_date_filter}")
print(f"去除缺失值后的最终记录数: {final_count}")
print(f"最终保留的特征: {list(df.columns)}")

# 对数值特征进行标准化处理
scaler = StandardScaler()
numeric_features = ['supplier_delivery_quantity', 'scan_quantity']
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# 构造时间特征
df['hour'] = df['create_time'].dt.hour
df['day_of_week'] = df['create_time'].dt.dayofweek
df['day_of_month'] = df['create_time'].dt.day
df['month'] = df['create_time'].dt.month

# 对时间特征进行周期性编码
def cyclical_encoding(data, col, max_val):
    data[f'{col}_sin'] = np.sin(2 * np.pi * data[col] / max_val)
    data[f'{col}_cos'] = np.cos(2 * np.pi * data[col] / max_val)

cyclical_encoding(df, 'hour', 24)
cyclical_encoding(df, 'day_of_week', 7)
cyclical_encoding(df, 'day_of_month', 31)
cyclical_encoding(df, 'month', 12)

# 添加星期几作为分类特征
df['day_name'] = df['create_time'].dt.day_name()

# 计算每个星期几的平均箱数
day_avg = df.groupby('day_name')['box_count'].mean().to_dict()
print("\n每个星期几的平均箱数:")
for day, avg in day_avg.items():
    print(f"{day}: {avg:.2f}")

# 添加day_avg_norm特征
df['day_avg_norm'] = df['day_name'].map(day_avg)
df['day_avg_norm'] = (df['day_avg_norm'] - df['day_avg_norm'].mean()) / df['day_avg_norm'].std()

# 添加前一周同一天的箱数作为特征
df = df.sort_values('create_time')
df['prev_week_same_day'] = df.groupby(df['create_time'].dt.dayofweek)['box_count'].shift(1)

# 然后再定义feature_columns
feature_columns = [
    'supplier_delivery_quantity', 'scan_quantity',
    'hour_sin', 'hour_cos',
    'day_of_week_sin', 'day_of_week_cos',
    'day_of_month_sin', 'day_of_month_cos',
    'month_sin', 'month_cos',
    'day_avg_norm'
]

# 现在可以安全地使用feature_columns
X = df[feature_columns].values
y = df['box_count'].values

# 对目标变量也进行标准化处理
y_scaler = StandardScaler()
y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

# 在数据预处理部分添加时间戳特征
df['timestamp'] = df['create_time'].apply(lambda x: x.timestamp())
min_ts = df['timestamp'].min()
max_ts = df['timestamp'].max()
df['timestamp_norm'] = (df['timestamp'] - min_ts) / (max_ts - min_ts)

# 切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 修改模型架构
class ImprovedTransformerPredictor(nn.Module):
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
        
        # 多头注意力层
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 周期性特征处理层
        self.periodic_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, 32),  # 合并周期性特征
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # 输入处理
        x = self.input_layer(x)
        
        # 提取周期性特征
        periodic_features = self.periodic_layer(x)
        
        # 自注意力机制
        x_unsqueezed = x.unsqueeze(0)  # 添加序列维度
        x_attn, _ = self.attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)
        x = x_attn.squeeze(0)  # 移除序列维度
        
        # 前馈网络
        x = self.feed_forward(x)
        
        # 合并周期性特征
        x = torch.cat([x, periodic_features], dim=1)
        
        # 输出层
        x = self.output_layer(x)
        return x

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 使用K折交叉验证
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 存储每个折的模型预测结果
fold_predictions = []
fold_models = []

# 在模型定义之后，训练循环之前添加损失函数定义
criterion = nn.HuberLoss(delta=1.0)  # 使用 Huber Loss，对异常值更稳健

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nTraining fold {fold + 1}/{n_splits}")
    
    # 准备数据
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # 转换为张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    
    # 初始化模型和优化器
    model = ImprovedTransformerPredictor(input_dim=len(feature_columns))
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False  # 设置 verbose=False 去除警告
    )
    
    # 训练循环
    best_val_loss = float('inf')
    patience = 20
    no_improve = 0
    
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs.squeeze(), y_val)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 保存当前fold的最佳模型到pth/photo目录
                model_path = os.path.join(model_dir, f'best_model_fold_{fold}.pth')
                torch.save(model.state_dict(), model_path)
                no_improve = 0
            else:
                no_improve += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/300, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # 加载最佳模型状态
    model_path = os.path.join(model_dir, f'best_model_fold_{fold}.pth')
    model.load_state_dict(torch.load(model_path))
    fold_models.append(model)

# 使用所有fold的模型进行集成预测
def ensemble_predict(models, X_input):
    predictions = []
    # 确保X_input是正确的形状
    if not isinstance(X_input, torch.Tensor):
        X_input = torch.FloatTensor(X_input)
    
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(X_input).squeeze().numpy()
            predictions.append(pred)
    
    # 取平均作为最终预测结果
    return np.mean(predictions, axis=0)

# 预测未来值
future_days = 7
today = pd.Timestamp.now().normalize()  # 获取当天日期（不含时间）
future_dates = [today + timedelta(days=i) for i in range(future_days)]  # 从当天开始预测7天
# 转换未来日期为归一化的时间戳特征
future_timestamps = [dt.timestamp() for dt in future_dates]
future_timestamp_norm = [(ts - min_ts) / (max_ts - min_ts) for ts in future_timestamps]

# 这里我们假设未来的 supplier_delivery_quantity 和 scan_quantity 与测试集均值相同
mean_supplier = df['supplier_delivery_quantity'].mean()
mean_scan = df['scan_quantity'].mean()

# 修改未来预测特征构造，加入星期几的平均值信息
future_features = []
for dt in future_dates:
    features = []
    # 添加标准化后的供应商数量和扫描数量
    features.extend([mean_supplier, mean_scan])  # 使用均值
    
    # 添加周期性时间特征
    hour = dt.hour
    day_of_week = dt.dayofweek
    day_of_month = dt.day
    month = dt.month
    
    # 获取该星期几的平均箱数，如果没有则使用整体平均值
    day_name = dt.day_name()
    day_avg_value = day_avg.get(day_name, df['box_count'].mean())
    # 将平均箱数标准化
    day_avg_norm = (day_avg_value - df['box_count'].mean()) / df['box_count'].std()
    
    features.extend([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * day_of_week / 7),
        np.cos(2 * np.pi * day_of_week / 7),
        np.sin(2 * np.pi * day_of_month / 31),
        np.cos(2 * np.pi * day_of_month / 31),
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        day_avg_norm  # 添加星期几平均箱数作为特征
    ])
    
    future_features.append(features)

X_future = torch.tensor(future_features, dtype=torch.float32)

# 预测未来值
future_predictions = ensemble_predict(fold_models, future_features)
future_predictions = y_scaler.inverse_transform(future_predictions.reshape(-1, 1))

# 确保预测结果为正整数
future_predictions = np.maximum(1, np.round(future_predictions))

# 更新预测结果DataFrame
future_df = pd.DataFrame({
    'date': future_dates,
    'day_of_week': [dt.day_name() for dt in future_dates],
    'predicted_box_count': future_predictions.flatten()
})

print("\n未来7天箱数预测（从今天开始）：")
print(future_df)

# 可视化预测结果
plt.figure(figsize=(12, 6))
plt.plot(future_df['date'], future_df['predicted_box_count'], marker='o', label='预测值')
plt.title('未来7天箱数预测（从今天开始）')
plt.xlabel('日期')
plt.ylabel('预测箱数')
plt.grid(True)
plt.xticks(rotation=45)
for i, (date, count, day) in enumerate(zip(future_df['date'], future_df['predicted_box_count'], future_df['day_of_week'])):
    plt.annotate(f"{day}\n{int(count)}", (date, count), textcoords="offset points", xytext=(0,10), ha='center')
plt.legend()
plt.tight_layout()
plt.show()

# 确保保存图片的目录存在
image_dir = os.path.join(project_root, 'data', 'photo')
os.makedirs(image_dir, exist_ok=True)
plt.savefig(os.path.join(image_dir, 'photo_prediction.png'))
plt.close()

# 打印预测的置信区间
confidence_interval = np.std([model(torch.FloatTensor(future_features)).detach().numpy() 
                            for model in fold_models], axis=0) * 1.96
print("\n预测置信区间（从今天开始）：")
for i, (date, pred) in enumerate(zip(future_dates, future_predictions)):
    lower = max(1, int(np.round(pred[0] - confidence_interval[i])))
    upper = int(np.round(pred[0] + confidence_interval[i]))
    print(f"{date.date()}: {int(pred[0])} (95% CI: {lower}-{upper})")

# 保存预测结果到CSV文件
os.makedirs(os.path.join(project_root, 'data'), exist_ok=True)
future_df['predicted_box_count'] = future_df['predicted_box_count'].astype(int)
future_df.to_csv(os.path.join(project_root, 'data', 'photo_predictions.csv'), index=False)
print(f"\n预测结果已保存到 {os.path.join(project_root, 'data', 'photo_predictions.csv')}")
