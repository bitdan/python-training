import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from config.db_config import get_connection, close_connection

def get_data():
    """
    从数据库获取数据
    """
    conn = None
    try:
        # 获取数据库连接
        conn = get_connection()
        
        # 执行查询
        query = """
        SELECT supplier_delivery_quantity, scan_quantity, box_count
        FROM wms_delivery_photo
        WHERE deleted = 0
        """
        df = pd.read_sql(query, conn)
        
        # 打印数据统计信息
        print(f"查询到的总记录数: {len(df)}")
        
        return df
        
    except Exception as e:
        print(f"数据查询失败: {e}")
        raise e
    
    finally:
        # 确保连接被关闭
        close_connection(conn)

# 读取数据
df = get_data()
print("原始数据预览：")
# print(df.head())

# 如果有缺失值可以进行处理，这里我们直接丢弃
df = df.dropna()

# 定义输入特征和目标变量（预测到货箱数）
# 此处只选择了 supplier_delivery_quantity 和 scan_quantity 作为输入特征
X = df[['supplier_delivery_quantity', 'scan_quantity']].values
y = df['box_count'].values

# 切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 定义 Transformer 模型
class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, nhead=2):
        super().__init__()
        # 将输入特征映射到隐藏维度
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # Transformer Encoder 需要输入形状为 (sequence_length, batch_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输出层，预测一个标量
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x 形状：[batch_size, input_dim]
        x = self.relu(self.fc1(x))  # 映射到 hidden_dim
        # 添加 sequence 维度，设序列长度为1
        x = x.unsqueeze(0)  # 变为 [1, batch_size, hidden_dim]
        x = self.transformer_encoder(x)  # Transformer 处理
        # 去除序列维度并预测
        x = x.squeeze(0)  # [batch_size, hidden_dim]
        x = self.fc2(x)   # [batch_size, 1]
        return x

# 实例化模型
model = TransformerPredictor(input_dim=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模型训练
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    predictions = model(X_train).squeeze()
    
    # 计算损失
    loss = criterion(predictions, y_train)
    
    # 反向传播和优化
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 测试与预测
model.eval()
with torch.no_grad():
    predictions = model(X_test).squeeze()
    # 计算平均绝对误差
    error = torch.abs(predictions - y_test).mean().item()
    print(f"平均预测误差: {error:.2f}")