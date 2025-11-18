import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class TransformerBlock(nn.Layer):
    def __init__(self, model_dim=512, num_heads=8, ff_dim=2048, dropout=0.1):
        """
        Args:
            model_dim: Transformer的维度（即d_model）
            num_heads: 多头注意力的头数
            ff_dim: 前馈网络中间层维度
            dropout: dropout概率
        """
        super().__init__()
        self.model_dim = model_dim
        
        # 如果输入通道C不等于model_dim，需要先投影
        self.input_proj = nn.Linear(model_dim, model_dim)
        
        # 多头注意力
        self.multihead_attn = nn.MultiHeadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout)
        )
        
        # 归一化层
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状 [B, C, L]
        Returns:
            输出张量，形状 [B, C, L] (假设C=model_dim)
        """
        dim_flag = (len(x.shape) == 3)
        if dim_flag:
            B, L, C = x.shape
        else:
            B, C, H, W = x.shape
            x = x.reshape([B, C, -1])
            x = x.transpose([0, 2, 1])  # [B, L, C]
        
        # 1. 将输入从 [B, C, L] 转换为 [B, L, C] 并投影到model_dim
        # x = x.transpose([0, 2, 1])  # [B, L, C]
        if C != self.model_dim:
            x = self.input_proj(x)  # [B, L, model_dim]
        
        # 2. 多头注意力（带残差连接）
        # Paddle的MultiHeadAttention需要 [L, B, model_dim] 格式
        x_attn = x.transpose([1, 0, 2])  # [L, B, model_dim]
        attn_output = self.multihead_attn(x_attn, x_attn, x_attn)
        x = x + self.dropout1(attn_output).transpose([1, 0, 2])  # 恢复 [B, L, model_dim]
        x = self.norm1(x)
        
        # 3. 前馈网络（带残差连接）
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        
        if dim_flag:
            return x
        x = x.transpose([0, 2, 1])
        x = x.reshape([B, C, H, W])
        
        return x

# 示例用法
if __name__ == "__main__":
    B, C, L = 4, 64, 100  # 批次大小4，通道64，序列长度100
    model_dim = 128       # Transformer内部维度
    
    # 创建Transformer Block
    transformer = TransformerBlock(model_dim=model_dim)
    
    # 模拟输入数据 [B, C, L]
    x = paddle.randn([B, C, L])
    
    # 如果输入通道C不等于model_dim，需要先调整
    if C != model_dim:
        proj = nn.Linear(C, model_dim)
        x_proj = proj(x.transpose([0, 2, 1])).transpose([0, 2, 1])  # [B, model_dim, L]
    else:
        x_proj = x
    
    # 前向传播
    out = transformer(x_proj)
    print("输入形状:", x.shape)
    print("输出形状:", out.shape)  # 应为 [B, model_dim, L] 或 [B, C, L] 如果C==model_dim