
from typing import Union
import torch
from escnn import gspaces, nn
from escnn.group import CyclicGroup
from einops import rearrange, repeat  
from EfficientFlow.model.diffusion.conditional_unet1d import ConditionalUnet1D
from EfficientFlow.model.rdtmodels.rdt.model import RDT 

# 定义一个等变扩散U-Net模型
class Equidit(torch.nn.Module):
    def __init__(self, act_emb_dim, local_cond_dim, global_cond_dim, diffusion_step_embed_dim, down_dims, kernel_size, n_groups, cond_predict_scale, N):
        super().__init__()
        # 初始化条件U-Net模型
        # self.unet = ConditionalUnet1D(
        #     input_dim=act_emb_dim,          # 输入维度：动作嵌入维度
        #     local_cond_dim=local_cond_dim,  # 局部条件维度（时序条件）
        #     global_cond_dim=global_cond_dim, # 全局条件维度（场景级条件）
        #     diffusion_step_embed_dim=diffusion_step_embed_dim, # 扩散步嵌入维度
        #     down_dims=down_dims,            # U-Net下采样维度列表
        #     kernel_size=kernel_size,        # 卷积核尺寸
        #     n_groups=n_groups,              # 分组归一化的组数
        #     cond_predict_scale=cond_predict_scale # 条件预测缩放因子
        # )
        self.model = RDT(
            output_dim=act_emb_dim,
            horizon=16,
            hidden_size=512,
            depth=24,
            num_heads=8,            
            img_cond_len=global_cond_dim,            
            #img_pos_embed_config=img_pos_embed_config,
            dtype=torch.float32,
        )
        # 定义循环群的大小
         # 对称性参数配置
        self.N = N  # 循环群的阶数（旋转离散化个数）
        self.group = gspaces.no_base_space(CyclicGroup(self.N))  # 创建循环群（无空间基）
        self.order = self.N  # 群阶数别名
        # 定义等变特征场类型
        self.act_type = nn.FieldType(
            self.group, 
            act_emb_dim * [self.group.regular_repr]  # 正则表示（regular representation）
        )
        # 输出层（等变线性变换）
        self.out_layer = nn.Linear(self.act_type, self.getOutFieldType())
        # 动作编码器（等变序列模块）
        self.enc_a = nn.SequentialModule(
            nn.Linear(self.getOutFieldType(), self.act_type),  # 等变线性层
            nn.ReLU(self.act_type)  # 等变ReLU激活
        )

    # 定义输出场类型（混合不可约表示和平凡表示）
    def getOutFieldType(self):
        return nn.FieldType(
            self.group,
            4 * [self.group.irrep(1)] # 8  4个二维旋转不可约表示（每个占2通道，共8通道）
            + 2 * [self.group.trivial_repr], # 2   2个平凡表示（旋转不变量，共2通道）
        )

    # 定义输出函数
    def getOutput(self, conv_out):
        """将网络原始输出解析为结构化动作参数"""
        # 分解输出张量为具体组件（SE3版本）
        xy = conv_out[:, 0:2]    # XY平移分量
        cos1 = conv_out[:, 2:3]  # 第一个旋转角的余弦
        sin1 = conv_out[:, 3:4]  # 第一个旋转角的正弦
        cos2 = conv_out[:, 4:5]  # 第二个旋转角的余弦
        sin2 = conv_out[:, 5:6]  # 第二个旋转角的正弦
        cos3 = conv_out[:, 6:7]  # 第三个旋转角的余弦
        sin3 = conv_out[:, 7:8]  # 第三个旋转角的正弦
        z = conv_out[:, 8:9]     # Z轴平移分量
        g = conv_out[:, 9:10]    # 夹爪状态

        # 构造动作张量
        action = torch.cat((xy, z, cos1, cos2, cos3, sin1, sin2, sin3, g), dim=1)
        return action
    
    # 定义动作几何张量
    def getActionGeometricTensor(self, act):
        # 获取批次大小
        batch_size = act.shape[0] # batch_size=2048   [2048,10]
         # 分解动作分量
        xy = act[:, 0:2]  # XY坐标（平面平移）
        z = act[:, 2:3]   # Z坐标（高度）
        rot = act[:, 3:9]  # 三维旋转参数（3组cos/sin）
        g = act[:, 9:]     # 夹爪控制信号

        # 构造几何张量
        cat = torch.cat(
            (
                xy.reshape(batch_size, 2),
                rot[:, 0].reshape(batch_size, 1),
                rot[:, 3].reshape(batch_size, 1),
                rot[:, 1].reshape(batch_size, 1),
                rot[:, 4].reshape(batch_size, 1),
                rot[:, 2].reshape(batch_size, 1),
                rot[:, 5].reshape(batch_size, 1),
                z.reshape(batch_size, 1),
                g.reshape(batch_size, 1),
            ),
            dim=1,
        )
        return nn.GeometricTensor(cat, self.getOutFieldType())# 应用预定义的场类型
    
    # 定义前向传播函数
    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # 获取批次大小和时间步长
        B, T = sample.shape[:2]
        # 重塑样本张量
        sample = rearrange(sample, "b t d -> (b t) d") #
        # 获取动作几何张量
        sample = self.getActionGeometricTensor(sample)
        # 编码器输出 enc_a动作编码器（等变序列模块）
        enc_a_out = self.enc_a(sample).tensor.reshape(B, T, -1)  #
        # # 重塑编码器输出 准备U-Net输入（扩展组维度）
        enc_a_out = rearrange(enc_a_out, "b t (c f) -> (b f) t c", f=self.order) #8*b 16 1024
        # 处理时间步长
        if type(timestep) == torch.Tensor and len(timestep.shape) == 1:
            timestep = repeat(timestep, "b -> (b f)", f=self.order)#
        # 处理局部条件
        if local_cond is not None:
            local_cond = rearrange(local_cond, "b t (c f) -> (b f) t c", f=self.order)  
        # 处理全局条件
        if global_cond is not None: #[65,  4096] 
            global_cond = rearrange(global_cond, "b (c f) -> (b f) c", f=self.order).unsqueeze(1) #[65, 4096] => 65*8 512    self.order=8  
            #global_cond = global_cond.repeat(1, 2).unsqueeze(1)
        # Dit输出
        out = self.model(enc_a_out,timestep, global_cond)#x, t, img_c, img_mask=None
        
        # 后处理：恢复原始维度结构
        # 重塑U-Net输出
        out = rearrange(out, "(b f) t c -> (b t) (c f)", f=self.order)  #b 65 t 16 c 128 f 8   ->65*16 1024
        # 转换为几何张量
        out = nn.GeometricTensor(out, self.act_type)
        # 输出层
        out = self.out_layer(out).tensor.reshape(B * T, -1)

        # 获取最终输出
        out = self.getOutput(out)
        # 重塑输出
        out = rearrange(out, "(b t) n -> b t n", b=B)
        return out #[128, 16, 10]


# 定义一个等变扩散U-Net模型（SE2版本）轻量版等变扩散模型（SE2对称性，平面运动）
class EquiDiffusionUNetSE2(torch.nn.Module):
    def __init__(self, act_emb_dim, local_cond_dim, global_cond_dim, diffusion_step_embed_dim, down_dims, kernel_size, n_groups, cond_predict_scale, N):

        super().__init__()
        # 初始化条件U-Net模型
        self.unet = ConditionalUnet1D(
            input_dim=act_emb_dim,
            local_cond_dim=local_cond_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
        # 定义循环群的大小
        self.N = N
        # 定义群作用
        self.group = gspaces.no_base_space(CyclicGroup(self.N))
        # 定义群作用的阶数
        self.order = self.N
        # 定义激活类型
        self.act_type = nn.FieldType(self.group, act_emb_dim * [self.group.regular_repr])
        # 定义输出层
        self.out_layer = nn.Linear(self.act_type, 
                                   self.getOutFieldType())
        # 定义编码器
        self.enc_a = nn.SequentialModule(
            nn.Linear(self.getOutFieldType(), self.act_type), 
            nn.ReLU(self.act_type)
        )

    # 定义输出场类型
    def getOutFieldType(self):
        return nn.FieldType(
            self.group,
            2 * [self.group.irrep(1)] # 4  # 2个二维旋转表示（共4通道）
            + 2 * [self.group.trivial_repr], # 2
        )

    # 定义输出函数
    def getOutput(self, conv_out):
        """SE2版本输出解析（仅单个旋转角）"""
        xy = conv_out[:, 0:2]    # 平面平移
        cos1 = conv_out[:, 2:3]  # 旋转角余弦
        sin1 = conv_out[:, 3:4]  # 旋转角正弦
        z = conv_out[:, 4:5]     # 高度（不变）
        g = conv_out[:, 5:6]     # 夹爪

        # 构造动作张量
        action = torch.cat((xy, z, cos1, sin1, g), dim=1)
        return action
    
    # 定义动作几何张量
    def getActionGeometricTensor(self, act):
        # 获取批次大小
        batch_size = act.shape[0]
        # 提取xy坐标
        xy = act[:, 0:2]
        # 提取z坐标
        z = act[:, 2:3]
        # 提取旋转角度的余弦值仅处理单个旋转角
        cos = act[:, 3:4]
        # 提取旋转角度的正弦值
        sin = act[:, 4:5]
        # 提取g值
        g = act[:, 5:]

        # 构造几何张量
        cat = torch.cat(
            (
                xy.reshape(batch_size, 2),
                cos.reshape(batch_size, 1),
                sin.reshape(batch_size, 1),
                z.reshape(batch_size, 1),
                g.reshape(batch_size, 1),
            ),
            dim=1,
        )
        return nn.GeometricTensor(cat, self.getOutFieldType())
    
    # 定义前向传播函数
    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # 获取批次大小和时间步长
        B, T = sample.shape[:2]
        # 重塑样本张量
        sample = rearrange(sample, "b t d -> (b t) d")
        # 获取动作几何张量
        sample = self.getActionGeometricTensor(sample)
        # 编码器输出
        enc_a_out = self.enc_a(sample).tensor.reshape(B, T, -1)
        # 重塑编码器输出
        enc_a_out = rearrange(enc_a_out, "b t (c f) -> (b f) t c", f=self.order)
        # 处理时间步长
        if type(timestep) == torch.Tensor and len(timestep.shape) == 1:
            timestep = repeat(timestep, "b -> (b f)", f=self.order)
        # 处理局部条件
        if local_cond is not None:
            local_cond = rearrange(local_cond, "b t (c f) -> (b f) t c", f=self.order)
        # 处理全局条件
        if global_cond is not None:
            global_cond = rearrange(global_cond, "b (c f) -> (b f) c", f=self.order)
        # U-Net输出
        out = self.unet(enc_a_out, timestep, local_cond, global_cond, **kwargs)
        # 重塑U-Net输出
        out = rearrange(out, "(b f) t c -> (b t) (c f)", f=self.order)
        # 转换为几何张量
        out = nn.GeometricTensor(out, self.act_type)
        # 输出层
        out = self.out_layer(out).tensor.reshape(B * T, -1)
        # 获取最终输出
        out = self.getOutput(out)
        # 重塑输出
        out = rearrange(out, "(b t) n -> b t n", b=B)
        return out