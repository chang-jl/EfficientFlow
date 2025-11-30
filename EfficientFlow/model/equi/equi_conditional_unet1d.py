
from typing import Union
import torch
from escnn import gspaces, nn
from escnn.group import CyclicGroup
from einops import rearrange, repeat  
from EfficientFlow.model.diffusion.conditional_unet1d import ConditionalUnet1D,ConditionalUnet1DShortcut,MeanflowConditionalUnet1D

# 定义一个等变扩散U-Net模型
class EquiDiffusionUNet(torch.nn.Module):
    def __init__(self, act_emb_dim, local_cond_dim, global_cond_dim, diffusion_step_embed_dim, down_dims, kernel_size, n_groups, cond_predict_scale, N):
        super().__init__()
        # 初始化条件U-Net模型
        self.unet = ConditionalUnet1D(
            input_dim=act_emb_dim,          # 输入维度：动作嵌入维度
            local_cond_dim=local_cond_dim,  # 局部条件维度（时序条件）
            global_cond_dim=global_cond_dim, # 全局条件维度（场景级条件）
            diffusion_step_embed_dim=diffusion_step_embed_dim, # 扩散步嵌入维度
            down_dims=down_dims,            # U-Net下采样维度列表
            kernel_size=kernel_size,        # 卷积核尺寸
            n_groups=n_groups,              # 分组归一化的组数
            cond_predict_scale=cond_predict_scale # 条件预测缩放因子
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
        sample = rearrange(sample, "b t d -> (b t) d")
        # 获取动作几何张量
        sample = self.getActionGeometricTensor(sample)
        # 编码器输出 enc_a动作编码器（等变序列模块）
        enc_a_out = self.enc_a(sample).tensor.reshape(B, T, -1)  #[128, 16, 512]
        # 重塑编码器输出 准备U-Net输入（扩展组维度）
        enc_a_out = rearrange(enc_a_out, "b t (c f) -> (b f) t c", f=self.order) #[1024, 16, 64] self.order=8
        '''
        重排前：张量表示每个 batch 元素是一个时间序列，每个时间步有 c * f 个特征值，这些特征值被分组为 c 个特征组，每个组有 f 个特征。
        重排后：张量表示每个原始 batch 元素被扩展为 f 个新的 batch 元素，每个新的 batch 元素对应一个特征组，特征组的特征值被重新组织为 c 个特征。
        '''

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

        # 后处理：恢复原始维度结构
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
        return out #[128, 16, 10]


# class EquiMeanflowDit(torch.nn.Module):
#     def __init__(self, act_emb_dim, local_cond_dim, global_cond_dim, diffusion_step_embed_dim, down_dims, kernel_size, n_groups, cond_predict_scale, N):
#         super().__init__()
#         # 初始化条件U-Net模型
#         self.dit =MFDiT(input_dim=act_emb_dim)
#         # MeanflowConditionalUnet1D(
#         #     input_dim=act_emb_dim,          # 输入维度：动作嵌入维度
#         #     local_cond_dim=local_cond_dim,  # 局部条件维度（时序条件）
#         #     global_cond_dim=global_cond_dim, # 全局条件维度（场景级条件）
#         #     diffusion_step_embed_dim=diffusion_step_embed_dim, # 扩散步嵌入维度
#         #     down_dims=down_dims,            # U-Net下采样维度列表
#         #     kernel_size=kernel_size,        # 卷积核尺寸
#         #     n_groups=n_groups,              # 分组归一化的组数
#         #     cond_predict_scale=cond_predict_scale # 条件预测缩放因子
#         # )
#         # 定义循环群的大小
#          # 对称性参数配置
#         self.N = N  # 循环群的阶数（旋转离散化个数）
#         self.group = gspaces.no_base_space(CyclicGroup(self.N))  # 创建循环群（无空间基）
#         self.order = self.N  # 群阶数别名
#         # 定义等变特征场类型
#         self.act_type = nn.FieldType(
#             self.group, 
#             act_emb_dim * [self.group.regular_repr]  # 正则表示（regular representation）
#         )
#         # 输出层（等变线性变换）
#         self.out_layer = nn.Linear(self.act_type, self.getOutFieldType())
#         # 动作编码器（等变序列模块）
#         self.enc_a = nn.SequentialModule(
#             nn.Linear(self.getOutFieldType(), self.act_type),  # 等变线性层
#             nn.ReLU(self.act_type)  # 等变ReLU激活
#         )
        

#     # 定义输出场类型（混合不可约表示和平凡表示）
#     def getOutFieldType(self):
#         return nn.FieldType(
#             self.group,
#             4 * [self.group.irrep(1)] # 8  4个二维旋转不可约表示（每个占2通道，共8通道）
#             + 2 * [self.group.trivial_repr], # 2   2个平凡表示（旋转不变量，共2通道）
#         )

#     # 定义输出函数
#     def getOutput(self, conv_out):
#         """将网络原始输出解析为结构化动作参数"""
#         # 分解输出张量为具体组件（SE3版本）
#         xy = conv_out[:, 0:2]    # XY平移分量
#         cos1 = conv_out[:, 2:3]  # 第一个旋转角的余弦
#         sin1 = conv_out[:, 3:4]  # 第一个旋转角的正弦
#         cos2 = conv_out[:, 4:5]  # 第二个旋转角的余弦
#         sin2 = conv_out[:, 5:6]  # 第二个旋转角的正弦
#         cos3 = conv_out[:, 6:7]  # 第三个旋转角的余弦
#         sin3 = conv_out[:, 7:8]  # 第三个旋转角的正弦
#         z = conv_out[:, 8:9]     # Z轴平移分量
#         g = conv_out[:, 9:10]    # 夹爪状态

#         # 构造动作张量
#         action = torch.cat((xy, z, cos1, cos2, cos3, sin1, sin2, sin3, g), dim=1)
#         return action
    
#     # 定义动作几何张量
#     def getActionGeometricTensor(self, act):
#         # 获取批次大小
#         batch_size = act.shape[0] # batch_size=2048   [2048,10]
#          # 分解动作分量
#         xy = act[:, 0:2]  # XY坐标（平面平移）
#         z = act[:, 2:3]   # Z坐标（高度）
#         rot = act[:, 3:9]  # 三维旋转参数（3组cos/sin）
#         g = act[:, 9:]     # 夹爪控制信号

#         # 构造几何张量
#         cat = torch.cat(
#             (
#                 xy.reshape(batch_size, 2),
#                 rot[:, 0].reshape(batch_size, 1),
#                 rot[:, 3].reshape(batch_size, 1),
#                 rot[:, 1].reshape(batch_size, 1),
#                 rot[:, 4].reshape(batch_size, 1),
#                 rot[:, 2].reshape(batch_size, 1),
#                 rot[:, 5].reshape(batch_size, 1),
#                 z.reshape(batch_size, 1),
#                 g.reshape(batch_size, 1),
#             ),
#             dim=1,
#         )
#         return nn.GeometricTensor(cat, self.getOutFieldType())# 应用预定义的场类型
    
    
    
#     # 定义前向传播函数
#     def forward(self, 
#             z: torch.Tensor, 
#             t: Union[torch.Tensor, float, int], 
#             r: Union[torch.Tensor, float, int], 
#             global_cond=None, **kwargs):
#         """
#         x: (B,T,input_dim)
#         timestep: (B,) or int, diffusion step
#         local_cond: (B,T,local_cond_dim)
#         global_cond: (B,global_cond_dim)
#         output: (B,T,input_dim)
#         """
#         # 获取批次大小和时间步长
#         sample=z
#         B, T = sample.shape[:2]
#         # 重塑样本张量
#         sample = rearrange(sample, "b t d -> (b t) d")
#         # 获取动作几何张量
#         sample = self.getActionGeometricTensor(sample) #[2048, 10]
#         # 编码器输出 enc_a动作编码器（等变序列模块）
#         enc_a_out = self.enc_a(sample).tensor.reshape(B, T, -1)  #[b*8, 16, 512]
#         # 重塑编码器输出 准备U-Net输入（扩展组维度）
#         enc_a_out = rearrange(enc_a_out, "b t (c f) -> (b f) t c", f=self.order) #[b*8, 16, 512] self.order=8
#         '''
#         重排前：张量表示每个 batch 元素是一个时间序列，每个时间步有 c * f 个特征值，这些特征值被分组为 c 个特征组，每个组有 f 个特征。
#         重排后：张量表示每个原始 batch 元素被扩展为 f 个新的 batch 元素，每个新的 batch 元素对应一个特征组，特征组的特征值被重新组织为 c 个特征。
#         '''

#         # 处理时间步长
#         if type(t) == torch.Tensor and len(t.shape) == 1:
#             t = repeat(t, "b -> (b f)", f=self.order)
#         if type(r) == torch.Tensor and len(r.shape) == 1:
#             r = repeat(r, "b -> (b f)", f=self.order)

#         # 处理局部条件

#         # 处理全局条件
#         if global_cond is not None:
#             global_cond = rearrange(global_cond, "b (c f) -> (b f) c", f=self.order)
#         # U-Net输出
#         out = self.dit(x=enc_a_out, t=t,r=r, y=global_cond)

#         # 后处理：恢复原始维度结构
#         # 重塑U-Net输出
#         out = rearrange(out, "(b f) t c -> (b t) (c f)", f=self.order)  #128 8 16 512    
#         # 转换为几何张量
#         out = nn.GeometricTensor(out, self.act_type)
#         # 输出层
#         out = self.out_layer(out).tensor.reshape(B * T, -1)

#         # 获取最终输出
#         out = self.getOutput(out)
#         # 重塑输出
#         out = rearrange(out, "(b t) n -> b t n", b=B)
#         return out #[128, 16, 10]

class EquiMeanflowUNet(torch.nn.Module):
    def __init__(self, act_emb_dim, local_cond_dim, global_cond_dim, diffusion_step_embed_dim, down_dims, kernel_size, n_groups, cond_predict_scale, N):
        super().__init__()
        # 初始化条件U-Net模型
        self.unet = MeanflowConditionalUnet1D(
            input_dim=act_emb_dim,          # 输入维度：动作嵌入维度
            local_cond_dim=local_cond_dim,  # 局部条件维度（时序条件）
            global_cond_dim=global_cond_dim, # 全局条件维度（场景级条件）
            diffusion_step_embed_dim=diffusion_step_embed_dim, # 扩散步嵌入维度
            down_dims=down_dims,            # U-Net下采样维度列表
            kernel_size=kernel_size,        # 卷积核尺寸
            n_groups=n_groups,              # 分组归一化的组数
            cond_predict_scale=cond_predict_scale # 条件预测缩放因子
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
            z: torch.Tensor, 
            t: Union[torch.Tensor, float, int], 
            r: Union[torch.Tensor, float, int], 
            global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # 获取批次大小和时间步长
        sample  =z
        B, T = sample.shape[:2]
        # 重塑样本张量
        sample = rearrange(sample, "b t d -> (b t) d")
        # 获取动作几何张量
        sample = self.getActionGeometricTensor(sample)
        # 编码器输出 enc_a动作编码器（等变序列模块）
        enc_a_out = self.enc_a(sample).tensor.reshape(B, T, -1)  #[128, 16, 512]
        # 重塑编码器输出 准备U-Net输入（扩展组维度）
        enc_a_out = rearrange(enc_a_out, "b t (c f) -> (b f) t c", f=self.order) #[1024, 16, 64] self.order=8
        '''
        重排前：张量表示每个 batch 元素是一个时间序列，每个时间步有 c * f 个特征值，这些特征值被分组为 c 个特征组，每个组有 f 个特征。
        重排后：张量表示每个原始 batch 元素被扩展为 f 个新的 batch 元素，每个新的 batch 元素对应一个特征组，特征组的特征值被重新组织为 c 个特征。
        '''

        # 处理时间步长
        if type(t) == torch.Tensor and len(t.shape) == 1:
            t = repeat(t, "b -> (b f)", f=self.order)
        if type(r) == torch.Tensor and len(r.shape) == 1:
            r = repeat(r, "b -> (b f)", f=self.order)

        # 处理局部条件
        local_cond=None
        if local_cond is not None:
            local_cond = rearrange(local_cond, "b t (c f) -> (b f) t c", f=self.order)
        # 处理全局条件
        if global_cond is not None:
            global_cond = rearrange(global_cond, "b (c f) -> (b f) c", f=self.order)
        # U-Net输出
        out = self.unet(sample=enc_a_out, t=t,r=r, global_cond=global_cond, **kwargs)

        # 后处理：恢复原始维度结构
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
        return out #[128, 16, 10]


# 定义一个等变扩散U-Net模型
class EquiShortcutUNet(torch.nn.Module):
    def __init__(self, act_emb_dim, local_cond_dim, global_cond_dim, diffusion_step_embed_dim, down_dims, kernel_size, n_groups, cond_predict_scale, N):
        super().__init__()
        # 初始化条件U-Net模型
        self.unet = ConditionalUnet1DShortcut(
            input_dim=act_emb_dim,          # 输入维度：动作嵌入维度
            local_cond_dim=local_cond_dim,  # 局部条件维度（时序条件）
            global_cond_dim=global_cond_dim, # 全局条件维度（场景级条件）
            diffusion_step_embed_dim=diffusion_step_embed_dim, # 扩散步嵌入维度
            down_dims=down_dims,            # U-Net下采样维度列表
            kernel_size=kernel_size,        # 卷积核尺寸
            n_groups=n_groups,              # 分组归一化的组数
            cond_predict_scale=cond_predict_scale # 条件预测缩放因子
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
            t: Union[torch.Tensor, float, int], 
            dt: Union[torch.Tensor, float, int],
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
        # 编码器输出 enc_a动作编码器（等变序列模块）
        enc_a_out = self.enc_a(sample).tensor.reshape(B, T, -1)  #[128, 16, 512]
        # 重塑编码器输出 准备U-Net输入（扩展组维度）
        enc_a_out = rearrange(enc_a_out, "b t (c f) -> (b f) t c", f=self.order) #[1024, 16, 64] self.order=8
        '''
        重排前：张量表示每个 batch 元素是一个时间序列，每个时间步有 c * f 个特征值，这些特征值被分组为 c 个特征组，每个组有 f 个特征。
        重排后：张量表示每个原始 batch 元素被扩展为 f 个新的 batch 元素，每个新的 batch 元素对应一个特征组，特征组的特征值被重新组织为 c 个特征。
        '''

        # 处理时间步长
        if type(t) == torch.Tensor and len(t.shape) == 1:
            t = repeat(t, "b -> (b f)", f=self.order)
        if type(dt) == torch.Tensor and len(dt.shape) == 1:
            dt = repeat(dt, "b -> (b f)", f=self.order)
        # 处理局部条件
        if local_cond is not None:
            local_cond = rearrange(local_cond, "b t (c f) -> (b f) t c", f=self.order)
        # 处理全局条件
        if global_cond is not None:
            global_cond = rearrange(global_cond, "b (c f) -> (b f) c", f=self.order)
        # U-Net输出
        out = self.unet(enc_a_out, t,dt, local_cond, global_cond, **kwargs)

        # 后处理：恢复原始维度结构
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
        return out #[128, 16, 10]

# 假设 ConditionalUnet1D 已经导入且功能不变
# from EfficientFlow.model.diffusion.conditional_unet1d import ConditionalUnet1D

# 定义一个等变扩散U-Net模型
class EquiDiffusionSFlowUNet(torch.nn.Module):
    def __init__(self, act_emb_dim, local_cond_dim, global_cond_dim, diffusion_step_embed_dim, down_dims, kernel_size, n_groups, cond_predict_scale, N):
        super().__init__()
        
        # 定义循环群的大小
        self.N = N  # 循环群的阶数（旋转离散化个数）
        self.group = gspaces.no_base_space(CyclicGroup(self.N))  # 创建循环群（无空间基）
        self.order = self.N  # 群阶数别名

        # 定义输入/原始动作的几何场类型 (10通道)
        # 根据你的 getActionGeometricTensor 和 getOutput 函数，输入和原始输出动作的结构是 4*irrep(1) + 2*trivial_repr = 8 + 2 = 10
        self._action_field_type = nn.FieldType(
            self.group,
            4 * [self.group.irrep(1)] + 2 * [self.group.trivial_repr],
        )
        
        # 定义模型最终原始输出的几何场类型 (10通道预测 + 1通道s_flow = 11通道)
        # 我们在原始动作类型的基础上增加一个平凡表示用于预测 s_flow
        self._predicted_output_field_type = nn.FieldType(
            self.group,
            self._action_field_type.representations +(self.group.trivial_repr,), # 增加一个平凡表示
        )
        # 验证总通道数是否为 11
        assert self._predicted_output_field_type.size == 11, f"Expected output channels 11, but got {self._predicted_output_field_type.size}"

        # 定义等变特征场类型 (U-Net内部处理的特征类型)
        self.act_type = nn.FieldType(
            self.group, 
            act_emb_dim * [self.group.regular_repr]  # 正则表示
        )

        # 初始化条件U-Net模型
        # 输入维度现在是等变特征维度 (act_emb_dim * order)
        self.unet = ConditionalUnet1D(
            input_dim=act_emb_dim,          # 输入维度：动作嵌入维度
            local_cond_dim=local_cond_dim,  # 局部条件维度（时序条件）
            global_cond_dim=global_cond_dim, # 全局条件维度（场景级条件）
            diffusion_step_embed_dim=diffusion_step_embed_dim, # 扩散步嵌入维度
            down_dims=down_dims,            # U-Net下采样维度列表
            kernel_size=kernel_size,        # 卷积核尺寸
            n_groups=n_groups,              # 分组归一化的组数
            cond_predict_scale=cond_predict_scale # 条件预测缩放因子
        )
        

        # 动作编码器：将输入的原始动作几何张量映射到等变特征空间
        self.enc_a = nn.SequentialModule(
            nn.Linear(self._action_field_type, self.act_type),  # 从10通道动作类型到act_emb_dim*order特征类型
            nn.ReLU(self.act_type)  # 等变ReLU激活
        )

        # 输出层：将等变特征空间映射到包含预测动作和s_flow的几何张量
        self.out_layer = nn.Linear(self.act_type, self._predicted_output_field_type) # 从act_emb_dim*order特征类型到11通道预测类型


    # 这个函数用于将 (B*T, 10) 的原始动作张量转换为 GeometricTensor
    def getActionGeometricTensor(self, act):
        """将原始动作张量转换为GeometricTensor"""
        # 获取批次大小
        batch_size = act.shape[0] # batch_size=2048   [2048,10]
        
        # 根据你原有的 getOutput 函数的逻辑，输入张量的顺序是 xy, z, cos1, cos2, cos3, sin1, sin2, sin3, g
        # 但你的 getActionGeometricTensor 函数的cat顺序是 xy, rot[:,0], rot[:,3], rot[:,1], rot[:,4], rot[:,2], rot[:,5], z, g
        # 即 xy, cos1, sin1, cos2, sin2, cos3, sin3, z, g
        # 我们需要确保这里的顺序和 self._action_field_type 定义的表示顺序一致
        # FieldType( group, [irrep(1)]*4 + [trivial_repr]*2 ) 对应 4个二维向量 + 2个标量
        # 常见的将SE(3)动作映射到这种表示的方式是将 XY, Z 视为平凡表示，旋转分量 (cos/sin对) 视为 irrep(1)
        # 例如：[trivial_repr, trivial_repr, irrep(1), irrep(1), irrep(1), irrep(1)] 对应 Z, G, (cos1,sin1), (cos2,sin2), (cos3,sin3), XY ???
        # 你原代码的 getActionGeometricTensor 顺序是 (xy, cos1, sin1, cos2, sin2, cos3, sin3, z, g)
        # 这对应 FieldType 中的表示顺序可能是： ir_1, ir_2, ir_3, ir_4, tr_1, tr_2
        # (cos1, sin1) -> irrep(1)
        # (cos2, sin2) -> irrep(1)
        # (cos3, sin3) -> irrep(1)
        # XY -> irrep(1) (这是一个常见但需要明确的设计选择，XY平移在旋转下通常是irrep(1))
        # Z -> trivial_repr
        # G -> trivial_repr
        # 所以 FieldType 应该更像： [irrep(1)]*4 + [trivial_repr]*2
        # 对应 (cos1,sin1), (cos2,sin2), (cos3,sin3), (xy), z, g
        # 你的 getActionGeometricTensor 的 cat 顺序是 xy, cos1, sin1, cos2, sin2, cos3, sin3, z, g
        # 这与 (cos1,sin1), (cos2,sin2), (cos3,sin3), (xy), z, g 的顺序不符。
        # **请检查并确认 `getActionGeometricTensor` 函数中的 `cat` 顺序与 `_action_field_type` 中表示的顺序严格对应。**
        # **假设你原有的 getActionGeometricTensor 函数的 cat 顺序是正确的，并且与 _action_field_type 定义的 10个通道的表示顺序匹配。**
        # **这里我保留你原有的cat逻辑，并假设其产生的10通道张量对应 self._action_field_type.**

        xy = act[:, 0:2]
        z = act[:, 2:3]
        rot = act[:, 3:9]
        g = act[:, 9:]

        # **重要：请根据 self._action_field_type 的表示顺序调整这里的cat顺序！**
        # 例如，如果你的 field type 定义是 [irrep(1)]*4 + [trivial_repr]*2
        # 并且你希望它对应 (vec1_x, vec1_y), (vec2_x, vec2_y), (vec3_x, vec3_y), (vec4_x, vec4_y), scalar1, scalar2
        # 你需要确保 cat 的 10个通道按这个顺序排列。
        # 根据你的 getActionGeometricTensor 原始代码 cat 顺序： xy, cos1, sin1, cos2, sin2, cos3, sin3, z, g
        # 如果 xy 是 irrep(1), (cos1,sin1) 是 irrep(1) 等，这 4 个 irrep(1) 和 2 个 trivial 应该如何对应？
        # 看起来你的原始设计可能是：
        # irrep(1) 1: xy
        # irrep(1) 2: (cos1, sin1)
        # irrep(1) 3: (cos2, sin2)
        # irrep(1) 4: (cos3, sin3)
        # trivial_repr 1: z
        # trivial_repr 2: g
        # 如果是这样，`_action_field_type` 的定义和 `getActionGeometricTensor` 的 `cat` 顺序需要调整。
        # `_action_field_type` = nn.FieldType(self.group, [self.group.irrep(1)]*4 + [self.group.trivial_repr]*2)
        # `cat` 顺序应该是： xy, rot[:, 0:2], rot[:, 2:4], rot[:, 4:6], z, g
        # 假设你原有的 getActionGeometricTensor 实现是正确的，并且对应了某个10通道的GeometricTensor结构，
        # 我将使用你原有的 cat 逻辑，并假设它生成了对应 `_action_field_type` 的张量。
        # **再次强调：请务必验证 `getActionGeometricTensor` 的 cat 顺序与 `_action_field_type` 的表示顺序匹配。**

        cat = torch.cat(
             (
                 xy, # [B, 2]
                 rot[:, 0:1], # cos1 [B, 1]
                 rot[:, 3:4], # sin1 [B, 1]
                 rot[:, 1:2], # cos2 [B, 1]
                 rot[:, 4:5], # sin2 [B, 1]
                 rot[:, 2:3], # cos3 [B, 1]
                 rot[:, 5:6], # sin3 [B, 1]
                 z, # [B, 1]
                 g, # [B, 1]
             ),
             dim=1,
         )
        # 返回一个具有正确场类型的 GeometricTensor
        return nn.GeometricTensor(cat, self._action_field_type)
    
    # 新增函数：将模型原始输出张量分割成预测动作(噪声)和预测s_flow
    def split_predicted_output(self, raw_output_tensor):
        """
        将模型原始输出张量 (B*T, 11) 分割为预测动作/噪声 (B*T, 10) 和预测s_flow (B*T, 1)
        """
        # 前10个通道是预测的动作/噪声分量
        predicted_action_or_noise_flat = raw_output_tensor[:, :self._action_field_type.size] # self._action_field_type.size is 10
        # 最后一个通道是预测的s_flow
        predicted_sflow_flat = raw_output_tensor[:, self._action_field_type.size:] # 从第10个通道开始 (索引10) 到最后

        return predicted_action_or_noise_flat, predicted_sflow_flat

    # 定义前向传播函数
    def forward(self, 
            sample: torch.Tensor, # 可能是带噪声的动作 (B, T, 10)
            timestep: Union[torch.Tensor, float, int], # 扩散步 (B,) 或 int
            local_cond=None, # 局部条件 (B, T, local_cond_dim)
            global_cond=None, # 全局条件 (B, global_cond_dim)
            **kwargs):
        """
        sample: (B,T,10) # 这里的输入sample取决于你的扩散模型是预测噪声还是去噪后的样本
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,10) for predicted action/noise, (B,T,1) for predicted s_flow
        """
        # 获取批次大小和时间步长
        B, T = sample.shape[:2]
        
        # 1. 处理输入样本 (带噪声的动作)
        # 重塑样本张量 (B*T, 10)
        sample_flat = rearrange(sample, "b t d -> (b t) d")
        # 将样本转换为 GeometricTensor
        sample_geometric = self.getActionGeometricTensor(sample_flat)

        # 2. 通过动作编码器将其映射到等变特征空间
        enc_a_out_geometric = self.enc_a(sample_geometric)
        enc_a_out_tensor = enc_a_out_geometric.tensor # (B*T, act_emb_dim * order)

        # 3. 重塑编码器输出 准备U-Net输入 (扩展组维度)
        # (B*T, act_emb_dim * order) -> (B, T, act_emb_dim * order) -> (B*order, T, act_emb_dim)
        unet_input = rearrange(enc_a_out_tensor, "(b t) (c f) -> (b f) t c", b=B, f=self.order)

        # 4. 处理条件输入并进行重塑以匹配 U-Net 输入的批量维度 (B*order)
        if type(timestep) == torch.Tensor and len(timestep.shape) == 1:
            # 重复 timestep 以匹配 B*order 的批量维度
            timestep_unet = repeat(timestep, "b -> (b f)", f=self.order)
        else:
             # 如果 timestep 是标量或已经有 (B*order,) 的形状，则直接使用
             # 注意：如果 timestep 是标量， repeat 会出错，此处可能需要更严谨处理
             if isinstance(timestep, (int, float)):
                 # 将标量 timestep 转换为张量并重复
                 timestep_unet = torch.full((B * self.order,), float(timestep), device=sample.device)
             elif torch.is_tensor(timestep) and timestep.shape == (B * self.order,):
                 timestep_unet = timestep # 已经匹配 U-Net 批量维度
             else:
                 raise ValueError(f"Unsupported timestep shape: {timestep.shape}")


        if local_cond is not None:
            # (B, T, local_cond_dim) -> (B*order, T, local_cond_dim / order)
            # 假设 local_cond_dim 也是 act_emb_dim * order 这种结构
            assert local_cond.shape[-1] % self.order == 0, "local_cond_dim must be divisible by order"
            local_cond_unet = rearrange(local_cond, "b t (c f) -> (b f) t c", f=self.order)
        else:
            local_cond_unet = None

        if global_cond is not None:
             # (B, global_cond_dim) -> (B*order, global_cond_dim / order)
             assert global_cond.shape[-1] % self.order == 0, "global_cond_dim must be divisible by order"
             global_cond_unet = rearrange(global_cond, "b (c f) -> (b f) c", f=self.order)
        else:
            global_cond_unet = None

        # 5. 通过 U-Net 进行处理
        # U-Net 输出的形状是 (B*order, T, act_emb_dim)
        unet_out_tensor = self.unet(unet_input, timestep_unet, local_cond_unet, global_cond_unet, **kwargs)

        # 6. 后处理：恢复到 (B*T, act_emb_dim * order) 形状
        unet_out_tensor = rearrange(unet_out_tensor, "(b f) t c -> (b t) (c f)", b=B, f=self.order)
        # 将 U-Net 输出转换为 GeometricTensor
        unet_out_geometric = nn.GeometricTensor(unet_out_tensor, self.act_type)

        # 7. 通过最终输出层得到原始预测结果 (B*T, 11)
        raw_output_tensor = self.out_layer(unet_out_geometric).tensor # (B*T, 11)

        # 8. 分割原始输出为预测动作/噪声和预测 s_flow
        predicted_action_or_noise_flat, predicted_sflow_flat = self.split_predicted_output(raw_output_tensor)

        # 9. 重塑回 (B, T, N) 的形状
        predicted_action_or_noise = rearrange(predicted_action_or_noise_flat, "(b t) n -> b t n", b=B) # (B, T, 10)
        predicted_sflow = rearrange(predicted_sflow_flat, "(b t) n -> b t n", b=B) # (B, T, 1)

        # 10. 返回两个输出
        return predicted_action_or_noise, predicted_sflow # (B, T, 10), (B, T, 1)
    
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