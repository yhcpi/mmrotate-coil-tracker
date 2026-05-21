# ============================================================================
# 基础配置部分
# ============================================================================
# 继承默认运行时配置（包含日志、检查点保存等基础设置）
_base_ = ['../_base_/default_runtime.py']

# ============================================================================
# 日志配置
# ============================================================================
log_config = dict(
    interval=5,  # 每5个迭代（iteration）记录一次日志信息
    hooks=[
        dict(type='TextLoggerHook'),  # 文本日志钩子：将日志输出到控制台和日志文件
        dict(type='TensorboardLoggerHook')  # TensorBoard日志钩子：记录训练指标用于可视化
    ])

# ============================================================================
# 角度编码版本配置
# ============================================================================
# 'le90'表示角度范围为[-90°, 0°)，即left-edge 90度表示法
# 这种表示法将角度限制在90度范围内，避免角度歧义
angle_version = 'le90'

# ============================================================================
# 模型核心配置
# ============================================================================
model = dict(
    type='RotatedFCOS',  # 模型类型：旋转版FCOS（Fully Convolutional One-Stage Detector）

    # ----------------------------------------------------------------------------
    # 主干网络（Backbone）：特征提取器
    # ----------------------------------------------------------------------------
    backbone=dict(
        type='ResNet',  # 使用ResNet残差网络作为主干
        depth=18,  # ResNet深度：18层（轻量级，计算速度快）
        num_stages=4,  # 特征提取阶段数：4个阶段（对应conv2_x, conv3_x, conv4_x, conv5_x）
        out_indices=(0, 1, 2, 3),  # 输出哪些阶段的特征图：全部4个阶段都输出
        frozen_stages=0,  # 冻结的阶段数：0表示所有阶段都参与训练（不冻结）
        norm_cfg=dict(type='BN', requires_grad=True),  # 归一化层配置：批归一化（BatchNorm），梯度可更新
        norm_eval=True,  # 评估模式时使用BN的统计量（均值和方差）
        style='pytorch',  # ResNet风格：PyTorch实现（与Caffe实现略有不同）
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),  # 初始化配置：使用torchvision预训练的ResNet18权重

    # ----------------------------------------------------------------------------
    # 颈部网络（Neck）：特征金字塔
    # ----------------------------------------------------------------------------
    neck=dict(
        type='FPN',  # 特征金字塔网络（Feature Pyramid Network）：融合多尺度特征
        in_channels=[64, 128, 256, 512],  # 输入通道数：对应ResNet 4个阶段的输出通道数
        out_channels=128,  # 输出通道数：将所有特征图统一转换为128通道
        start_level=1,  # 起始层级：从第1个阶段开始构建FPN（跳过第0阶段）
        add_extra_convs='on_output',  # 额外卷积层添加方式：在FPN输出上添加额外卷积层
        num_outs=5,  # 输出特征图数量：生成5个不同尺度的特征图（P2-P6）
        relu_before_extra_convs=True),  # 在额外卷积层前是否使用ReLU激活函数

    # ----------------------------------------------------------------------------
    # 检测头（Head）：分类和回归预测
    # ----------------------------------------------------------------------------
    bbox_head=dict(
        type='RotatedFCOSHead',  # 旋转版FCOS检测头
        num_classes=1,  # 检测类别数：1类（铁路缺陷检测）
        in_channels=128,  # 输入通道数：与FPN输出通道数一致
        stacked_convs=2,  # 堆叠的卷积层数量：2层卷积用于特征提取
        feat_channels=128,  # 特征通道数：卷积层的通道数为128
        strides=[8, 16, 32, 64, 128],  # 步长列表：5个特征层级相对于输入图像的步长

        # 中心采样配置：改善正样本选择
        center_sampling=True,  # 启用中心采样：只在中心区域选择正样本
        center_sample_radius=1.5,  # 中心采样半径：1.5倍步长

        norm_on_bbox=False,  # 是否对边界框预测进行归一化：False
        centerness_on_reg=True,  # 是否在回归分支上预测centerness：True（推荐做法）
        separate_angle=False,  # 是否分离角度预测：False（角度与其他回归参数一起预测）
        scale_angle=True,  # 是否对角度的scale进行缩放：True

        # 边界框编码器配置
        bbox_coder=dict(
            type='DistanceAnglePointCoder',  # 距离-角度点编码器：基于点到边界的距离和角度编码
            angle_version=angle_version),  # 角度版本：使用前面定义的le90

        # 分类损失配置
        loss_cls=dict(
            type='FocalLoss',  # Focal Loss：解决正负样本不平衡问题
            use_sigmoid=True,  # 使用Sigmoid激活函数（多标签分类）
            gamma=2.0,  # Focal Loss的gamma参数：降低简单样本的权重
            alpha=0.25,  # Focal Loss的alpha参数：平衡正负样本比例
            loss_weight=1.0),  # 分类损失的权重系数

        # 边界框回归损失配置
        loss_bbox=dict(
            type='GDLoss_v1',  # 广义分布损失v1版本（Generalized Distribution Loss）
            loss_type='kld',  # 损失类型：KL散度（Kullback-Leibler Divergence）
            fun='log1p',  # 变换函数：log(1+x)，使损失更平滑
            tau=1,  # 温度参数：控制分布的平滑程度
            loss_weight=1.0),  # 回归损失的权重系数

        # Centerness损失配置（用于抑制低质量检测框）
        loss_centerness=dict(
            type='CrossEntropyLoss',  # 交叉熵损失
            use_sigmoid=True,  # 使用Sigmoid激活
            loss_weight=1.0)),  # Centerness损失的权重系数

    # 训练配置：None表示使用默认配置
    train_cfg=None,

    # 测试配置：推理时的参数
    test_cfg=dict(
        nms_pre=200,  # NMS（非极大值抑制）前保留的最高分检测框数量：200
        min_bbox_size=0,  # 最小边界框尺寸：0（不过滤小目标）
        score_thr=0.05,  # 置信度阈值：只保留置信度>0.05的检测框
        nms=dict(iou_thr=0.1),  # NMS的IoU阈值：0.1（较严格，去除重叠框）
        max_per_img=50))  # 每张图像最多保留的检测框数量：50

# ============================================================================
# 数据路径配置
# ============================================================================
data_root = 'data/'  # 数据集根目录

# ============================================================================
# 图像归一化配置
# ============================================================================
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # ImageNet数据集的RGB通道均值
    std=[58.395, 57.12, 57.375],  # ImageNet数据集的RGB通道标准差
    to_rgb=True)  # 是否转换为RGB格式：是

# ============================================================================
# 训练数据预处理流程（Data Pipeline）
# ============================================================================
train_pipeline = [
    dict(type='LoadImageFromFile'),  # 步骤1：从文件加载图像
    dict(type='LoadAnnotations', with_bbox=True),  # 步骤2：加载标注信息（包括边界框）

    # 步骤3：多边形随机旋转数据增强
    dict(type='PolyRandomRotate',
         rotate_ratio=0.5,  # 旋转概率：50%的图像会被旋转
         mode='range',  # 旋转模式：在指定角度范围内随机旋转
         angles_range=10,  # 角度范围：±10度
         auto_bound=True),  # 自动调整图像边界以适应旋转

    dict(type='RResize', img_scale=(1600, 900)),  # 步骤4：调整图像尺寸到1600x900
    dict(type='RRandomFlip', flip_ratio=0.5),  # 步骤5：随机翻转，概率50%
    dict(type='Normalize', **img_norm_cfg),  # 步骤6：图像归一化（减均值除标准差）
    dict(type='Pad', size_divisor=32),  # 步骤7：填充图像，确保尺寸能被32整除（FPN要求）
    dict(type='DefaultFormatBundle'),  # 步骤8：将数据打包成默认格式
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])  # 步骤9：收集需要的数据字段
]

# ============================================================================
# 测试数据预处理流程
# ============================================================================
test_pipeline = [
    dict(type='LoadImageFromFile'),  # 步骤1：从文件加载图像
    dict(
        type='MultiScaleFlipAug',  # 多尺度翻转增强（测试时增强策略）
        img_scale=(1600, 900),  # 测试图像尺寸：1600x900
        flip=False,  # 是否启用翻转：否
        transforms=[
            dict(type='RResize'),  # 子步骤1：调整图像尺寸
            dict(type='Normalize', **img_norm_cfg),  # 子步骤2：图像归一化
            dict(type='Pad', size_divisor=32),  # 子步骤3：填充图像
            dict(type='DefaultFormatBundle'),  # 子步骤4：数据打包
            dict(type='Collect', keys=['img'])  # 子步骤5：只收集图像数据（测试时不需要标注）
        ])
]

# ============================================================================
# 数据加载器配置
# ============================================================================
data = dict(
    samples_per_gpu=8,  # 每个GPU的批次大小（batch size）：8张图像
    workers_per_gpu=2,  # 每个GPU的数据加载工作进程数：2个进程并行加载数据

    # 训练集配置
    train=dict(
        type='RailDataset',  # 数据集类型：自定义的铁路数据集
        ann_file=data_root + 'train/labels/',  # 训练集标注文件路径
        img_prefix=data_root + 'train/images/',  # 训练集图像文件路径前缀
        pipeline=train_pipeline,  # 使用的预处理流程：训练流程
        version=angle_version),  # 角度编码版本：le90

    # 验证集配置
    val=dict(
        type='RailDataset',  # 数据集类型：铁路数据集
        ann_file=data_root + 'val/labels/',  # 验证集标注文件路径
        img_prefix=data_root + 'val/images/',  # 验证集图像文件路径前缀
        pipeline=test_pipeline,  # 使用的预处理流程：测试流程（验证时不做增强）
        version=angle_version),  # 角度编码版本：le90

    # 测试集配置
    test=dict(
        type='RailDataset',  # 数据集类型：铁路数据集
        ann_file=data_root + 'test/labels/',  # 测试集标注文件路径
        img_prefix=data_root + 'test/images/',  # 测试集图像文件路径前缀
        pipeline=test_pipeline,  # 使用的预处理流程：测试流程
        version=angle_version))  # 角度编码版本：le90

# ============================================================================
# 优化器配置
# ============================================================================
optimizer = dict(
    type='SGD',  # 优化器类型：随机梯度下降（Stochastic Gradient Descent）
    lr=0.001,  # 初始学习率：0.001
    momentum=0.9,  # 动量参数：0.9（加速收敛并减少震荡）
    weight_decay=0.0001)  # 权重衰减系数：0.0001（L2正则化，防止过拟合）

# ============================================================================
# 优化器钩子配置
# ============================================================================
optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2))  # 梯度裁剪：防止梯度爆炸，最大L2范数为35

# ============================================================================
# 学习率调度配置
# ============================================================================
lr_config = dict(
    policy='CosineAnnealing',  # 学习率策略：余弦退火（Cosine Annealing）
    warmup='linear',  # 预热策略：线性增长
    warmup_iters=20,  # 预热迭代次数：前20次迭代进行预热
    warmup_ratio=1.0 / 3,  # 预热起始学习率比例：从1/3的基础学习率开始
    min_lr=1e-6)  # 最小学习率：余弦退火的最低学习率为1e-6

# ============================================================================
# 运行器配置
# ============================================================================
runner = dict(
    type='EpochBasedRunner',  # 运行器类型：基于epoch的运行器
    max_epochs=200)  # 最大训练轮数：200个epoch

# ============================================================================
# 检查点配置
# ============================================================================
checkpoint_config = dict(
    interval=1)  # 检查点保存间隔：每1个epoch保存一次模型

# ============================================================================
# 评估配置
# ============================================================================
evaluation = dict(
    interval=1,  # 评估间隔：每1个epoch进行一次评估
    metric='mAP')  # 评估指标：mAP（mean Average Precision，平均精度均值）
