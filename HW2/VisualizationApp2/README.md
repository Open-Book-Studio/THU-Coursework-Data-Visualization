---
title: Generalized Gaussian Mixture Visualization
emoji: 🔄
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: 'Interactive visualization of Generalized Gaussian Mixture Distribution.'
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 广义高斯混合分布可视化

## 可视化思路

1. 页面布局：
```plaintext
+-----------------+----------------------+
|   参数侧边栏    |       主显示区域      |
|  - 形状参数p    |  +--------+--------+ |
|  - 分量数K      |  |        |        | |
|  - 分量参数     |  |   3D   |  等高线 | |
|                 |  | Surface | Plot  | |
+-----------------+  |        |        | |
                    +--------+--------+ |
                    |    参数说明       |
                    +----------------+ |
```

2. 图表配置：
- 左图：3D曲面图 (Surface Plot)
  - X轴：第一维坐标
  - Y轴：第二维坐标
  - Z轴：概率密度值
  - 使用viridis配色方案

- 右图：等高线图 (Contour Plot)
  - X轴：第一维坐标
  - Y轴：第二维坐标
  - 颜色：概率密度值
  - 标记分量中心点

3. Plotly配置要点：
```python
# 子图布局
specs=[[{'type': 'surface'}, {'type': 'contour'}]]

# 坐标轴配置
scene=dict(  # 3D图的坐标轴
    xaxis_title='X',
    yaxis_title='Y',
    zaxis_title='Density'
)
xaxis=dict(title='X'),  # 2D图X轴
yaxis=dict(title='Y')   # 2D图Y轴
```

## 数据处理流程

1. 参数处理
- 基本参数：p(形状), K(分量数)
- 每个分量：中心点、尺度、权重
- 参数改变时实时更新

2. 数据生成
- 使用meshgrid生成网格点
- 计算每个点的概率密度
- 重塑数据以适配plotly格式

3. 交互更新
- 参数变化触发重新计算
- 动态更新图表和说明
