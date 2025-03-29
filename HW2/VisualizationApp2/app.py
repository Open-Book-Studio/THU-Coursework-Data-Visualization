import streamlit as st
import numpy as np
from pathlib import Path
from experiments.gmm_dataset import GeneralizedGaussianMixture
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple

def init_session_state():
    """初始化session state"""
    if 'prev_K' not in st.session_state:
        st.session_state.prev_K = 3
    if 'p' not in st.session_state:
        st.session_state.p = 2.0
    if 'centers' not in st.session_state:
        st.session_state.centers = np.array([[-2, -2], [0, 0], [2, 2]], dtype=np.float64)
    if 'scales' not in st.session_state:
        st.session_state.scales = np.array([[0.3, 0.3], [0.2, 0.2], [0.4, 0.4]], dtype=np.float64)
    if 'weights' not in st.session_state:
        st.session_state.weights = np.ones(3, dtype=np.float64) / 3
    if 'sample_points' not in st.session_state:
        st.session_state.sample_points = None

def create_default_parameters(K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """创建默认参数"""
    # 在[-3, 3]范围内均匀生成K个中心点
    x = np.linspace(-3, 3, K)
    y = np.linspace(-3, 3, K)
    centers = np.column_stack((x, y))
    
    # 默认尺度和权重
    scales = np.ones((K, 2), dtype=np.float64) * 3
    weights = np.random.random(size=K).astype(np.float64)
    weights /= weights.sum()  # 归一化权重
    return centers, scales, weights

def generate_latex_formula(p: float, K: int, centers: np.ndarray, 
                         scales: np.ndarray, weights: np.ndarray) -> str:
    """生成LaTeX公式"""
    formula = r"P(x) = \sum_{k=1}^{" + str(K) + r"} \pi_k P_{\theta_k}(x) \\"
    formula += r"P_{\theta_k}(x) = \eta_k \exp(-s_k d_k(x)) = \frac{p}{2\alpha_k \Gamma(1/p) }\exp(-\frac{|x-c_k|^p}{\alpha_k^p})= \frac{p}{2\alpha_k \Gamma(1/p) }\exp(-|\frac{x-c_k}{\alpha_k}|^p) \\"
    formula += r"\text{where: }"
    
    for k in range(K):
        c = centers[k]
        s = scales[k]
        w = weights[k]
        component = f"P_{k+1}(x) = \\frac{{{p:.1f}}}{{2\\alpha_{k+1} \\Gamma(1/{p:.1f})}}\\exp(-|\\frac{{x-({c[0]:.1f}, {c[1]:.1f})}}{{{s[0]:.1f}, {s[1]:.1f}}}|^{{{p:.1f}}}) \\\\"
        formula += component
        formula += f"\\pi_{k+1} = {w:.2f} \\\\"
    
    return formula

st.set_page_config(page_title="GMM Distribution Visualization", layout="wide")
st.title("广义高斯混合分布可视化")

# 初始化session state
init_session_state()

# 侧边栏参数设置
with st.sidebar:
    st.header("分布参数")
    
    # 分布基本参数
    st.session_state.p = st.slider("形状参数 (p)", 0.1, 5.0, st.session_state.p, 0.1,
                                 help="p=1: 拉普拉斯分布, p=2: 高斯分布, p→∞: 均匀分布")
    K = st.slider("分量数 (K)", 1, 5, st.session_state.prev_K)
    
    # 如果K发生变化，重新初始化参数
    if K != st.session_state.prev_K:
        centers, scales, weights = create_default_parameters(K)
        st.session_state.centers = centers
        st.session_state.scales = scales
        st.session_state.weights = weights
        st.session_state.prev_K = K
    
    # 高级参数设置
    st.subheader("高级设置")
    show_advanced = st.checkbox("显示分量参数", value=False)
    
    if show_advanced:
        # 为每个分量设置参数
        centers_list: List[List[float]] = []
        scales_list: List[List[float]] = []
        weights_list: List[float] = []
        
        for k in range(K):
            st.write(f"分量 {k+1}")
            col1, col2 = st.columns(2)
            with col1:
                cx = st.number_input(f"中心X_{k+1}", -5.0, 5.0, float(st.session_state.centers[k][0]), 0.1)
                cy = st.number_input(f"中心Y_{k+1}", -5.0, 5.0, float(st.session_state.centers[k][1]), 0.1)
            with col2:
                sx = st.number_input(f"尺度X_{k+1}", 0.1, 3.0, float(st.session_state.scales[k][0]), 0.1)
                sy = st.number_input(f"尺度Y_{k+1}", 0.1, 3.0, float(st.session_state.scales[k][1]), 0.1)
            w = st.slider(f"权重_{k+1}", 0.0, 1.0, float(st.session_state.weights[k]), 0.1)
            
            centers_list.append([cx, cy])
            scales_list.append([sx, sy])
            weights_list.append(w)
        
        centers = np.array(centers_list, dtype=np.float64)
        scales = np.array(scales_list, dtype=np.float64)
        weights = np.array(weights_list, dtype=np.float64)
        weights = weights / weights.sum()
        
        st.session_state.centers = centers
        st.session_state.scales = scales
        st.session_state.weights = weights
    else:
        centers = st.session_state.centers
        scales = st.session_state.scales
        weights = st.session_state.weights

    # 采样设置
    st.subheader("采样设置")
    n_samples = st.slider("采样点数", 5, 20, 10)
    if st.button("重新采样"):
        # 生成随机样本
        samples = []
        for _ in range(n_samples):
            # 选择分量
            k = np.random.choice(K, p=weights)
            # 从选定的分量生成样本
            sample = np.random.normal(centers[k], scales[k], size=2)
            samples.append(sample)
        st.session_state.sample_points = np.array(samples)

# 创建GMM数据集
dataset = GeneralizedGaussianMixture(
    D=2,
    K=K,
    p=st.session_state.p,
    centers=centers[:K],
    scales=scales[:K],
    weights=weights[:K]
)

# 生成网格数据
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
xy = np.column_stack((X.ravel(), Y.ravel()))

# 计算概率密度
Z = dataset.pdf(xy).reshape(X.shape)

# 创建2D和3D可视化
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'contour'}]],
    subplot_titles=('3D概率密度曲面', '等高线图与分量中心')
)

# 3D Surface
surface = go.Surface(
    x=X, y=Y, z=Z,
    colorscale='viridis',
    showscale=True,
    colorbar=dict(x=0.45)
)
fig.add_trace(surface, row=1, col=1)

# Contour Plot with component centers
contour = go.Contour(
    x=x, y=y, z=Z,
    colorscale='viridis',
    showscale=True,
    colorbar=dict(x=1.0),
    contours=dict(
        showlabels=True,
        labelfont=dict(size=12)
    )
)
fig.add_trace(contour, row=1, col=2)

# 添加分量中心点
fig.add_trace(
    go.Scatter(
        x=centers[:K, 0], y=centers[:K, 1],
        mode='markers+text',
        marker=dict(size=10, color='red'),
        text=[f'C{i+1}' for i in range(K)],
        textposition="top center",
        name='分量中心'
    ),
    row=1, col=2
)

# 添加采样点（如果有）
if st.session_state.sample_points is not None:
    samples = st.session_state.sample_points
    # 计算每个样本点的概率密度
    probs = dataset.pdf(samples)
    # 计算每个样本点属于每个分量的后验概率
    posteriors = []
    for sample in samples:
        component_probs = [
            weights[k] * np.exp(-np.sum(((sample - centers[k]) / scales[k])**st.session_state.p)) 
            for k in range(K)
        ]
        total = sum(component_probs)
        posteriors.append([p/total for p in component_probs])
    
    # 添加样本点到图表
    fig.add_trace(
        go.Scatter(
            x=samples[:, 0], y=samples[:, 1],
            mode='markers+text',
            marker=dict(
                size=8,
                color='yellow',
                line=dict(color='black', width=1)
            ),
            text=[f'S{i+1}' for i in range(len(samples))],
            textposition="bottom center",
            name='采样点'
        ),
        row=1, col=2
    )
    
    # 显示样本点的概率信息
    st.subheader("采样点信息")
    for i, (sample, prob, post) in enumerate(zip(samples, probs, posteriors)):
        st.write(f"样本点 S{i+1} ({sample[0]:.2f}, {sample[1]:.2f}):")
        st.write(f"- 概率密度: {prob:.4f}")
        st.write("- 后验概率:")
        for k in range(K):
            st.write(f"  - 分量 {k+1}: {post[k]:.4f}")
        st.write("---")

# 更新布局
fig.update_layout(
    title='广义高斯混合分布',
    showlegend=True,
    width=1200,
    height=600,
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='密度'
    )
)

# 更新2D图的坐标轴
fig.update_xaxes(title_text='X', row=1, col=2)
fig.update_yaxes(title_text='Y', row=1, col=2)

# 显示图形
st.plotly_chart(fig, use_container_width=True)

# 添加参数说明
with st.expander("分布参数说明"):
    st.markdown("""
    - **形状参数 (p)**：控制分布的形状
        - p = 1: 拉普拉斯分布
        - p = 2: 高斯分布
        - p → ∞: 均匀分布
    - **分量参数**：每个分量由以下参数确定
        - 中心 (μ): 峰值位置，通过X和Y坐标确定
        - 尺度 (α): 分布的展宽程度，X和Y方向可不同
        - 权重 (π): 混合系数，所有分量权重和为1
    """)

# 显示当前参数的数学公式
st.latex(generate_latex_formula(st.session_state.p, K, centers[:K], scales[:K], weights[:K]))