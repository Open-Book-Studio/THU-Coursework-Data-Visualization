import plotly.graph_objects as go

# 示例数据（请替换为实际数据）
methods = ['RSRM', 'PSRN', 'NGGP', 'PySR', 'BMS', 'uDSR', 'AIF', 
          'DGSR', 'E2E', 'SymINDy', 'PhySO', 'TPSR', 'SPL', 
          'DEAP', 'SINDy', 'NSRS', 'gplearn', 'SNIP', 'KAN', 'EQL']
recovery_rates = [85, 78, 92, 88, 76, 83, 95, 81, 89, 77, 84, 86, 80, 
                79, 82, 87, 75, 88, 90, 84]  # 恢复率百分比
errors = [3, 4, 2, 3, 5, 2, 1, 3, 2, 4, 3, 2, 3, 4, 2, 3, 5, 2, 3, 2]  # 误差范围

# 创建图形对象
fig = go.Figure()

# 添加带误差线的数据点
fig.add_trace(go.Scatter(
    x=recovery_rates,
    y=methods,
    mode='markers',
    error_x=dict(
        type='data',
        array=errors,
        visible=True,
        color='#FF5733',
        thickness=2,
        width=10
    ),
    marker=dict(
        size=12,
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
              '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'],
        opacity=0.8
    )
))

# 设置布局
fig.update_layout(
    title='不同方法的恢复率比较',
    xaxis=dict(
        title='恢复率 (%)',
        range=[0, 100],
        dtick=20,
        title_standoff=25
    ),
    yaxis=dict(
        title='Methods',
        title_font=dict(size=14),
        tickfont=dict(size=12),
        autorange="reversed"  # 使第一个方法显示在最上方
    ),
    hovermode='closest',
    width=1000,
    height=600,
    showlegend=False
)

# 添加注释（可选）
fig.add_annotation(
    x=0,
    y=0.95,
    xref='paper',
    yref='paper',
    text='知乎 @x66ccff',
    showarrow=False,
    font=dict(size=10)
)

fig.show()