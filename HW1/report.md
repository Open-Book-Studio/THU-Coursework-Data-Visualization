# Visualizing Air Pollution Dataset to gain insights——Data Visualization Assignment 1

叶璨铭, ycm24@mails.tsinghua.edu.cn, 

> In this assignment, you will apply the learned techniques to visualize data. Specifically, you will use the dataset provided below to plot **3** different types of charts.

## 题目背景与数据准备

根据老师的介绍，我们这次可视化的S&M-HSTPM2d5是个开源数据集，通过有效地利用这种数据集里面的信息，可以做到很多。这里有几个背景问题我想搞明白：

- PM2.5的物理意义与健康影响？为什么对环境污染来说比较重要？数据集中PM2.5含量的单位是什么？
- PM2.5数据集是怎么通过可视化来提供科学洞察的？
- zenodo是什么？
- Foshan, Cangzhou, and Tianjin在哪里？
- S&M-HSTPM2d5的设计思路是什么？

经过查询，这几个问题的答案如下

1. PM2.5
   根据AI搜索到的信息, 如[科普文章](https://mp.weixin.qq.com/s?__biz=MzI0NjM4MDQ3MQ==&mid=2247643240&idx=7&sn=37cbf5d6ff15d2c2cf7690b3a9b2587a&chksm=e87f455ded881fc4955f3721ca000f8690d90082ab53624969fca829b43b7a1c364f3836a23d#rd)，PM2.5（空气动力学直径≤2.5微米的颗粒物）是空气污染的核心指标之一。其粒径仅为人类头发丝的1/20至1/30，能够穿透人体呼吸系统屏障，直接进入肺泡甚至血液循环。其表面吸附的重金属（如铅、汞）、多环芳烃（致癌物）和病原微生物等有毒物质，会导致呼吸系统疾病（如哮喘、慢性支气管炎）、心血管疾病（如动脉硬化、心肌梗死）及免疫系统损伤。

​	所以这就是PM2.5为什么会危害健康，被认为是Air Pollusion的一部分。那么PM2.5数据集是怎么检测的呢？根据[博客文章](https://www.163.com/dy/article/JFDOF8AV055663N5.html)，具体检测的方式有光散射法、β射线法等，具体我们就不了解了。

​	参考[博客文章](https://cloud.tencent.com/developer/article/2386438?from=15425)，在科学研究和环境监测中，PM2.5的浓度统一以 **微克每立方米（µg/m³）** 为单位。

2. PM2.5可视化。PM2.5有时空分布和污染模式，所以可以在时间上揭示PM2.5浓度的日变化、季节性和年际趋势；也可以在空间上结合热力图、散点图可展示区域污染差异，高污染暴露区域可以及时指导医疗资源分配。

   AI搜索到了一个[学位论文](https://cdmd.cnki.com.cn/Article/CDMD-10710-1023420587.htm), 通过WebGIS平台发现PM2.5与交通拥堵、能源结构高度相关，这种多维度关联分析可以看到污染的成因。

3. zenodo

   Zenodo是由欧洲核子研究中心（CERN）运营的多学科开放知识库，旨在长期保存和共享科学研究成果。其核心功能包括：

   - **数据存储与DOI分配**：支持上传数据集、代码、论文等各类文件（单文件≤50GB），并为每项资源分配唯一数字对象标识符（DOI），确保可追溯和引用。
   - **协作与版本控制**：支持多版本数据管理，并与GitHub集成，实现代码与数据的同步更新。
   - **跨学科覆盖**：涵盖环境科学（如PM2.5数据集）、生物医学（如单细胞测序数据）等领域，用户可免费下载或上传数据

4. **佛山（Foshan）**位于中国广东省中南部， **沧州（Cangzhou）**位于河北省东南部，**天津（Tianjin）**地处华北平原东北部。根据[政府官网](https://www.cangzhou.gov.cn/cangzhou/c100056/202306/b92034f256c34684adcfa617223af1a0.shtml)，这次作业要看的沧州在这个位置，

   ![img](https://www.cangzhou.gov.cn/cangzhou/c100056/202306/b92034f256c34684adcfa617223af1a0/images/8d85fca319864cbd88e311c91572355f.jpg)

5. 直接打开[数据集zenodo网站](https://zenodo.org/records/4028130#.Y_tUUGRBwuU)，首先就注意到我们的授课老师陈老师是这个研究的第一作者，非常厉害。

   第一段说到采样率，沧州设置为一分钟，在佛山和天津设置为三秒。可能沧州的PM2.5比较特殊吧。可能就是因为时间间隔长，时间分辨率小，所以才是老师作业要求说的“the smallest portion of this dataset”。

   第二第三段告诉我们这个数据集是已经预处理过的，去掉了异常数据。

   第四段告诉了我们格式，每个 CSV 文件包含三种类型的数据：时间戳（中国标准时间 GMT+8）、地理坐标（纬度和经度）以及 PM2.5 浓度（单位为每立方米微克）。CSV 文件存储在 Static 或 Mobile 文件夹中，这代表了设备的类型。Static 和 Mobile 文件夹存储在相应城市的文件夹中。

## 1.1 选择我的可视化工具

> 1.1. You need to select a programming tool that helps you create the visualization application.

很多同学瞬间就反应过来，那不就是选择我们最熟悉的Python+matplotlib吗？有些同学还会纠结是不是用R语言、MATLAB来画啊。但是这些都不是重点，这次作业不一样，老师的要求里面提到代码部分我们要做的是“VisualizationApp”，既然是App，那就是具有一定的交互性的，而不只是画出来一个图片放到报告里面。所以我们需要寻找具有交互性的可视化画图库，经过和同学讨论，我了解到gradio、streamlit可以做简单交互式前端，还了解到plotly库比matplotlib更有交互性，以及jupyter notebook里面可以显示html，nbdev+quarto可以发布jupyter notebook为交互式的网站。

我们通过大模型搜索一下

<img src="./report.assets/image-20250315162758726.png" alt="image-20250315162758726" style="zoom:50%;" />

<img src="./report.assets/image-20250315162818359.png" alt="image-20250315162818359" style="zoom: 50%;" />

大模型也推荐了plotly，和我自己之前了解到的一致，看起来是个比较热门的有人维护的工具。

我们还可以看下huggingface space推荐的可视化库，如果用这些库，我们作业做出来可以部署上去。

![image-20250315163140846](./report.assets/image-20250315163140846.png)

我喜欢Python语言，所以首先排除R、MATLAB、js等。而huggingface space上没有plotly或者dash，结合LLM的选择，加之streamlit可以引入plotly作为其绘图能力的一部分，我决定使用streamlit。

<img src="./report.assets/image-20250315164616766.png" alt="image-20250315164616766" style="zoom:50%;" />

现在我们可以初始化一下我们的git版本控制结构


```bash
cd HW1
git submodule add https://huggingface.co/spaces/2catycm/pm2.5-visualization VisualizationApp
```

![alt text](./report.assets/image.png)

并且安装库

```bash
pip install streamlit plotly
```

## 1.2 根据数据绘制三种不同类型的图表

> You need to create 3 different types of graphs based on the given dataset

### (a) A Line Chart that shows how the PM2.5 level changes over time

> i. Use the ’.csv’ files in the static folder (in the VisualizationApp/data/Cangzhou) and filter the data by time stamps
> between 2019-01-01 00:00:00 and 2019-01-01 12:00:00.
> ii. The x-axis and the y-axis represent time and the value of PM2.5, respecetively.
> iii. Plot one line for each static sensor and distinguish the sensors with colors.

#### 代码编写

首先我们自己需要检查数据集的格式，

```csv
,pm2d5,lat,lon
2018-12-27 00:00:00,60.206,38.322876,116.856699
2018-12-27 00:01:00,63.90164,38.322876,116.856699
2018-12-27 00:02:00,64.392,38.322876,116.856699
```

每一个文件是一个设备，每一个设备的csv格式是时间、PM2.5浓度、纬度、经度。由于是静态的，最后两列在同一个文件中是固定的，这个是我们对数据的理解。

我们的git项目结构是submodule嵌套的，为了让python解释器无论在什么地方运行都不出错，我们直接通过`__file__`去计算相对于`app.py`自己的路径，这个AI写代码的总是忽略这个重要问题。
```python
from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
data_cangzhou_folder = this_directory / "data/Cangzhou"
```

现在我们可以告诉LLM我们的数据集格式是什么样子，然后询问LLM，根据streamlit的文档，怎么实现line_chart的需求。有些LLM不懂streamlit，也无法访问其文档，有些AI会生成错误的plotly代码。
这需要我们自己去做决策，不是问一个问题LLM就能画好的。

首先是对数据的理解，三道题目的数据其实同样的格式，经过对AI的提示，成功写出带有参数的过滤代码。
```python
# 加载和过滤数据的函数，接受时间区间参数
def load_and_filter_data(file_path, start_time, end_time):
    # 根据提供的CSV格式，第一列是时间戳但没有列名
    df = pd.read_csv(file_path, header=0, names=['timestamp', 'pm2d5', 'lat', 'lon'])
    
    # 确保时间戳列是datetime格式
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 过滤指定时间段内的数据
    filtered_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
    
    return filtered_df
```

然后是对这道题绘制折线图的理解。

```python
# 第一列：PM2.5随时间变化的折线图
with col1:
    st.header("PM2.5随时间变化")
    
    # 定义过滤时间段
    start_dt = datetime(2019, 1, 1, 0, 0, 0)
    end_dt = datetime(2019, 1, 1, 12, 0, 0)
    
    # 获取数据文件夹中的所有CSV文件
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    # 创建字典存储每个传感器的数据
    sensor_data = {}
    
    # 加载并过滤每个传感器的数据
    for file in csv_files:
        file_path = os.path.join(data_folder, file)
        sensor_name = file.split('.')[0]  # 从文件名提取传感器名称
        sensor_data[sensor_name] = load_and_filter_data(file_path, start_dt, end_dt)
    
    # 准备可视化数据
    # 创建一个以时间戳为索引，每个传感器的PM2.5值为列的数据框
    chart_data = pd.DataFrame()
    
    for sensor, data in sensor_data.items():
        # 使用pm2d5列作为PM2.5数据
        chart_data[sensor] = data.set_index('timestamp')['pm2d5']
    
    # 绘制折线图
    st.line_chart(
        chart_data,
        x=None,  # 使用索引(timestamp)作为x轴
        y=list(sensor_data.keys()),  # 使用所有传感器名称作为y列
        x_label="时间",
        y_label="PM2.5水平"
    )
```

其中streamlit自动会分配为不同dict key分配对应的颜色。

由于streamlit允许使用plotly来画图，把画图结果嵌入到streamlit的一部分，所以我们也可以不用`st.line_chart`，而是这样画

```python
fig_line = px.line(chart_data, title="PM2.5随时间变化折线图")
fig_line.update_layout(xaxis_title="时间", yaxis_title="PM2.5水平")
st.plotly_chart(fig_line, use_container_width=True)
```

plotly画出来的更好看。

此外，我们可以用streamlit在网页上展示绘图要求、图片描述等。

用到的语法是

```python
st.header("（a）折线图：展示 PM2.5 浓度水平随时间的变化")
    with st.expander("绘制要求"):
        st.markdown("1. 使用 static 文件夹中的 .csv 文件，并筛选出时间戳在 2019-01-01 00:00:00 到 2019-01-01 12:00:00 之间的数据。")
        st.markdown("2. X 轴和 Y 轴分别表示时间和 PM2.5 浓度水平。")
        st.markdown("3. 为每个静态传感器绘制一条折线，并用不同颜色进行区分。")
```

#### 绘制效果与分析

##### 绘制的图片

运行 

```bash
streamlit run HW1/VisualizationApp/app.py
```

![image-20250315175500700](./report.assets/image-20250315175500700.png)

可以看到我们鼠标悬停的时候，可以看到交互式的信息，图片本身可以放大缩小看到细节。

<img src="./report.assets/image-20250315183024178.png" alt="image-20250315183024178" style="zoom:50%;" />



我们将`st.line_chart`画的图片导出为svg格式，然后放到本文档中

<img src="./report.assets/visualization.svg" alt="visualization" style="zoom:33%;" />

而plotly的只能导出为png格式。

![newplot](./report.assets/newplot.png)

##### 描述图片以及从图片中看到的findings

我首先就注意到8:00的时候浅蓝色的D02突然变得很高，随后绿色的线D05似乎是跟着D02的趋势，在D02下来的时候D05就上升了，也到达了一个峰值。

因为是交互式图表，我们可以随时鼠标悬停查看感兴趣的位置，

<img src="./report.assets/image-20250315183726329.png" alt="image-20250315183726329" style="zoom:33%;" />

紧接着在10:46左右，深蓝色的D1突然升高，随后其他的线的趋势非常类似，也一起升高，最后12:00左右又降到比较低的水平。

![image-20250315183745729](./report.assets/image-20250315183745729.png)



现在我们让多模态大模型尝试解读一下图片。

##### 对findings的解释

### (b) A Scatter Plot that shows how vehicles carrying mobile sensors move in the city.

#### 代码编写

#### 绘制效果与分析

### (c) A 3D Histogram that shows the distribution of PM2.5 levels in the whole city at a specific time range.

#### 代码编写

在写这一题的时候，我们发现整个streamlit应用是可以自己检测代码更新然后rerun的，不过每次rerun需要重新读取一遍csv数据，这个是比较慢的，

所以我们查阅了文档，发现了`st.cache_data`装饰器可以解决问题。



#### 绘制效果与分析



## 1.3-1.4 报告要求检查与LLM使用情况声明

本次作业中，我们交叉使用了腾讯元宝、秘塔AI搜索、Github Copilot、Roo Code、Cline、streamlit doc AI等大模型工具。
