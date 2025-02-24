# 智能问卷对话系统

基于 LLM 和概率编程的智能问卷调查系统，支持自动因果推理和动态追问。

## 项目结构

```
causal_survey/
├── dialogue/        # 对话管理模块
├── inference/       # 因果推理模块
├── visualization/   # 可视化模块
├── logs/           # 日志存储
├── requirements.txt # 项目依赖
└── README.md       # 项目说明
```

## 主要功能

1. 交互式问答系统
   - 用户文本对话
   - LLM 自动分析回答
   - 动态追问机制

2. 因果推理
   - 基于 Pyro 的贝叶斯更新
   - 动态因果图构建

3. 可视化
   - NetworkX 静态图
   - D3.js 交互式图表

## 安装和使用

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置环境变量：
创建 .env 文件并设置：
```
ANTHROPIC_API_KEY=your_api_key
```

3. 运行系统：
```bash
python main.py
``` 