# LangChain 1.0 动态模型控制机械臂系统

基于 LangChain v1.0 构建的智能机械臂控制系统，实现了聊天问答与自动机械臂控制的多功能交互。系统通过中间件（middleware）的 `wrap_model_call` 机制实现动态模型选择，根据用户意图自动切换不同能力的模型，显著提升系统响应速度。

## ✨ 功能特性

- 🤖 **智能意图识别**：基于微调的 BERT 模型，准确判断用户输入为"聊天问答"或"机械臂控制命令"
- 🔄 **动态模型路由**：根据意图分类结果，自动选择最适合的模型，通过动态模型选择，显著降低系统响应时间
  - **聊天模式**：使用 `qwen-flash` 快速响应日常问答
  - **控制模式**：使用 `qwen3-max` 处理复杂的机械臂控制任务
- 🦾 **机械臂控制**：支持多种预定义动作和组合流程
- 🔧 **模拟模式**：在没有物理机械臂的环境下，可正常运行文本分类与模型选择功能

## 🚀 快速开始

### 1. 下载模型权重

从 Hugging Face 下载文本分类模型的 ONNX 权重：

```bash
# 访问以下链接下载 bert_classifier.onnx
https://huggingface.co/IsaraYu/Chat-Command_TextClassification
```

将下载的 `bert_classifier.onnx` 文件放置在项目根目录。

### 2. 安装依赖

使用 `uv`（推荐）：

```bash
uv sync
```

或使用 `pip`：

```bash
pip install -e .
```

### 3. 配置 API Key

1. 访问 [阿里云百炼控制台](https://bailian.console.aliyun.com/?tab=model#/api-key) 获取 API Key
2. 在项目根目录创建 `.env` 文件
3. 添加以下配置：

```env
DASHSCOPE_API_KEY=your_api_key_here
```

### 4. 运行程序

```bash
python langchain_onnx_qwen.py
```

### 工作流程

```
用户输入 
  ↓
BERT 文本分类模型（判断 chat/command）
  ↓
动态路由中间件
  ├─ chat → qwen-flash（快速响应）
  └─ command → qwen3-max（精确控制）
  ↓
Agent 处理
  ↓
机械臂执行（如为 command 且识别成功）
```

## 📋 系统要求

- Python >= 3.13
- 支持 ONNX Runtime 的环境
- （可选）亚博机械臂硬件及 Arm_Lib 库


## 📖 使用说明

### 基本命令

- 输入自然语言指令控制机械臂
- 输入 `list` 查看所有支持的动作
- 输入 `quit` 退出程序

### 支持的动作

**基础动作**：
- 初始化/复位
- 准备
- 抓取/夹取
- 松开
- 向上

**颜色动作**：
- 黄色/红色/绿色/蓝色

**组合流程**：
- 完整抓取
- 分拣黄色/红色/绿色/蓝色

### 示例对话

```
请输入指令: 帮我把红色的那个拿走
>>> 机械臂执行: [分拣红色]

请输入指令: 今天天气怎么样？
今天天气晴朗，温度适宜。

请输入指令: 初始化机械臂
>>> 机械臂执行: [初始化]
```

## 🔧 技术细节

### 文本分类模型

- **基础模型**：`bert-base-chinese`
- **训练方法**：在数据稀疏场景下，采用 LLM 辅助生成数据集，并进行人工审查和标注
- **推理方式**：ONNX Runtime，支持高效推理
- **分类类别**：
  - 0: 聊天问答（chat）
  - 1: 机械臂控制命令（command）

### 动态模型选择

系统使用 LangChain v1.0 的中间件机制实现动态模型路由：

```python
@wrap_model_call
def dynamic_deepseek_routing(request: ModelRequest, handler) -> ModelResponse:
    # 获取用户输入
    last_user = _get_last_user_text(messages)
    
    # BERT 模型预测
    pred, probs = predict(last_user)
    
    # 根据预测结果选择模型
    if pred == 1:  # command
        request.model = qwen_max_model
    else:  # chat
        request.model = qwen_fast_model
    
    return handler(request)
```

### 模拟模式

当系统检测不到 `Arm_Lib` 库时，会自动进入模拟模式：
- ✅ 文本分类功能正常
- ✅ 动态模型选择功能正常
- ✅ Agent 推理功能正常
- ⚠️ 仅无法执行实际的机械臂控制动作

## 📊 性能指标

经过优化后，系统响应时间显著降低：

| 平台 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| Jetson Orin Nano Super | 4.11 秒 | 2.38 秒 | **42%** ⬇️ |
| RTX 4070 Ti Super | - | 1.47 秒 | - |

✅ 通过所有测试样例

## 📁 项目结构

```
.
├── langchain_onnx_qwen.py    # 主程序文件
├── pyproject.toml            # 项目依赖配置
├── .env                      # 环境变量配置（需自行创建）
├── bert_classifier.onnx      # 文本分类模型（需从 Hugging Face 下载）
└── README.md                 # 项目说明文档
```

## 🔍 代码结构

### 主要模块

1. **文本分类模块**（第 33-59 行）
   - `predict()`: 使用 ONNX 模型进行意图分类

2. **机械臂控制层**（第 73-233 行）
   - `ArmController`: 封装所有机械臂操作
   - 支持模拟模式（无硬件时自动启用）

3. **动态路由中间件**（第 284-313 行）
   - `dynamic_deepseek_routing()`: 实现模型动态选择

4. **主程序**（第 265-368 行）
   - 初始化 Agent 和机械臂控制器
   - 交互循环处理用户输入

## 🛠️ 开发说明

### 添加新动作

在 `ArmController` 类中：

1. 在 `action_map` 字典中添加新的动作映射
2. 实现对应的动作函数（如 `action_xxx()`）
3. 如需新位置，在 `positions` 字典中添加

### 调整模型

修改 `main()` 函数中的模型初始化：

```python
qwen_fast_model = ChatTongyi(model="qwen-flash")  # 聊天模型
qwen_max_model = ChatTongyi(model="qwen3-max")    # 控制模型
```

## 📝 注意事项

1. **模型文件**：确保 `bert_classifier.onnx` 文件在项目根目录
2. **API Key**：必须正确配置 `.env` 文件中的 `DASHSCOPE_API_KEY`
3. **硬件连接**：如有机械臂硬件，确保正确连接并安装 `Arm_Lib` 库
4. **Python 版本**：要求 Python >= 3.13

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

[添加您的许可证信息]

## 🙏 致谢

- LangChain 团队提供的优秀框架
- Hugging Face 提供的模型和工具
- 阿里云百炼平台提供的 API 服务

---

**项目作者**：IsaraYu  
**最后更新**：2024

