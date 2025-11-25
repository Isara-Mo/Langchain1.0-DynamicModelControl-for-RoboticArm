<div align="right">

**Language / 语言**: [English](#) | [中文](#chinese-version)

</div>

# LangChain 1.0 Dynamic Model Control for Robotic Arm System

An intelligent robotic arm control system built on LangChain v1.0, enabling multi-functional interaction between chat Q&A and automatic robotic arm control. The system implements dynamic model selection through the middleware's `wrap_model_call` mechanism, automatically switching between models of different capabilities based on user intent, significantly improving system response speed.

The text classification model weights has been placed on https://huggingface.co/IsaraYu/Chat-Command_TextClassification/tree/main
## ✨ Features

- 🤖 **Intelligent Intent Recognition**: Based on fine-tuned BERT model, accurately determines whether user input is "chat Q&A" or "robotic arm control command"
- 🔄 **Dynamic Model Routing**: Automatically selects the most suitable model based on intent classification results, significantly reducing system response time through dynamic model selection
  - **Chat Mode**: Uses `qwen-flash` for fast responses to daily Q&A
  - **Control Mode**: Uses `qwen3-max` to handle complex robotic arm control tasks
- 🦾 **Robotic Arm Control**: Supports various predefined actions and combined workflows
- 🔧 **Simulation Mode**: Can run text classification and model selection functions normally without physical robotic arm hardware

## 🚀 Quick Start

### 1. Download Model Weights

Download the ONNX weights of the text classification model from Hugging Face:

```bash
# Visit the following link to download bert_classifier.onnx
https://huggingface.co/IsaraYu/Chat-Command_TextClassification/tree/main
```

Place the downloaded `bert_classifier.onnx` file in the project root directory.

### 2. Install Dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or using `pip`:

```bash
pip install -e .
```

### 3. Configure API Key

1. Visit [Alibaba Cloud Bailian Console](https://bailian.console.aliyun.com/?tab=model#/api-key) to get your API Key
2. Create a `.env` file in the project root directory
3. Add the following configuration:

```env
DASHSCOPE_API_KEY=your_api_key_here
```

### 4. Run the Program

```bash
python langchain_onnx_qwen.py
```

### Workflow

```
User Input 
  ↓
BERT Text Classification Model (judge chat/command)
  ↓
Dynamic Routing Middleware
  ├─ chat → qwen-flash (fast response)
  └─ command → qwen3-max (precise control)
  ↓
Agent Processing
  ↓
Robotic Arm Execution (if command and successfully recognized)
```

## 📋 System Requirements

- Python >= 3.13
- Environment supporting ONNX Runtime
- (Optional) Yabo robotic arm hardware and Arm_Lib library

## 📖 Usage

### Basic Commands

- Enter natural language commands to control the robotic arm
- Enter `list` to view all supported actions
- Enter `quit` to exit the program

### Supported Actions

**Basic Actions**:
- Initialize/Reset
- Ready
- Grab/Clamp
- Release
- Move Up

**Color Actions**:
- Yellow/Red/Green/Blue

**Combined Workflows**:
- Full Grab Sequence
- Sort Yellow/Red/Green/Blue

### Example Dialogues

```
Please enter command: Help me take away the red one
>>> Robotic arm executing: [Sort Red]

Please enter command: What's the weather like today?
The weather is sunny and the temperature is pleasant.

Please enter command: Initialize robotic arm
>>> Robotic arm executing: [Initialize]
```

## 🔧 Technical Details

### Text Classification Model

- **Base Model**: `bert-base-chinese`
- **Training Method**: In data-sparse scenarios, uses LLM-assisted dataset generation with manual review and annotation
- **Inference Method**: ONNX Runtime for efficient inference
- **Classification Categories**:
  - 0: Chat Q&A (chat)
  - 1: Robotic arm control command (command)

### Dynamic Model Selection

The system uses LangChain v1.0's middleware mechanism to implement dynamic model routing:

```python
@wrap_model_call
def dynamic_deepseek_routing(request: ModelRequest, handler) -> ModelResponse:
    # Get user input
    last_user = _get_last_user_text(messages)
    
    # BERT model prediction
    pred, probs = predict(last_user)
    
    # Select model based on prediction result
    if pred == 1:  # command
        request.model = qwen_max_model
    else:  # chat
        request.model = qwen_fast_model
    
    return handler(request)
```

### Simulation Mode

When the system cannot detect the `Arm_Lib` library, it automatically enters simulation mode:
- ✅ Text classification function works normally
- ✅ Dynamic model selection function works normally
- ✅ Agent reasoning function works normally
- ⚠️ Only unable to execute actual robotic arm control actions

## 📊 Performance Metrics

After optimization, system response time is significantly reduced:

| Platform | Before Optimization | After Optimization | Improvement |
|----------|---------------------|-------------------|-------------|
| Jetson Orin Nano Super | 4.11 seconds | 2.38 seconds | **42%** ⬇️ |
| RTX 4070 Ti Super | - | 1.47 seconds | - |

✅ Passed all test cases

## 📁 Project Structure

```
.
├── langchain_onnx_qwen.py    # Main program file
├── pyproject.toml            # Project dependency configuration
├── .env                      # Environment variable configuration (create manually)
├── bert_classifier.onnx      # Text classification model (download from Hugging Face)
└── README.md                 # Project documentation
```

## 🔍 Code Structure

### Main Modules

1. **Text Classification Module** (Lines 33-59)
   - `predict()`: Uses ONNX model for intent classification

2. **Robotic Arm Control Layer** (Lines 73-233)
   - `ArmController`: Encapsulates all robotic arm operations
   - Supports simulation mode (automatically enabled without hardware)

3. **Dynamic Routing Middleware** (Lines 284-313)
   - `dynamic_deepseek_routing()`: Implements dynamic model selection

4. **Main Program** (Lines 265-368)
   - Initializes Agent and robotic arm controller
   - Interactive loop for processing user input

## 🛠️ Development Guide

### Adding New Actions

In the `ArmController` class:

1. Add new action mapping in the `action_map` dictionary
2. Implement corresponding action function (e.g., `action_xxx()`)
3. Add new positions in the `positions` dictionary if needed

### Adjusting Models

Modify model initialization in the `main()` function:

```python
qwen_fast_model = ChatTongyi(model="qwen-flash")  # Chat model
qwen_max_model = ChatTongyi(model="qwen3-max")    # Control model
```

## 📝 Notes

1. **Model File**: Ensure `bert_classifier.onnx` file is in the project root directory
2. **API Key**: Must correctly configure `DASHSCOPE_API_KEY` in the `.env` file
3. **Hardware Connection**: If you have robotic arm hardware, ensure proper connection and install `Arm_Lib` library
4. **Python Version**: Requires Python >= 3.13

## 🤝 Contributing

Welcome to submit Issues and Pull Requests!

## 📄 License

[Add your license information]

## 🙏 Acknowledgments

- LangChain team for the excellent framework
- Hugging Face for models and tools
- Alibaba Cloud Bailian platform for API services

---

**Project Author**: IsaraYu  
**Last Updated**: 2024

---

<div id="chinese-version"></div>

<div align="right">

**Language / 语言**: [English](#) | [中文](#chinese-version)

</div>

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
https://huggingface.co/IsaraYu/Chat-Command_TextClassification/tree/main
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
