#!/usr/bin/env python3
# coding=utf-8
"""
文字控制机械臂系统 - 分层解耦版本
Part 1: 机械臂控制层 (ArmController)
Part 2: AI 决策层 (AIDecisionMaker)
Part 3: 主程序逻辑
"""

import json
import time
import re
from openai import OpenAI
import os
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv 

import numpy as np
import onnxruntime as ort
from transformers import BertTokenizer

MODEL_NAME = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
session = ort.InferenceSession("bert_classifier.onnx")
load_dotenv(override=True)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="np",      # 生成 numpy
        padding="max_length",
        truncation=True,
        max_length=128
    )

    # 🔴 关键：显式转成 int64
    input_ids = inputs["input_ids"].astype("int64")
    attention_mask = inputs["attention_mask"].astype("int64")

    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    logits = session.run(None, ort_inputs)[0]
    probs = softmax(logits[0])
    pred = np.argmax(probs)

    return pred, probs

# 尝试导入机械臂库，如果是在没有机械臂的电脑上开发AI，可以避免报错
try:
    from Arm_Lib import Arm_Device
    HAS_ARM = True
except ImportError:
    print("警告: 未检测到 Arm_Lib，进入模拟模式 (仅用于调试AI逻辑)")
    HAS_ARM = False


# ==========================================
# Part 1: 机械臂控制层 (只负责动，不负责思考)
# ==========================================
class ArmController:
    def __init__(self):
        self.connected = False
        if HAS_ARM:
            try:
                self.arm = Arm_Device('/dev/ttyCH341USB0')
                time.sleep(0.1)
                self.connected = True
            except Exception as e:
                print(f"机械臂连接失败: {e}")
        
        # 预定义位置数据
        self.positions = {
            "初始位置": [90, 130, 0, 0, 90],
            "准备位置": [90, 80, 50, 50, 270],
            "抓取位置": [90, 53, 33, 36, 270],
            "放置黄色": [65, 22, 64, 56, 270],
            "放置红色": [117, 19, 66, 56, 270],
            "放置绿色": [136, 66, 20, 29, 270],
            "放置蓝色": [44, 66, 20, 28, 270],
        }
        
        # 注册动作回调
        # 这里定义了外界可以通过什么"关键词"来驱动机械臂
        self.action_map = {
            # 基础动作
            "初始化": self.action_init,
            "复位": self.action_init,
            "准备": self.action_ready,
            "抓取": self.action_grab,
            "夹取": self.action_grab,
            "松开": self.action_release,
            "向上": self.action_move_up,
            
            # 颜色动作
            "黄色": self.action_place_yellow,
            "红色": self.action_place_red,
            "绿色": self.action_place_green,
            "蓝色": self.action_place_blue,
            
            # 组合流程
            "完整抓取": self.action_full_grab_sequence,
            "分拣黄色": self.action_sort_yellow,
            "分拣红色": self.action_sort_red,
            "分拣绿色": self.action_sort_green,
            "分拣蓝色": self.action_sort_blue,
        }

        if self.connected:
            self.init_arm()

    # --- 接口函数 ---
    
    def get_available_commands(self):
        """对外提供机械臂支持的所有指令列表"""
        return list(self.action_map.keys())

    def execute(self, command_key):
        """统一执行接口：接收字符串，执行对应动作"""
        if command_key in self.action_map:
            print(f">>> 机械臂执行: [{command_key}]")
            self.action_map[command_key]()  # 调用对应的函数
            return True
        else:
            print(f"错误: 机械臂不支持指令 [{command_key}]")
            return False

    # --- 硬件底层函数 ---

    def init_arm(self):
        print("初始化机械臂...")
        self.arm_clamp_block(0)
        self.arm_move(self.positions["初始位置"], 1000)

    def arm_clamp_block(self, enable):
        if not self.connected: return
        if enable == 0:
            self.arm.Arm_serial_servo_write(6, 60, 400)
        else:
            self.arm.Arm_serial_servo_write(6, 130, 400)
        time.sleep(0.5)

    def arm_move(self, position, s_time=500):
        if not self.connected: return
        for i in range(5):
            servo_id = i + 1
            if servo_id == 5:
                time.sleep(0.1)
                self.arm.Arm_serial_servo_write(servo_id, position[i], int(s_time * 1.2))
            else:
                self.arm.Arm_serial_servo_write(servo_id, position[i], s_time)
            time.sleep(0.01)
        time.sleep(s_time / 1000)

    def arm_move_up(self):
        if not self.connected: return
        self.arm.Arm_serial_servo_write(2, 90, 1500)
        self.arm.Arm_serial_servo_write(3, 90, 1500)
        self.arm.Arm_serial_servo_write(4, 90, 1500)
        time.sleep(1.5)

    # --- 动作具体实现 ---
    # (这里省略了重复的print，保留核心逻辑)
    
    def action_init(self):
        self.arm_clamp_block(0)
        self.arm_move(self.positions["初始位置"], 1000)

    def action_ready(self):
        self.arm_move(self.positions["准备位置"], 1000)

    def action_grab(self):
        self.arm_move(self.positions["抓取位置"], 1000)
        self.arm_clamp_block(1)

    def action_release(self):
        self.arm_clamp_block(0)

    def action_move_up(self):
        self.arm_move_up()

    def action_place_yellow(self):
        self.arm_move(self.positions["放置黄色"], 1000)
    def action_place_red(self):
        self.arm_move(self.positions["放置红色"], 1000)
    def action_place_green(self):
        self.arm_move(self.positions["放置绿色"], 1000)
    def action_place_blue(self):
        self.arm_move(self.positions["放置蓝色"], 1000)

    def action_full_grab_sequence(self):
        self.action_ready()
        time.sleep(0.5)
        self.action_grab()
        time.sleep(0.5)
        self.action_move_up()

    def action_sort_yellow(self):
        self.action_full_grab_sequence()
        self.action_place_yellow()
        self.action_release()
        self.action_move_up()
    
    def action_sort_red(self):
        self.action_full_grab_sequence()
        self.action_place_red()
        self.action_release()
        self.action_move_up()

    def action_sort_green(self):
        self.action_full_grab_sequence()
        self.action_place_green()
        self.action_release()
        self.action_move_up()

    def action_sort_blue(self):
        self.action_full_grab_sequence()
        self.action_place_blue()
        self.action_release()
        self.action_move_up()
    ###########################################
def build_system_prompt(valid_actions) -> str:
    """
    根据 valid_actions 构造 system prompt
    """
    actions_str = "、".join(valid_actions)

    system_prompt = f"""如果你是max模型，那么，你是一个机械臂指令解析器。
请从以下【可用动作库】中，选择一个最符合用户意图的动作关键词。

【可用动作库】：
{actions_str}

规则：
1. 只返回一个关键词。
2. 如果无法匹配，返回"未知"。
3. 不要包含标点符号或解释性文字。

示例：
- 用户："帮我把红色的那个拿走" -> 返回："分拣红色"
- 用户："手抬起来" -> 返回："向上"
- 用户："松手" -> 返回："松开"
如果你是flash模型，那么请正常聊天，不要返回动作关键词。
"""
    return system_prompt
    


# ==========================================
# Part 3: 主程序 (作为胶水连接两部分)
# ==========================================
chat_flag=1                #1为chat  2为reasoner
def main():
    global chat_flag                #1为chat  2为reasoner
    # 1. 实例化 AI 控制器 (你可以在这里开发AI，无需连接机械臂)

    basic_model = ChatDeepSeek(model="deepseek-chat")        # 简单问题：快速、经济
    reasoner_model = ChatDeepSeek(model="deepseek-reasoner") # 复杂问题：推理更强
    qwen_fast_model = ChatTongyi(model="qwen-flash")
    qwen_max_model = ChatTongyi(model="qwen3-max")
    

    ############################################################################
    def _get_last_user_text(messages) -> str:
        """从消息列表中取最近一条用户消息文本（无则返回空串）"""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                # content 可能是纯字符串或富内容；这里只处理为字符串的常见情况
                return m.content if isinstance(m.content, str) else ""
        return ""

    @wrap_model_call
    def dynamic_deepseek_routing(request: ModelRequest, handler) -> ModelResponse:
        """
        根据对话复杂度动态选择 DeepSeek 模型：
        - 复杂：deepseek-reasoner
        - 简单：deepseek-chat
        """
        global chat_flag
        messages = request.state.get("messages", [])
        
        # 获取用户的最后一条消息
        last_user = _get_last_user_text(messages)
        
        # 使用 BERT 模型预测复杂度
        pred, probs = predict(last_user)
        
        print(f"[BERT推理] 预测标签: {pred}, 预测概率: {probs}")
        
        # 根据预测结果选择模型
        if pred == 1:  # 如果是复杂问题
            chat_flag = 2
            request.model = qwen_max_model
        else:  # 如果是简单问题
            chat_flag = 1
            request.model = qwen_fast_model

        print(f"选择的模型: {request.model}")

        # 调用被包裹的下游（真正的模型调用）
        return handler(request)

    #############################################################################


    # 2. 实例化 机械臂控制器

    arm_robot = ArmController()
    valid_actions = arm_robot.get_available_commands()
    system_prompt = build_system_prompt(valid_actions)
    agent = create_agent(model=qwen_fast_model,system_prompt=system_prompt,middleware=[dynamic_deepseek_routing])
    print("\n=== 智能机械臂控制系统启动 ===")
    print("输入 'quit' 退出，'list' 查看支持动作")

    while True:
        try:
            # A. 获取用户输入
            user_input = input("\n请输入指令: ").strip()
            
            if not user_input: continue
            if user_input == 'quit': break
            
            # 辅助命令
            if user_input == 'list':
                print("支持的动作:", arm_robot.get_available_commands())
                continue
            
            # B. AI 进行决策
            # 关键点：AI只需要知道"有哪些动作可选"(list)，不需要知道动作怎么做
            messages = {"messages": [{"role": "user", "content": user_input}]}
            begin_time = time.time()
            reply = agent.invoke(messages)
            end_time = time.time()
            print(f"AI响应时间: {end_time - begin_time:.2f} 秒")
            decision_key = reply["messages"][-1].content
            # C. 机械臂执行决策
            # 关键点：机械臂只接收标准化的字符串Key，不需要知道这是AI算出来的还是人输入的
            if chat_flag==1:
                print(decision_key)
            elif chat_flag==2 and decision_key != "未知":
                arm_robot.execute(decision_key)
            else:
                print("抱歉，我没听懂您的指令，或者该指令不在支持范围内。")
                
        except KeyboardInterrupt:
            print("\n程序中断退出")
            break
        except Exception as e:
            print(f"运行时错误: {e}")

    # 清理工作
    if HAS_ARM and arm_robot.connected:
        del arm_robot.arm

if __name__ == '__main__':
    main()