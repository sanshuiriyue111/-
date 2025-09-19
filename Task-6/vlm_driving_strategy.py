import os
import json
import base64
from PIL import Image
import io
import requests

# --------------------------
# 配置参数
# --------------------------
VLM_API_URL = "http://localhost:11434/api/chat"
DATA_FOLDER = "/Users/duqiu/Desktop/焦点计划/Task-6/data" 
MODEL_NAME = "llava:latest"


# --------------------------
# 工具函数：清洗LLaVA返回的JSON文本
# --------------------------
def clean_json_text(text):
    """
    去除LLaVA返回结果中的多余标记（如```json）和转义字符，确保JSON格式正确
    """
    if not text:
        return ""
    
    # 1. 去除首尾空白字符
    cleaned = text.strip()
    
    # 2. 去除可能的```json开头和```结尾标记
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-len("```")]
    
    # 3. 去除可能的转义字符（如\_ → _）
    cleaned = cleaned.replace("\\_", "_")
    
    # 4. 再次去除首尾空白（处理标记去除后的残留空格）
    return cleaned.strip()


# --------------------------
# 1. 图像处理工具
# --------------------------
def image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            if img.format != "PNG":
                raise ValueError(f"图像 {image_path} 不是PNG格式")
            
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"图像处理失败（{image_path}）：{str(e)}")
        return None


# --------------------------
# 2. 视觉感知（带结果清洗）
# --------------------------
感知提示词 = """
请分析这张交通场景图像，识别以下关键驾驶元素，并用JSON格式返回结果（仅返回JSON，不要添加任何额外文字、标记或解释）：
{
    "front_car_distance": 前车与本车的距离（单位：米，无前车则为null）,
    "has_pedestrian": 是否有行人（true/false）,
    "traffic_light_color": 交通信号灯颜色（"red"/"green"/"yellow"/null，无信号灯则为null）,
    "lane_obstacle": 本车道是否有障碍物（true/false）,
    "adjacent_lane_car": 相邻车道是否有车辆（true/false）
}
"""

def call_vlm_perception(image_base64):
    if not image_base64:
        return None
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": 感知提示词,
                "images": [image_base64]
            }
        ],
        "stream": False
    }
    
    try:
        response = requests.post(
            VLM_API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30
        )
        response.raise_for_status()
        
        # 提取结果并清洗
        result = response.json()
        raw_content = result["message"]["content"].strip()
        cleaned_content = clean_json_text(raw_content)
        
        # 尝试解析清洗后的JSON
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        print(f"清洗后仍无法解析JSON：{cleaned_content}（原始内容：{raw_content}）")
        return None
    except Exception as e:
        print(f"感知API调用失败：{str(e)}")
        return None


# --------------------------
# 3. 形式化推理（带结果清洗）
# --------------------------
def get_planning_prompt(perception_result):
    return f"""
基于以下交通场景感知结果，按照规则生成驾驶策略，用JSON格式返回（仅返回JSON，不要添加任何额外文字、标记或解释）：
感知结果：{json.dumps(perception_result)}

规则：
1. 若traffic_light_color为"red" → 刹车至停车
   {{
       "brake": true,
       "brake_strength": 1.0,
       "throttle": false,
       "steering_angle": 0.0
   }}
2. 若front_car_distance < 10米 → 紧急刹车
   {{
       "brake": true,
       "brake_strength": 0.8,
       "throttle": false,
       "steering_angle": 0.0
   }}
3. 若front_car_distance > 30米且traffic_light_color为"green" → 加速前进
   {{
       "brake": false,
       "throttle": true,
       "throttle_strength": 0.5,
       "steering_angle": 0.0
   }}
4. 若lane_obstacle为true且adjacent_lane_car为false → 左变道避让
   {{
       "brake": false,
       "throttle": true,
       "throttle_strength": 0.3,
       "steering_angle": -30.0
   }}
5. 其他情况 → 保持当前速度直行
   {{
       "brake": false,
       "throttle": true,
       "throttle_strength": 0.3,
       "steering_angle": 0.0
   }}
"""

def call_vlm_planning(image_base64, perception_result):
    if not image_base64 or not perception_result:
        return None
    
    planning_prompt = get_planning_prompt(perception_result)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": planning_prompt,
                "images": [image_base64]
            }
        ],
        "stream": False
    }
    
    try:
        response = requests.post(
            VLM_API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30
        )
        response.raise_for_status()
        
        # 提取结果并清洗
        result = response.json()
        raw_content = result["message"]["content"].strip()
        cleaned_content = clean_json_text(raw_content)
        
        # 尝试解析清洗后的JSON
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        print(f"清洗后仍无法解析规划JSON：{cleaned_content}（原始内容：{raw_content}）")
        return None
    except Exception as e:
        print(f"规划API调用失败：{str(e)}")
        return None


# --------------------------
# 4. 批量处理所有图像
# --------------------------
def process_all_images():
    if not os.path.exists(DATA_FOLDER):
        print(f"错误：文件夹 {DATA_FOLDER} 不存在")
        return
    
    for filename in os.listdir(DATA_FOLDER):
        if filename.lower().endswith(".png"):
            file_path = os.path.join(DATA_FOLDER, filename)
            print(f"\n===== 处理图像：{filename} =====")
            
            # 步骤1：图像转Base64
            img_base64 = image_to_base64(file_path)
            if not img_base64:
                continue
            
            # 步骤2：视觉感知
            perception = call_vlm_perception(img_base64)
            if not perception:
                continue
            print(f"感知结果：{json.dumps(perception, indent=2)}")
            
            # 步骤3：生成驾驶策略
            strategy = call_vlm_planning(img_base64, perception)
            if not strategy:
                continue
            print(f"驾驶策略：{json.dumps(strategy, indent=2)}")
            
            # 保存结果
            result_path = os.path.join(DATA_FOLDER, f"{filename}_result.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump({"perception": perception, "strategy": strategy}, f, indent=2)
            print(f"结果已保存至：{result_path}")


# --------------------------
# 主程序
# --------------------------
if __name__ == "__main__":
    process_all_images()
    print("\n===== 所有图像处理完成 =====")

