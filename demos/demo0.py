import requests
import cv2
import base64
from PIL import Image
import numpy as np
from io import BytesIO

def main0():
    url = "http://127.0.0.1:10860/sdapi/v1/txt2img"

    payload = {
        "prompt": "A beautiful landscape with mountains and rivers",
        "steps": 20,
        "cfg_scale": 7.5,
        "width": 512,
        "height": 512
    }

    response = requests.post(url, json=payload)

    # 保存生成的图片
    if response.status_code == 200:
        result = response.json()
        image_data = result['images'][0]
        with open("output_img.png", "wb") as f:
            f.write(base64.b64decode(image_data))

def main1():
    import requests
    import base64
    from PIL import Image
    from io import BytesIO

    # 加载初始图片
    image = Image.open("/workspace/stable-diffusion-webui/demos/images/in/18.png")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # ControlNet 模型的参考条件图片（例如姿态图）
    control_image = Image.open("/workspace/stable-diffusion-webui/demos/images/in/16.png")
    buffered_control = BytesIO()
    control_image.save(buffered_control, format="PNG")
    control_img_str = base64.b64encode(buffered_control.getvalue()).decode()

    # API 地址
    url = "http://127.0.0.1:10860/sdapi/v1/img2img"

    # 构建请求数据
    payload = {
        "init_images": [img_str],
        "prompt": "A futuristic city skyline",
        "steps": 20,
        "cfg_scale": 7.5,
        "width": 512,
        "height": 512,
        "controlnet_input_images": [control_img_str],  # ControlNet 输入图像
        "controlnet_enable": True,                    # 启用 ControlNet
        "controlnet_module": "pose",                  # ControlNet 模块（如“姿态”、“草图”等）
        "controlnet_model": "control_v11p_sd15_openpose"  # ControlNet 模型
    }

    # 发送请求
    response = requests.post(url, json=payload)

    # 处理响应
    if response.status_code == 200:
        result = response.json()
        image_data = result['images'][0]
        image_data = base64.b64decode(image_data)

        # 保存生成的图片
        with open("output_img.png", "wb") as f:
            f.write(image_data)
    else:
        print(f"Error: {response.status_code}")


# 生成全0（黑色）的掩码图像
def create_black_mask(width, height):
    return Image.new("L", (width, height), 0)  # "L" 表示单通道图像，0 是黑色

# 生成全1（白色）的掩码图像
def create_white_mask(width, height):
    return Image.new("L", (width, height), 255)  # 255 是白色

# 将图像转换为 base64 字符串
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()



def main2():
    import requests
    import base64
    from PIL import Image
    from io import BytesIO

    # 加载初始图片
    img_path0 = "/workspace/stable-diffusion-webui/demos/images/in/18.png"
    image = Image.open(img_path0)
    # image = Image.open("input_img.png")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # ControlNet 0 的参考条件图片（instant_id_face_embedding）
    img_path1 = "/workspace/stable-diffusion-webui/demos/images/in/16.png"
    control_image_0 = Image.open(img_path1)
    buffered_control_0 = BytesIO()
    control_image_0.save(buffered_control_0, format="PNG")
    control_img_str_0 = base64.b64encode(buffered_control_0.getvalue()).decode()

    # ControlNet 1 的参考条件图片（instant_id_face_keypoints）
    control_image_1 = Image.open(img_path0)
    buffered_control_1 = BytesIO()
    control_image_1.save(buffered_control_1, format="PNG")
    control_img_str_1 = base64.b64encode(buffered_control_1.getvalue()).decode()

    # API 地址
    url = "http://127.0.0.1:10860/sdapi/v1/img2img"

    # mask_image = create_white_mask(512, 512)
    mask_image = cv2.imread("/workspace/stable-diffusion-webui/demos/images/in/18_gray.png")
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    # mask_image = Image.open("/workspace/stable-diffusion-webui/demos/images/in/18_gray.png").convert()
    # mask_image = np.ones(image.size[:2], np.uint8) * 255
    # mask_image[:, -200:] = 0
    print("mask_image.shape:", mask_image.shape)
    mask_image = Image.fromarray(mask_image)
    mask_base64 = image_to_base64(mask_image)

    # 构建请求数据
    payload = {
        "init_images": [img_str],
        "prompt": "Realistic 1boy, handsome",
        "mask": mask_base64,
        "inpainting_fill": 1,
        "steps": 20,
        "cfg_scale": 7.5,
        "width": 512,
        "height": 512,
        "controlnet": [
            {
                "input_image": control_img_str_0,  # 第一组 ControlNet 的输入
                "module": "instant_id_face_embedding",  # 指定 preprocessor
                "model": "ip-adapter_instant_id_sdxl [eb2d3ec0]", # 指定 ControlNet 模型
                "weight": 1.0,
            },
            {
                "input_image": control_img_str_1,  # 第二组 ControlNet 的输入
                "module": "instant_id_face_keypoints",  # 指定 preprocessor
                "model": "control_instant_id_sdxl [c5c25a50]",  # 指定 ControlNet 模型
                "weight": 0.3,
            }
        ]
    }

    # 发送请求
    response = requests.post(url, json=payload)

    # 处理响应
    if response.status_code == 200:
        result = response.json()
        image_data = result['images'][0]
        image_data = base64.b64decode(image_data)

        # 保存生成的图片
        with open("output_img.png", "wb") as f:
            f.write(image_data)
    else:
        print(f"Error: {response.status_code}")

main2()