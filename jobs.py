
import requests
from config import url
import base64
import time


def txt2img(request: dict):
    try:
        requests.post(url=f'{url}/sdapi/v1/txt2img', json=request, timeout=5)
    except Exception as e:
        pass
    
    response = requests.get(url=f'{url}/sdapi/v1/progress?skip_current_image=true')
    while response.json()["progress"] != 0:
        time.sleep(3)
        response = requests.get(url=f'{url}/sdapi/v1/progress?skip_current_image=true')
    time.sleep(5)
    images_out = []
    for i, image_base64 in enumerate(response.json()['current_processed_images']):
        images_out.append(image_base64)

    return {"images": images_out}

def upscale(request: dict):
    try:
        requests.post(url=f'{url}/sdapi/v1/img2img', json=request, timeout=5)
    except Exception as e:
        pass

    response = requests.get(url=f'{url}/sdapi/v1/progress?skip_current_image=true')
    while response.json()["progress"] != 0:
        time.sleep(3)
        response = requests.get(url=f'{url}/sdapi/v1/progress?skip_current_image=true')
    time.sleep(5)
    return {"image": response.json()['current_processed_images'][0]}
