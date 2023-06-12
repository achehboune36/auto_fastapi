
import requests
from config import url
import base64
import time

def txt2img(request: dict):
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=request).json()
    images_out = []

    for i, image_base64 in enumerate(response['images']):
        image_data = base64.b64decode(image_base64)

        timestamp = int(time.time())
        filename = f"images/image_{timestamp}_{i}.png"

        with open(filename, "wb") as f:
            f.write(image_data)

        images_out.append(image_base64)

    return {"images": images_out}
