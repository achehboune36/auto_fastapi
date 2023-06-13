
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

    images_out = []
    for i, image_base64 in enumerate(response.json()['current_processed_images']):
        # #saving a copy of the generated images
        # image_data = base64.b64decode(image_base64)

        # timestamp = int(time.time())
        # filename = f"images/image_{timestamp}_{i}.png"

        # with open(filename, "wb") as f:
        #     f.write(image_data)

        images_out.append(image_base64)

    return {"images": images_out}
