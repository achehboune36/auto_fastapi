
import requests
from PIL import Image
import base64
import io
import time
from config import url

def test1():

    # request_body = {
    #     "prompt": "{high quality}, masterpiece, solo, 1boy, dark lord, long horns, evil smile, dark aura, destruction in the background, standing on top of crushed cars, destroyed houses, SciFiRobotic, <lora:SciFiRobotic:1>",
    #     "negative_prompt": "nsfw, low quality, bad anatomy, easynegative",
    #     "seed": -1,
    #     "steps": 50,
    #     "cfg_scale": 7,
    #     "sampler_index": "DPM++ 2M Karras",
    #     "width": 900,
    #     "height": 512
    # }

    request_body = {
        "prompt": "oil on matte canvas, sharp details, the expanse scifi spacescape ceres colony, intricate, highly detailed, digital painting, rich color, smooth, sharp focus, illustration, Unreal Engine 5, 8K, art by artgerm and greg rutkowski and alphonse mucha",
        "negative_prompt": "(worst quality, low quality:1.4), (blurry:1.2), (greyscale, monochrome:1.1), 3D face, cropped, lowres, text, jpeg artifacts, signature, watermark, username, blurry, artist name, trademark, watermark, title, multiple view, nsfw, reference sheet, plump, fat, muscular female, strabismus, out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))",
        "seed": 47552436,
        "steps": 25,
        "cfg_scale": 8,
        "sampler_index": "DPM++ SDE Karras",
        "width": 512,
        "height": 512,
        "batch_size": 1,
        "n_iter": 5,
    }
    response = requests.get(f"http://127.0.0.1:8000/txt2img", json=request_body)
    # print(response)
    # for i, image_base64 in enumerate(response.json()['images']):
    #     image_data = base64.b64decode(image_base64)

    #     filename = f"images/test1.png"

    #     with open(filename, "wb") as f:
    #         f.write(image_data)
    # output = response.json()

    print(response.json())

def test2():
    request_body = {
        "img_path": "images/test1.png"
    }

    response = requests.get(f"{url}/upscale", json=request_body)
    for i, image_base64 in enumerate(response.json()['images']):
        image_data = base64.b64decode(image_base64)

        filename = f"images/outscaled_test1.png"

        with open(filename, "wb") as f:
            f.write(image_data)
    output = response.json()

    print(output)

def test3():
    model_name = "cetusMix_v4.safetensors"
    response = requests.get(f"http://127.0.0.1:8000/set-model?model_name={model_name}")
    print(response.json())

def test4():
    for i in range(20):
        time.sleep(3.5)
        response = requests.get("http://127.0.0.1:8000/progress")
        img64 = response.json()['current_image']
        image_data = base64.b64decode(img64)
        image = Image.open(io.BytesIO(image_data))
        image.save(f"progress/draft_{i}.png")


test1()
# test2()
# test3()
# test4()
