
from fastapi import FastAPI, HTTPException
from PIL import Image
import requests
import base64
from config import url, pil_to_base64
from rq import Queue
from redis import Redis
from jobs import txt2img
import time

app = FastAPI()
redis_conn = Redis()
ai_queue = Queue('ai_queue', connection=redis_conn)
    
@app.get("/")
async def root():
   return {"message": "Hello World"}

@app.get("/txt2img")
async def txt2img_endpoint(request_body: dict):
   prompt = request_body.get("prompt")
   if not prompt:
      return {"error": "Missing mandatory 'prompt' field in the request."}

   query = {
      "prompt": request_body.get("prompt"),
      "negative_prompt": request_body.get("negative_prompt", ""),
      "seed": request_body.get("seed", -1),
      "cfg_scale": request_body.get("cfg_scale", 7),
      "sampler_index": request_body.get("sampler_index", "DPM++ 2M Karras"),
      "width": request_body.get("width", 900),
      "height": request_body.get("height", 512),
      "steps": request_body.get("steps", 25)
   }

   # job = ai_queue.enqueue(txt2img, query)
   # return {
   #    'request_id': job.id,
   # }

   response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=query).json()
   images_out = []

   for i, image_base64 in enumerate(response['images']):
      image_data = base64.b64decode(image_base64)

      timestamp = int(time.time())
      filename = f"images/image_{timestamp}_{i}.png"

      with open(filename, "wb") as f:
         f.write(image_data)

      images_out.append(image_base64)

   return {"images": images_out}

# @app.get("/txt2img/{request_id}")
# async def txt2img_endpoint(request_id: str):
#    job = ai_queue.fetch_job(request_id)
#    if not job:
#       raise HTTPException(status_code=404, detail="request not found")

#    if job.get_status() == 'failed':
#       raise HTTPException(status_code=500, detail="Job failed")

#    if job.get_status() != 'finished':
#       return {
#          'status': job.get_status()
#       }

#    return {
#       'status': job.get_status(),
#       'result': job.result
#    }

@app.get("/upscale")
async def txt2img_endpoint(request_body: dict):
   img = request_body["img_path"]
   if not img:
      return {"error": "Missing mandatory 'img_path' field in the request."}
   img_name = img.split('/')[-1]
   pil_image = Image.open(img)
   request_body = {
      "denoising_strength": 0.1,
      "init_images": [pil_to_base64(pil_image)],
      "script_args": ["",64,"ESRGAN_4x",2],
      "script_name": "SD upscale"
   }

   response = requests.post(url=f'{url}/sdapi/v1/img2img', json={**request_body}, timeout=280).json()
   images_out = []
   for i, image_base64 in enumerate(response['images']):
      image_data = base64.b64decode(image_base64)

      filename = f"images/upscaled_{img_name}.png"

      with open(filename, "wb") as f:
         f.write(image_data)

      images_out.append(image_base64)

   return {"images": images_out}

@app.get("/models")
async def current_model():
   response = requests.get(f'{url}/sdapi/v1/sd-models')
   titles = []

   for item in response.json():
      titles.append(item['model_name'])
   return {"models": titles}

@app.get("/current-model")
async def current_model():
   opt = requests.get(url=f'{url}/sdapi/v1/options')
   print(opt.json())
   current_model = opt.json()["sd_model_checkpoint"]
   return {"message": f"current loaded model is {current_model}"}

@app.get("/set-model")
async def switch_model(model_name: str):
   opt = requests.get(url=f'{url}/sdapi/v1/options')
   opt_json = opt.json()
   opt_json['sd_model_checkpoint'] = model_name
   response = requests.post(url=f'{url}/sdapi/v1/options', json=opt_json)
   if response.status_code == 200:
      return {"message": f"Successfully switched to {model_name} model"}
   else:
      print(response.json())
      return {"message": "Error occurred while switching the model"}
   
@app.get("/progress")
async def get_progress():
   response = requests.get(url=f'{url}/sdapi/v1/progress?skip_current_image=false')
   return response.json()
