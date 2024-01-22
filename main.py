
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
import requests
import base64
from config import url, pil_to_base64, llama_host, get_llama_request_body
from rq import Queue
from redis import Redis
from jobs import txt2img, upscale,img2img
from rq.registry import FinishedJobRegistry
import io
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pydantic import BaseModel
from fastapi.responses import FileResponse
import time
from fastapi.middleware.cors import CORSMiddleware
import torch
import gc
from transformers import AutoProcessor, BarkModel
import soundfile as sf

app = FastAPI()
redis_conn = Redis()
ai_queue = Queue('ai_queue', connection=redis_conn)
registry = FinishedJobRegistry(queue=ai_queue)



app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
      "width": request_body.get("width", 512),
      "height": request_body.get("height", 512),
      "steps": request_body.get("steps", 25),
      "n_iter": request_body.get("n_iter", 1),
      "Asymmetric_tiling":{"args":[True,True,False,0,-1]},
      "tiling":request_body.get("tiling", False),
       "hr_sampler_name": request_body.get("hr_sampler_name", "Euler"),
       "sampler_name": request_body.get("sampler_name", "Euler")
   }

   job = ai_queue.enqueue(txt2img, query)
   return {
      'task_id': job.id,
   }

@app.get("/img2img")
async def img2img_endpoint(request_body: dict):
   prompt = request_body.get("prompt")
   init_images = request_body.get("init_images")
   if not prompt:
      return {"error": "Missing mandatory 'prompt' field in the request."}
   if not init_images:
      return {"error": "Missing mandatory 'init_images' field in the request."}
   
   query = {
      "prompt": request_body.get("prompt"),
      "negative_prompt": request_body.get("negative_prompt", ""),
      "seed": request_body.get("seed", -1),
      "cfg_scale": request_body.get("cfg_scale", 7),
      "sampler_index": request_body.get("sampler_index", "DPM++ 2M Karras"),
      "width": request_body.get("width", 512),
      "height": request_body.get("height", 512),
      "steps": request_body.get("steps", 25),
      "n_iter": request_body.get("n_iter", 1),
      "Asymmetric_tiling":{"args":[True,True,False,0,-1]},
      "tiling":request_body.get("tiling", False),
      "hr_sampler_name": request_body.get("hr_sampler_name", "Euler"),
      "sampler_name": request_body.get("sampler_name", "Euler"),
      "init_images":request_body.get("init_images")
   }

   job = ai_queue.enqueue(img2img, query)
   return {
      'task_id': job.id,
   }


@app.get("/txt2img/{task_id}")
async def txt2img_endpoint(task_id: str):
   job = ai_queue.fetch_job(task_id)
   if not job:
      raise HTTPException(status_code=404, detail="request not found")

   if job.get_status() == 'failed':
      raise HTTPException(status_code=500, detail="Job failed")

   if job.get_status() == 'started':
      response = requests.get(url=f'{url}/sdapi/v1/progress?skip_current_image=true')
      return {
         'status': job.get_status(),
         'progress': response.json()['progress'],
         'eta_relative': response.json()['eta_relative']
      }

   if job.get_status() != 'finished':
      return {
         'status': job.get_status()
      }

   return {
      'status': job.get_status(),
      'result': job.result
   }

import base64

@app.get("/upscale")
async def upscale_endpoint(request_body: dict):
   img_base64 = request_body.get("img_base64")
   if not img_base64:
      return {"error": "Missing mandatory 'img_base64' field in the request."}

   img_data = base64.b64decode(img_base64)
   pil_image = Image.open(io.BytesIO(img_data))

   request_body = {
      "denoising_strength": 0.1,
      "init_images": [pil_to_base64(pil_image)],
      "script_args": ["", 64, "ESRGAN_4x", 2],
      "script_name": "SD upscale"
   }

   job = ai_queue.enqueue(upscale, request_body)
   return {
      'task_id': job.id,
   }

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
   response = requests.get(url=f'{url}/sdapi/v1/progress?skip_current_image=true')
   return response.json()

class MusicRequest(BaseModel):
    prompt: str
    duration: int = 10

@app.post("/txt2music")
async def get_progress(request: Request, request_body: MusicRequest):
    model = MusicGen.get_pretrained('melody')
    prompt = request_body.prompt
    duration = request_body.duration

    if not prompt:
        return {"error": "Missing mandatory 'prompt' field in the request."}

    if duration > 60:
        duration = 60

    model.set_generation_params(duration=duration)
    wav = model.generate_unconditional(4)
    descriptions = [prompt]
    wav = model.generate(descriptions)

    mp4_files = []
    for idx, one_wav in enumerate(wav):
      mp4_path = f'output/{idx}'
      audio_write(mp4_path, one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
      mp4_files.append(mp4_path+'.wav')
    del model
    del wav
    torch.cuda.empty_cache()
    gc.collect()
    return FileResponse(mp4_files[0], media_type="video/mp4")

@app.get('/png-info')
async def png_info(payload: dict):
   response = requests.post(url=f'{url}/sdapi/v1/png-info', json=payload)
   return response

@app.get('/llama')
async def llama(prompt: str):
   payload = get_llama_request_body(prompt)
   response = requests.post(url=f'{llama_host}/api/v1/generate', json=payload)
   return response

class AudioRequest(BaseModel):
    prompt: str

@app.post("/generate_audio")
def generate_audio_endpoint(request: AudioRequest):

    # Bark AI congig 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark").to(device)
    
    voice_preset = "v2/en_speaker_6"
    inputs = processor(request.prompt, voice_preset=voice_preset, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        audio_array = model.generate(**inputs)
    
    audio_array = audio_array.cpu().numpy().squeeze()

    buffer = io.BytesIO()
    sf.write(buffer, audio_array, 24000, format='WAV')
    buffer.seek(0)

    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    del processor
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return {"audio_base64": audio_base64}
