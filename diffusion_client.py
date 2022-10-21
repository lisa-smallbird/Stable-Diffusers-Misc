import os
import datetime
import torch
import math
from PIL import Image
from pytorch_lightning import seed_everything
from random import randint
from diffusers import StableDiffusionPipeline,StableDiffusionImg2ImgPipeline,StableDiffusionInpaintPipeline
from stable_diffusion_videos.stable_diffusion_pipeline import StableDiffusionWalkPipeline
from stable_diffusion_videos.stable_diffusion_img2img_pipeline import StableDiffusionWalkImg2ImgPipeline
from stable_diffusion_videos.stable_diffusion_walk import StableDiffusionVideoCreater

class DiffusionClient:
  def __init__(self):
    t_delta = datetime.timedelta(hours=9)
    self.JST = datetime.timezone(t_delta, 'JST')
    self.pipe_type = ""
    self.library = ""
    self.revision = ""
    self.setupInitDirs()
    self.seed = randint(0, 500000000)
    self.vc = StableDiffusionVideoCreater()
    self.text_encoder=None
    self.tokenizer=None
    self.less_memory = True
    self.results = []
    self.nsfw_ok = False
    self.hugging_face_token = ""
    self.model_loaded = False
  def setSeed(self, seed):
    self.seed = seed
  def setupInitDirs(self):
    self.base_path = f'/content/drive/MyDrive/diffusion/'
    self.user_data_path = self.base_path + f'user-data'
    self.results_path = self.base_path + f'results'
    now = datetime.datetime.now(self.JST)
    now_label = now.strftime('%Y%m%d_%H%M%S')
    self.log_path =  self.base_path + f'results/log/{now_label}.tsv'
    os.makedirs(self.base_path + "results/log", exist_ok=True)
    os.makedirs(self.user_data_path, exist_ok=True)
    os.makedirs(self.results_path, exist_ok=True)
    os.makedirs(self.results_path, exist_ok=True)

    if not (os.path.isfile(self.log_path)):
      with open(self.log_path, 'w') as f:
        f.write("\t".join([
            "Time",
            "Action",
            "Counter",
            "Sum", 
            "Seed",
            "Width",
            "Height",
            "Dir",
            "Filename",
            "Base",
            "Mask",
            "Strength",
            "Prompt",
        ]) + "\n")
      
  def log(self, action, i, sum, seed, width, height, base, mask, strength, prompt, path):
    now = datetime.datetime.now(self.JST)
    filename = path.rsplit('/', 1)[1]
    dir = path.rsplit('/', 1)[0]
    with open(self.log_path, 'a') as f:
      f.write("\t".join([
          now.strftime('%Y%m%d_%H%M%S'),
          action,
          i, 
          sum,
          seed,
          width,
          height,
          dir,
          filename,
          base,
          mask,
          strength,
          prompt,
      ]) + "\n")

  def getResultsBase(self, dir):
    # 結果ディレクトリの用意
    dir_path = self.results_path + "/" + dir
    os.makedirs(dir_path, exist_ok=True)

    # 結果ファイル名の用意

    now = datetime.datetime.now(self.JST)
    return dir_path +  "/" + now.strftime('%Y%m%d_%H%M%S')

  def text2img(
      self,
      dir,
      count,
      width,
      height,
      prompt,
      num_inference_steps: int = 50,
      guidance_scale: float = 7.5,
      eta: float = 0.0,
      negative_prompt = None,
      ):
    self.setUpPipe("text2img")
    base_name = self.getResultsBase(dir)
    print(" start text2img(%s,%s)" % (dir, prompt))
    
    self.results = []
    for i in range(count):
      print(" create img2img(%d/%d)" % (i, count))
      seed_everything(self.seed)
      image = self.pipe(
          prompt,
          width=width,
          height=height,
          num_inference_steps = num_inference_steps,
          guidance_scale = guidance_scale,
          negative_prompt = negative_prompt,
          eta= eta,
          ).images[0]
      path = base_name + "_" + str(self.seed) + "_"  + str(i) + f".png"
      image.save(path)
      self.results.append(path)
      self.log(
          "text2img",
            str(i),
            str(count),
            str(self.seed),
          str(width),
          str(height),
          "",
          "",
          "",
          prompt,
          path
          )
      self.seed = self.seed + 1
    return self.results

  def img2img(
      self,
      dir,
      count,
      strength,
      width,
      height,
      image_name,
      prompt,
      num_inference_steps = 50,
      guidance_scale = 7.5,
      negative_prompt = None,
      eta = 0.0,
      ): 
    self.setUpPipe("img2img")

    base_name = self.getResultsBase(dir)
    image_path = self.user_data_path + "/" + image_name

    init_image = Image.open(image_path).convert("RGB").resize((width, height))

    print(" start img2img(%s,%s,%s)" % (dir, image_name, prompt))
    self.results = []

    for i in range(count):
      with torch.autocast("cuda"):
        print(" create img2img(%d/%d)" % (i, count))
        seed_everything(self.seed)

        image = self.pipe(
            prompt,
            init_image=init_image,
            strength=strength,
            num_inference_steps = num_inference_steps,
            guidance_scale = guidance_scale,
            negative_prompt = negative_prompt,
            eta = eta,
            ).images[0]
        path = base_name + "_" + str(self.seed) + "_" + str(i) + "_" + str(strength) + f".png"
        image.save(path)
        self.results.append(path)
        self.log(
            "img2img",
            str(i),
            str(count),
            str(self.seed),
            str(width),
            str(height),
            image_name,
            "",
            str(strength),
            prompt,
            path
            )
        self.seed = self.seed + 1
    return self.results

  def inpainting(
      self,
      dir,
      count,
      strength,
      width,
      height,
      init_name,
      mask_name,
      prompt,
      num_inference_steps = 50,
      guidance_scale= 7.5,
      negative_prompt= None,
      eta = 0.0,
      ): 
    self.setUpPipe("inpainting")
    base_name = self.getResultsBase(dir)

    init_path = self.user_data_path + "/" + init_name
    init_image = Image.open(init_path).convert("RGB").resize((width, height))

    mask_path = self.user_data_path + "/" + mask_name
    mask_image = Image.open(mask_path).convert("RGB").resize((width, height))

    self.results = []
    for i in range(count):
      with torch.autocast("cuda"):
        print(" create inpainting(%d/%d)" % (i, count))
        seed_everything(self.seed)
        image = self.pipe(
            prompt,
            init_image=init_image,
            mask_image=mask_image,
            strength=strength,
            num_inference_steps = num_inference_steps,
            guidance_scale= guidance_scale,
            negative_prompt= negative_prompt,
            eta = eta,
            ).images[0]
        path = base_name + "_" + str(self.seed) + "_"  + str(i) + "_" + str(strength) + f".png"
        image.save(path)
        self.results.append(path)
        self.log(
            "img2img",
            str(i),
            str(count),
            str(self.seed),
            str(width),
            str(height),
            init_name,
            mask_name,
            str(strength),
            prompt,
            path
            )
        self.seed = self.seed + 1
    return self.results
  def text2video(self, dir, count, width, height, prompts , num_steps, make_video=False, fps=10, do_loop=False, scheduler="ddim"):
    self.setUpPipe("text2video")
    base_name = self.getResultsBase(dir)
    prompts_text = " -> ".join(prompts)
    print(" start text2video(%s,%s)" % (dir, prompts_text))
    self.vc.setPipe(self.pipe)

    self.results = []
    for i in range(count):
      print(" create text2video(%d/%d)" % (i, count))
      seed_everything(self.seed)
      seeds = []
      for j in prompts:
        seeds.append(self.seed)

      if count > 1:
        path = base_name + "/" + str(self.seed)
      self.vc.walk(
        prompts = prompts,
        seeds =  seeds,
        num_steps=num_steps,
        make_video=make_video,
        width=width,
        height=height,
        do_loop=do_loop,
        fps=fps,
        output_dir=base_name,
        name=str(self.seed),
        scheduler=scheduler,
      )
      self.log(
          "text2video",
            str(i),
            str(count),
            str(self.seed),
          str(width),
          str(height),
          "",
          "",
          "",
          prompts_text,
          base_name + "/" + str(self.seed)
          )
      self.seed = self.seed + 1

  def img2video(self, dir, count, width, height, prompts, image_names, strength , num_steps, make_video=False, fps=10, do_loop=False, scheduler="ddim"):
    self.setUpPipe("img2video")
    base_name = self.getResultsBase(dir)
    prompts_text = " -> ".join(prompts)
    print(" start img2video(%s,%s)" % (dir, prompts_text))
    self.vc.setPipe(self.pipe)


    init_images = []
    self.results = []
    for imane_name in image_names:
      image_path = self.user_data_path + "/" + imane_name
      init_image = Image.open(image_path).convert("RGB").resize((width, height))
      init_images.append(init_image)


    for i in range(count):
      print(" create img2video(%d/%d)" % (i, count))
      seed_everything(self.seed)
      seeds = []
      for j in prompts:
        seeds.append(self.seed)

      path = base_name + "/" + str(self.seed) + str(strength)
      self.vc.walkImg2Img(
        prompts = prompts,
        seeds =  seeds,
        num_steps=num_steps,
        make_video=make_video,
        init_images = init_images,
        strength = strength,
        do_loop=do_loop,
        fps=fps,
        output_dir=base_name,
        name=str(self.seed),
        less_vram=True,
        scheduler=scheduler,
      )
      self.log(
          "img2video",
            str(i),
            str(count),
            str(self.seed),
          str(width),
          str(height),
          "",
          "",
          "",
          prompts_text,
          base_name + "/" + str(self.seed)
          )
      self.seed = self.seed + 1

  def setModel(self, library, revision):
    self.model_loaded= False
    self.library  = library
    self.revision = revision
  def setUpPipe(self, type):
    if self.model_loaded:
      if type == self.pipe_type:
        return
    if type == "text2img":
      print("setup text2img")
      if self.text_encoder is None:
        self.pipe = StableDiffusionPipeline.from_pretrained(
          self.library,
          revision=self.revision,
          use_auth_token=self.hugging_face_token,
          )
      else:
        self.pipe = StableDiffusionPipeline.from_pretrained(
          self.library,
          revision=self.revision,
          text_encoder=self.text_encoder,
          tokenizer=self.tokenizer,
          use_auth_token=self.hugging_face_token,
          )
      self.pipe.to("cuda")
    
    if type == "img2img":
      print("setup img2img")
      if self.text_encoder is None:
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
          self.library,
          revision=self.revision,
          torch_dtype=torch.float16,
          use_auth_token=self.hugging_face_token,
          )
      else:
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
          self.library,
          revision=self.revision,
          torch_dtype=torch.float16,
          text_encoder=self.text_encoder,
          tokenizer=self.tokenizer,
          use_auth_token=self.hugging_face_token,
          )
      self.pipe.to("cuda")
    
    if type == "inpainting":
      print("setup inpainting")
      if self.text_encoder is None:
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
          self.library,
          revision=self.revision,
          torch_dtype=torch.float16,
          use_auth_token=self.hugging_face_token)
      else:
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
          self.library,
          revision=self.revision,
          torch_dtype=torch.float16,
          text_encoder=self.text_encoder,
          tokenizer=self.tokenizer,
          use_auth_token=self.hugging_face_token)
      self.pipe.to("cuda")

    if type == "text2video":
      print("setup videos")
      if self.text_encoder is None:
        self.pipe = StableDiffusionWalkPipeline.from_pretrained(
          self.library,
          revision=self.revision,
          use_auth_token=self.hugging_face_token,
          )
      else:
        self.pipe = StableDiffusionWalkPipeline.from_pretrained(
          self.library,
          revision=self.revision,
          text_encoder=self.text_encoder,
          tokenizer=self.tokenizer,
          use_auth_token=self.hugging_face_token,
          )
      self.pipe.to("cuda")
    
    if type == "img2video":
      print("setup videos")
      if self.text_encoder is None:
        self.pipe = StableDiffusionWalkImg2ImgPipeline.from_pretrained(
          self.library,
          revision=self.revision,
          use_auth_token=self.hugging_face_token,
          )
      else:
        self.pipe = StableDiffusionWalkImg2ImgPipeline.from_pretrained(
          self.library,
          revision=self.revision,
          text_encoder=self.text_encoder,
          tokenizer=self.tokenizer,
          use_auth_token=self.hugging_face_token,
          )
        
      self.pipe.to("cuda")

    self.pipe_type = type

    if self.pipe is None:
      print ("初期化に失敗しました！")
      sys.exit()

    if self.nsfw_ok:
      def dummy(images, **kwargs): return images, False 
      self.pipe.safety_checker = dummy
    if self.less_memory:
      self.pipe.enable_attention_slicing()
  def showResults(self, cols, saveGrid = True):
    imgs = []
    for path in self.results:
      imgs.append(Image.open(path))
    cols = min(cols, len(imgs))
    rows = math.ceil(len(imgs) / cols)
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
      grid.paste(img, box=(i%cols*w, i//cols*h))
    if saveGrid:
      now = datetime.datetime.now(self.JST)
      now_label = now.strftime('%Y%m%d_%H%M%S')
      os.makedirs(f"{self.results_path}/grid", exist_ok=True)
      grid.save(f"{self.results_path}/grid/{now_label}.png")
    return grid