from pathlib import Path

import modal


diffusers_commit_sha = "df1d7b01f18795a2d81eb1fd3f5d220db58cfae6"
MINUTES = 60  # seconds


inference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "ffmpeg", "libsm6", "libxext6")
    .pip_install(
        "accelerate==1.5.1",
        "ftfy==6.3.1",
        f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
        "huggingface_hub[hf_transfer]==0.30.1",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "peft==0.14.0",
        "torch==2.5.1",
        "transformers==4.49.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
)

finetunes_vol = modal.Volume.from_name("finetune-video-models", create_if_missing=False)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
outputs_vol = modal.Volume.from_name("finetune-video-outputs", create_if_missing=True)

MODELS_DIR = Path("/root/models")
OUTPUTS_DIR = Path("/root/outputs")

app = modal.App("finetune-video-generate", image=inference_image)


@app.cls(
    gpu="h100",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        MODELS_DIR: finetunes_vol,
        OUTPUTS_DIR: outputs_vol,
    },
    timeout=30 * MINUTES,
    scaledown_window=5 * MINUTES,
)
class VideoGenerator:
    finetune_id: str = modal.parameter()

    @modal.enter()
    def init(self):
        self.config = load_config(MODELS_DIR / self.finetune_id / "config.yaml")

        # determine the base model used in training and load it
        self.base_model = self.config["model"]["name_or_path"]
        self.pipe = load_model(self.base_model)

        # prepare and load the fine-tuned adapter weights
        prep_lora_weights(self.finetune_id)
        self.pipe.load_lora_weights("diffusers_lora.safetensors")

        # determine the trigger word used in training
        self.trigger_word = self.config["trigger_word"]
        # determine sampling parameters used during training
        self.guidance_scale = self.config["sample"]["guidance_scale"]
        self.height = self.config["sample"]["height"]
        self.width = self.config["sample"]["width"]

    @modal.method()
    def run(
        self,
        prompt="[trigger] holding a sign that says 'I LOVE MODAL'",
        guidance_scale=None,
        num_frames=45,
    ) -> bytes:
        from diffusers.utils import export_to_video

        prompt = prompt.replace("[trigger]", self.trigger_word)
        guidance_scale = (
            self.guidance_scale if guidance_scale is None else guidance_scale
        )

        output = self.pipe(
            prompt=prompt,
            height=self.height,
            width=self.width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
        ).frames[0]

        output_dir = OUTPUTS_DIR / self.finetune_id
        output_dir.mkdir(exist_ok=True)

        output_path = Path(
            export_to_video(output, output_dir / (slugify(prompt) + ".mp4"), fps=15)
        )
        print(f"saved remotely at {output_path}")

        return output_path.read_bytes()


@app.local_entrypoint()
def test(
    prompt="[trigger] holding a sign that says 'I LOVE MODAL'",
    finetune_id="411651ccc88d49bff399bd579445dd62",
    num_frames: int = 1,
    guidance_scale: float = None,
):
    generator = VideoGenerator(finetune_id=finetune_id)

    result = generator.run.remote(
        prompt, num_frames=num_frames, guidance_scale=guidance_scale
    )

    output_path = Path("/tmp") / (slugify(prompt) + ".mp4")
    output_path.write_bytes(result)
    print(f"output saved locally at {output_path}")


def load_model(base_model, to_cuda=True):
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline

    vae = AutoencoderKLWan.from_pretrained(
        base_model, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanPipeline.from_pretrained(base_model, vae=vae, torch_dtype=torch.bfloat16)
    if to_cuda:
        pipe.to("cuda")
    return pipe


def load_and_convert(safetensors_file):
    from safetensors import safe_open

    f = safe_open(safetensors_file, framework="pt", device=0)
    return convert_to_diffusers({key: f.get_tensor(key) for key in f.keys()})


def save(state_dict, safetensors_file):
    from safetensors.torch import save_file

    save_file(state_dict, safetensors_file)


def load_config(yaml_path):
    import yaml

    with open(yaml_path) as f:
        return yaml.safe_load(f)["config"]["process"][0]


def prep_lora_weights(finetune_id):
    save(
        load_and_convert(MODELS_DIR / f"{finetune_id}/{finetune_id}.safetensors"),
        "diffusers_lora.safetensors",
    )


def convert_to_diffusers(state_dict):
    # copied from ostris/ai-toolkit
    new_state_dict = {}
    for key in state_dict:
        new_key = key
        # Base model name change
        if key.startswith("diffusion_model."):
            new_key = key.replace("diffusion_model.", "transformer.")

        # Attention blocks conversion
        if "self_attn" in new_key:
            new_key = new_key.replace("self_attn", "attn1")
        elif "cross_attn" in new_key:
            new_key = new_key.replace("cross_attn", "attn2")

        # Attention components conversion
        parts = new_key.split(".")
        for i, part in enumerate(parts):
            if part in ["q", "k", "v"]:
                parts[i] = f"to_{part}"
            elif part == "o":
                parts[i] = "to_out.0"
        new_key = ".".join(parts)

        # FFN conversion
        if "ffn.0" in new_key:
            new_key = new_key.replace("ffn.0", "ffn.net.0.proj")
        elif "ffn.2" in new_key:
            new_key = new_key.replace("ffn.2", "ffn.net.2")

        new_state_dict[new_key] = state_dict[key]
    return new_state_dict


def slugify(s: str, lim=100) -> str:
    return "-".join(c if c.isalnum() else "-" for c in s[:lim].split(" ")).strip("-")
