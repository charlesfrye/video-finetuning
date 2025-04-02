import os
import subprocess
import time

import modal

wan_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "ffmpeg", "libsm6", "libxext6")
    .run_commands(
        "git clone https://github.com/ostris/ai-toolkit.git /root/ai-toolkit",
        "cd /root/ai-toolkit && git submodule update --init --recursive",
    )
    .pip_install("torch", "huggingface_hub[hf_transfer]==0.26.2")
    .run_commands("cd /root/ai-toolkit && pip install -r requirements.txt")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

jupyter_image = (
    wan_image.pip_install("jupyter")
    .add_local_file("notebooks/training.ipynb", "/root/training.ipynb")
    .add_local_file(
        "config/train_lora_wan21_1b.yaml", "/root/ai-toolkit/config/train_cfg.yaml"
    )
    .add_local_dir("data", "/root/ai-toolkit/data")
)

app = modal.App("finetune-video-train", image=jupyter_image)

data_volume = modal.Volume.from_name("finetune-video-data", create_if_missing=True)
finetunes_volume = modal.Volume.from_name(
    "finetune-video-models", create_if_missing=True
)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


@app.function(
    max_containers=1,
    volumes={
        "/root/remote-data": data_volume,
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/outputs": finetunes_volume,
    },
    timeout=15_000,
    gpu="h100!",
)
def run_jupyter(timeout: int):
    jupyter_port = 8888
    with modal.forward(jupyter_port) as tunnel:
        jupyter_process = subprocess.Popen(
            [
                "jupyter",
                "notebook",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={jupyter_port}",
                "--NotebookApp.allow_origin='*'",
                "--NotebookApp.allow_remote_access=1",
                "--NotebookApp.token=''",
                "--NotebookApp.password=''",
            ],
            env=os.environ,
        )

        print(f"Jupyter available at => {tunnel.url}")

        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                time.sleep(5)
            print(f"Reached end of {timeout} second timeout period. Exiting...")
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            jupyter_process.kill()


@app.local_entrypoint()
def main(timeout: int = 15_000):
    run_jupyter.remote(timeout=timeout)
