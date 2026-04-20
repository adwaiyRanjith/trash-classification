import modal
import os
import subprocess
import shutil

# same volume that has your dataset and best_model.pth
volume = modal.Volume.from_name("trash-dataset")

image = (
    modal.Image.debian_slim()
    .apt_install("unzip")
    .pip_install("torch", "torchvision", "python-dotenv")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("config.py", remote_path="/root/config.py")
)

app = modal.App("trash-evaluator")

@app.function(
    gpu="A10G",
    image=image,
    volumes={"/data": volume},
    timeout=600
)
def evaluate():
    os.chdir("/root")

    print("Unzipping data")
    os.system("unzip -q /data/cleaned_trash_data.zip -d /tmp/")
    print("Done unzipping")

    os.makedirs("checkpoints", exist_ok=True)
    shutil.copy("/data/best_model.pth", "checkpoints/best_model.pth")
    print("Loaded best model weights!")

    os.environ["DATA_DIR"] = "/tmp/cleaned_trash_data/original"
    subprocess.run(["python", "src/evaluate.py"], capture_output=False)

@app.local_entrypoint()
def main():
    evaluate.remote()