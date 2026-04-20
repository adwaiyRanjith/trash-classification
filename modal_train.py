import modal
import os

volume = modal.Volume.from_name("trash-dataset")

image = (
    modal.Image.debian_slim()
    .apt_install("unzip")
    .pip_install("torch", "torchvision", "python-dotenv")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file("config.py", remote_path="/root/config.py")
)

app = modal.App("trash-classifier")

@app.function(
    gpu="A10G",
    image=image,
    volumes={"/data": volume},
    timeout=3600
)
def train():
    import subprocess
    os.chdir("/root")
    
    print("Unzipping data")
    os.system("unzip -q /data/cleaned_trash_data.zip -d /tmp/")
    os.system("ls /tmp/")
    os.system("ls /tmp/cleaned_trash_data/")
    os.system("find /tmp -type d -maxdepth 3")
    print("Done unzipping")
    os.environ["DATA_DIR"] = "/tmp/cleaned_trash_data/original"
    subprocess.run(["python", "src/train.py"], capture_output=False)

@app.local_entrypoint()
def main():
    train.remote()