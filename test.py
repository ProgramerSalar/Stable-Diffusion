from huggingface_hub import hf_hub_download




if __name__ == "__main__":
  # Download the checkpoint
  checkpoint_path = hf_hub_download(
    repo_id="CompVis/stable-diffusion-v-1-4-original",
    filename="sd-v1-4.ckpt",
    cache_dir="path/to/save/checkpoint"
  )
  print("Checkpoint downloaded to:", checkpoint_path)

  