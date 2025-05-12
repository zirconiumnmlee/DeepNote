from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-3B-Instruct",
    force_download=True,
    max_workers=1,
    resume_download=True,
    local_dir_use_symlinks=False
    )
