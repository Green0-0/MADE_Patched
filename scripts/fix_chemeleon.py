import os
import sys
import shutil
from pathlib import Path

HF_REPO = "G-reen/chemeleon_reupload"
LOCAL_CKPTS_DIR = Path("./ckpts").resolve()
GLOBAL_CACHE_DIR = Path.home() / ".cache" / "chemeleon_dng"

def purge_files():
    for directory in [LOCAL_CKPTS_DIR, GLOBAL_CACHE_DIR]:
        if directory.exists():
            print(f"🗑️ Wiping clean: {directory}")
            shutil.rmtree(directory, ignore_errors=True)

def download_straight_to_local() -> Path:
    """Download directly into ./ckpts without HF cache symlinks."""
    LOCAL_CKPTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"⬇️ Downloading fresh checkpoint directly to: {LOCAL_CKPTS_DIR}")
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ huggingface_hub is required. Run: uv pip install huggingface_hub")
        sys.exit(1)
        
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=str(LOCAL_CKPTS_DIR),
        local_dir_use_symlinks=False
    )
    
    exts = ("*.ckpt", "*.pt", "*.pth")
    matches = [f for ext in exts for f in LOCAL_CKPTS_DIR.rglob(ext) if f.is_file()]
    
    if not matches:
        raise RuntimeError("Download completed, but no checkpoint files were found!")
    
    return max(matches, key=lambda p: p.stat().st_size)

def main():
    print("Starting fresh Chemeleon environment setup...")
    
    try:
        purge_files()
        
        ckpt_path = download_straight_to_local()
        
        print("\n--- Running Health Check ---")
        print(f"Found checkpoint: {ckpt_path}")
        
        import chemeleon_dng
        sys.modules.setdefault("chemeleon_rl", chemeleon_dng)
        from chemeleon_dng.diffusion.diffusion_module import DiffusionModule
        
        print("Loading model weights (this may take a moment)...")
        dm = DiffusionModule.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=False)
        print("✅ Chemeleon checkpoint load OK! Model is healthy and in the correct directory.")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()