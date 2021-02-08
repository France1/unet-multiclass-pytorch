from pathlib import Path
import shutil

def make_checkpoint_dir(dir_checkpoint):
        
    path = Path(dir_checkpoint)
    # remove folder if it exists
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)