import os, glob, shutil

dirs = ["/nobackup/joon/1_Projects/L3C-PyTorch-myversion-runs/data/train_oi/*.png", "/nobackup/joon/1_Projects/L3C-PyTorch-myversion-runs/data/val_oi/*.png"]
tos = ["/nobackup/joon/1_Projects/L3C-PyTorch/data/train_oi/", "/nobackup/joon/1_Projects/L3C-PyTorch/data/val_oi/"]

for d, t in zip(dirs, tos):
    paths = glob.glob(d)[:160]
    print(len(paths))
    
    for p in paths:
        to_path = os.path.join(t, os.path.basename(p))
        shutil.copy(p, to_path)

