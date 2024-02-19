import os

open_mmlab_repos = [
    "mmcv",
    "mmyolo",
    "mmdetection",
    "mmdetection3d",
    "mmengine",
    "mmocr",
    "mmsegmentation",
    "mmhuman3d",
    "mmrotate",
    "mmediting",
]
for repo in open_mmlab_repos:
    os.system(f"git clone https://gitee.com/open-mmlab/{repo}.git")
os.system('python construct.py')
os.system('streamlit run app.py --server.address=0.0.0.0 --server.port 7860')