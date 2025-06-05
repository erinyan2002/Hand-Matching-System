import os
import glob
import cv2
import torch
import numpy as np
from itertools import combinations
from models.matching import Matching
from models.utils import frame2tensor, make_matching_plot_fast, error_colormap

# ======= 설정 =======
IMAGE_FOLDER = 'assets/hand'
RESULT_FOLDER = 'results'
MATCH_THRESHOLD = 100  # 이 값 이상이면 동일 인물로 간주
DEVICE = 'cpu'  # GPU를 사용할 수 있다면 'cuda'로 설정
RESIZE_SHAPE = (640, 480)

# ======= 초기 준비 =======
os.makedirs(RESULT_FOLDER, exist_ok=True)

config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    },
    'superglue': {
        'weights': 'indoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2
    }
}

matching = Matching(config).eval().to(DEVICE)

def load_image_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, RESIZE_SHAPE)
    return img

# ======= 이미지 불러오기 =======
image_paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, '*.*')))
image_names = [os.path.basename(p) for p in image_paths]
print(f"🔍 대상 이미지 수: {len(image_paths)}장")

# ======= 전체 쌍 비교 =======
for (idx_a, path_a), (idx_b, path_b) in combinations(enumerate(image_paths), 2):
    name_a = image_names[idx_a]
    name_b = image_names[idx_b]

    img0 = load_image_gray(path_a)
    img1 = load_image_gray(path_b)

    inp0 = frame2tensor(img0, DEVICE)
    inp1 = frame2tensor(img1, DEVICE)

    with torch.no_grad():
        pred = matching({'image0': inp0, 'image1': inp1})

    kpts0 = pred['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    valid = matches > -1

    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    scores = pred['matching_scores0'][0].detach().cpu().numpy()[valid]
    color = error_colormap(scores)[:, :3]

    match_count = np.sum(valid)
    is_same = match_count >= MATCH_THRESHOLD
    label = "SAME" if is_same else "DIFF"
    print(f"{name_a} ↔ {name_b} : {match_count}점 → {label}")

    # 이미지 저장
    result_path = os.path.join(RESULT_FOLDER, f"{name_a}_vs_{name_b}_{label}_{match_count}.png")
    make_matching_plot_fast(
        img0, img1,
        kpts0, kpts1,
        mkpts0, mkpts1,
        color,
        text=[f"{name_a} vs {name_b}", f"{match_count} matches", f"{label}"],
        path=result_path,
        show_keypoints=False
    )

print("✅ 전체 매칭 완료! 결과는 results/ 폴더에 저장되었습니다.")
