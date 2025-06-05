import os
import glob
import cv2
import torch
import numpy as np
from itertools import combinations
from collections import defaultdict
from sklearn.cluster import DBSCAN
from models.matching import Matching
from models.utils import frame2tensor, make_matching_plot_fast, error_colormap

# ======= 설정 =======
IMAGE_FOLDER = 'assets/hand3'
RESULT_FOLDER = 'results'
MATCH_THRESHOLD = 100
DEVICE = 'cpu'
RESIZE_SHAPE = (640, 480)
CLUSTER_EPS = 0.05  # 거리 기준값
CLUSTER_MIN_SAMPLES = 2

# ======= 초기화 =======
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
num_images = len(image_paths)
print(f"🔍 처리 대상 이미지 수: {num_images}장")

# ======= 거리 행렬 및 매칭 로그 초기화 =======
distance_matrix = np.ones((num_images, num_images)) * np.inf
np.fill_diagonal(distance_matrix, 0)

# ======= 전체 조합 비교 =======
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

    # 결과 이미지 저장
    result_path = os.path.join(RESULT_FOLDER, f"{name_a}_vs_{name_b}_{label}_{match_count}.png")
    make_matching_plot_fast(
        img0, img1, kpts0, kpts1, mkpts0, mkpts1, color,
        text=[f"{name_a} vs {name_b}", f"{match_count} matches", f"{label}"],
        path=result_path,
        show_keypoints=False
    )

    # 유사도를 거리로 변환
    dist = 1.0 / (match_count + 1e-5)
    distance_matrix[idx_a, idx_b] = dist
    distance_matrix[idx_b, idx_a] = dist

# ======= 클러스터링 =======
clustering = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES, metric='precomputed')
labels = clustering.fit_predict(distance_matrix)

# ======= 클러스터 출력 =======
cluster_map = defaultdict(list)
for idx, label in enumerate(labels):
    cluster_map[label].append(image_names[idx])

print("\n✅ 클러스터링 결과:")
for label, images in cluster_map.items():
    label_text = f"클러스터 {label}" if label != -1 else "노이즈"
    print(f"\n🟢 {label_text}:")
    for img in images:
        print(f"  - {img}")

print("\n✅ 처리 완료! 매칭 결과 및 클러스터 정보를 출력했습니다.")
