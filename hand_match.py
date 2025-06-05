import torch
import cv2
import numpy as np
from models.matching import Matching
from models.utils import frame2tensor, make_matching_plot_fast, error_colormap

# SuperPoint + SuperGlue 설정
config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    },
    'superglue': {
        'weights': 'indoor',  # 또는 'outdoor'
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2
    }
}

# 모델 초기화
device = 'cpu'  # GPU를 사용할 수 있다면 'cuda'로 변경
matching = Matching(config).eval().to(device)

# 손 이미지 불러오기 및 전처리
def load_image_gray(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (640, 480))  # 필요시 크기 통일
    return image

# 비교할 이미지 지정
img0 = load_image_gray("assets/hand/4.jpg")  # 설정 필요
img1 = load_image_gray("assets/hand/2.jpg")  # 설정 필요

# 텐서로 변환
inp0 = frame2tensor(img0, device)
inp1 = frame2tensor(img1, device)

# 특징점 매칭 실행
with torch.no_grad():
    pred = matching({'image0': inp0, 'image1': inp1})

# 특징점과 매칭 정보 가져오기
kpts0 = pred['keypoints0'][0].cpu().numpy()
kpts1 = pred['keypoints1'][0].cpu().numpy()
matches = pred['matches0'][0].cpu().numpy()
valid = matches > -1

# 유효한 매칭만 추출
mkpts0 = kpts0[valid]
mkpts1 = kpts1[matches[valid]]
scores = pred['matching_scores0'][0].detach().cpu().numpy()[valid]
color = error_colormap(scores)[:, :3]  # RGB 형식으로 변환

# 매칭된 수 표시
match_count = np.sum(valid)
print(f"매칭된 특징점 수: {match_count}")

# 동일 인물인지 여부 간이 판별
if match_count >= 30:
    print("✅ 동일 인물의 손일 가능성이 높음")
else:
    print("❌ 다른 사람의 손일 가능성이 높음")

# 시각화 및 저장
make_matching_plot_fast(
    img0, img1,
    kpts0, kpts1,         # 전체 키포인트
    mkpts0, mkpts1,       # 매칭된 키포인트
    color,
    text=[f'{match_count} matches'],
    path='result_match.png',
    show_keypoints=False,
    margin=10,
    opencv_display=False,
    opencv_title='Matching Result',
    small_text=[]
)

print("✅ result_match.png 에 매칭 결과가 저장되었습니다")
