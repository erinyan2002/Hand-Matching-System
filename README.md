##  손 사진 비교로 동일 인물 판별하기 (SuperPoint + SuperGlue)

두 장의 손 사진을 비교해서 **같은 사람인지 아닌지**를 판단하는 프로젝트입니다.  
딥러닝 모델인 SuperPoint와 SuperGlue를 이용해 **특징점을 자동으로 찾고**, **매칭 정도를 수치로 확인**합니다.

---

##  프로젝트 핵심 기능

- 손 사진 2장을 비교해 **특징점을 자동으로 추출하고 연결**합니다.
- 얼마나 많이 연결되었는지를 보고 **같은 사람의 손인지 판단**합니다.
- 결과는 이미지로 저장되며, **시각적으로 매칭 상태를 확인**할 수 있습니다.

---

### 사용된 모델

| 모델 이름 | 설명 |
|-----------|------|
|  SuperPoint | 손 이미지에서 특징점과 descriptor 추출 |
|  SuperGlue | 두 이미지 간 descriptor를 GNN 기반으로 정교하게 연결 |

---

##  폴더 구조 (중요한 것만)

1. assets/hand/ ← 손 사진 2장을 여기에 넣음
2. models/ ← SuperPoint, SuperGlue 코드 + 가중치
3. hand_match.py ← 메인 실행 코드
4. result_match.png ← 결과 이미지 (자동 생성)

---


##  실행 방법 

1. 손 사진을 `assets/hand/` 폴더에 넣기  
   예: `2.jpg`, `4.jpg`

2. `hand_match.py` 안에서 비교할 파일명 수정:

```
img0 = load_image_gray("assets/hand/4.jpg")
img1 = load_image_gray("assets/hand/2.jpg")
```

3. 실행
```
python hand_match.py
```

---

##  출력 결과 예시(사진)



![image](https://github.com/user-attachments/assets/45624d8c-1e37-4f4b-8778-8807f9c6b6fc)






