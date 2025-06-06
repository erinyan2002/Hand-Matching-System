# 🖐️ 손 사진 비교로 동일 인물 판별하기 (SuperPoint + SuperGlue)


![Uploading compareHand.![Handspng](https://github.com/user-attachments/assets/0f70257d-b0dc-4bbf-a2d0-ef1db0388f7f)
gif…]()



> **"이 손이 누구 손일까?"**  
> 두 장의 손 사진만 보고 같은 사람인지 아닌지를 맞힐 수 있을까요?

이 프로젝트는 **딥러닝을 이용한 손 사진 비교기**입니다.  
사람 눈 대신 **AI 모델이 손의 특징점을 찾아서**,  
서로 얼마나 잘 맞는지 확인하고,  
**같은 사람인지 아닌지를 자동으로 판단**해줍니다.

🧠 사용된 모델은 우리가 수업 시간에 배운 **SuperPoint**와 **SuperGlue**!  
두 모델을 직접 GitHub에서 가져와 코드로 적용하고 시각화까지 해봤습니다..

| 모델 이름 | 설명 |
|-----------|------|
|  SuperPoint | 손 이미지에서 특징점과 descriptor 추출 |
|  SuperGlue | 두 이미지 간 descriptor를 GNN 기반으로 정교하게 연결 |

---

## 🚀 이런 분들에게 추천합니다

- 이미지 속에서 같은 인물인지 알고 싶은 사람
- 딥러닝 기반의 이미지 매칭이 궁금한 사람
- 수업 내용을 실습으로 체험하고 싶은 학생

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






