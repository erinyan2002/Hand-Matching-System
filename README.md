# 🖐️  Hand Matching System - 손 사진 비교로 동일 인물 판별하기 (SuperPoint + SuperGlue)



![compareHand](https://github.com/user-attachments/assets/7ba9feea-84e6-49d6-b8df-70f99c5f0a64)



> **"이 손이 누구 손일까?"**  
> 두 장의 손 사진만 보고 같은 사람인지 아닌지를 맞힐 수 있을까요?



##  프로젝트 개요

   이 프로젝트는 SuperPoint와 SuperGlue 모델을 이용해 **손 이미지 간의 특징점 매칭을 통해 동일 인물 여부를 자동으로 판별**합니다.

   사람 눈 대신 **AI 모델이 손의 특징점을 찾아서**,  서로 얼마나 잘 맞는지 확인하고,  **같은 사람인지 아닌지를 자동으로 판단**해줍니다.

   또한 단순한 2장 비교를 넘어서,  **여러 손 사진을 동시에 비교하여 자동으로 클러스터링(그룹화)** 하는 기능까지 포함되어 있습니다.

---


   사용된 모델은 우리가 수업 시간에 배운 **SuperPoint**와 **SuperGlue**!  
두 모델을 직접 GitHub에서 가져와 코드로 적용하고 시각화까지 해봤습니다..

| 모델 이름 | 설명 |
|-----------|------|
|  SuperPoint | 손 이미지에서 특징점과 descriptor 추출 |
|  SuperGlue | 두 이미지 간 descriptor를 GNN 기반으로 정교하게 연결 |


---

##  폴더 구조 (중요한 것만)

1. assets/
   └── hand/ ← 2장 비교용 손 사진
   └── hand1/ ← 여러 장 비교용 폴더 (클러스터링 전용)
2. models/ ← 모델 코드 및 가중치
3. result/ ← 시각화 결과 이미지 저장 위치
4. hand_match.py ← 2장 비교 (기본 기능)
5. pairwise_match_all.py ← 모든 이미지 쌍 비교
6. cluster_match_results.py ← 클러스터링 기능



##  개발 환경

- Python 3.12
- PyTorch 2.x
- OpenCV 4.x
- 실행 환경: Windows 11 (VS Code + PowerShell)

> 필요한 패키지는 `requirements.txt`에 모두 포함되어 있습니다.


---


#  기능 1: 손 사진 2장 비교


### 사용 방법

1. `assets/hand/` 폴더에 비교할 손 사진 2장을 넣습니다.
2. `hand_match.py` 안에서 파일명을 설정합니다.

```
img0 = load_image_gray("assets/hand/4.jpg")
img1 = load_image_gray("assets/hand/2.jpg")
```
3. 실행:
```
python hand_match.py
```




##  출력 결과 예시(사진)


- ✅ 동일 인물의 손일 가능성이 높은 결과
  

![image](https://github.com/user-attachments/assets/45624d8c-1e37-4f4b-8778-8807f9c6b6fc)



- ❌ 다른 사람의 손일 가능성이 높은 결과


![image](https://github.com/user-attachments/assets/5b4292b3-041e-4ea1-bf01-34883d34313c)




---

##  결과 해석

- 실행하면 `result_match.png` 파일이 생성되며, 두 손 사진이 나란히 표시됩니다.
- 사진 사이에 초록색/주황색 선들이 그려지는데, 이 선들이 **서로 연결된 특징점**을 의미합니다.

| 색상 | 의미 |
|------|------|
| 초록색 선 | 강한 매칭 (정확하게 맞는 점) |
| 주황색 선 | 약한 매칭 (덜 확실한 점) |

- 선이 많고 손가락 위치에 고르게 분포되어 있다면 → **같은 사람일 가능성 높음**
- 선이 적거나 이상한 방향으로 흩어져 있다면 → **다른 사람일 가능성 높음**

##  간단한 판단 기준

| 매칭된 점 수 | 해석 |
|--------------|------|
| 30개 이상     | ✅ 동일 인물일 가능성 높음 |
| 30개 미만     | ❌ 다른 사람일 가능성 높음 |

---

#  기능 2: 여러 손 이미지 자동 비교 및 클러스터링

cluster_match_results.py는 폴더 내 모든 손 이미지들을 자동으로 서로 비교하고, 같은 사람의 손으로 판단된 이미지들을 클러스터링합니다.

## 실행 방법

```
python cluster_match_results.py
```

## 출력 예시(터미널 결과)


![image](https://github.com/user-attachments/assets/e7dc54d3-46cf-49a9-b7b3-5a7fbc8cf8c1)


## 클러스터링 기술 설명


1. 매칭 수 기준 >>>	100개 이상 → "같은 사람"으로 판단 (조정 가능)
2. 거리 계산 방식	>>> distance = 1 / (match_count + ε)
3. 클러스터링 알고리즘 >>> 	DBSCAN (밀도 기반 클러스터링)
4. 노이즈 분리 >>>	유사한 그룹에 속하지 않는 이미지는 자동으로 제외

---


# 기능 3:  모든 손 사진 쌍 자동 비교 (Pairwise Matching)

`pairwise_match_all.py`는 `assets/hand/` 폴더에 있는 모든 손 이미지를 **쌍으로 묶어 전부 비교**합니다.

각 조합마다 SuperPoint와 SuperGlue를 이용하여 특징점을 매칭하고,  
**매칭 수에 따라 동일 인물 여부를 자동으로 판단**하며,  
결과는 시각화 이미지로 저장됩니다.

### 📂 사용 예시

```
python pairwise_match_all.py
```

##출력 예시

```
🔍 대상 이미지 수: 10장
1.jpg ↔ 2.jpg : 0점 → DIFF
1.jpg ↔ 3.jpg : 0점 → DIFF
...
2.jpg ↔ 4.jpg : 164점 → SAME
5.jpg ↔ 6.jpg : 164점 → SAME
...
✅ 전체 매칭 완료! 결과는 results/ 폴더에 저장되었습니다.
```
비교 결과는 results/ 폴더에 자동 저장되며, 파일명에는 비교 쌍, 판별 결과, 매칭 수가 포함됩니다:

## result file 결과 예시



![image](https://github.com/user-attachments/assets/7ca5fe2c-c5da-4635-be6c-6ec81df6a1b7)


## 유용한점

- 이미지를 일일이 비교하지 않아도, 전체 자동 매칭 결과를 한 번에 확인 가능

- 매칭 수 기준으로 동일 인물 여부를 자동 판단해줌

- 결과 이미지 저장으로 시각적 분석 가능

- 데이터셋 전처리 및 그룹 분류에 활용 가능


---


##  프로젝트 배경


이 프로젝트는 **컴퓨터비전 수업의 기말 과제**로 진행했습니다.  
교수님 강의 자료에서 소개된 SuperPoint와 SuperGlue 모델을 직접 fork하여  
실제 손 사진 비교 문제에 적용해보며, 딥러닝 기반 매칭 알고리즘의 흐름을 체험했습니다.

---

## 📚 참고 자료

- SuperPoint 논문: https://arxiv.org/abs/1712.07629
- SuperGlue 논문: https://arxiv.org/abs/1911.11763
- 원본 코드 저장소: https://github.com/magicleap/SuperGluePretrainedNetwork
- 내가 fork한 저장소: https://github.com/erinyan2002/SuperGluePretrainedNetwork

---


##  확장 가능성

-  얼굴, 귀, 발 등 다른 신체 부위에도 동일 방식으로 적용 가능
-  웹캠과 연결해 **실시간 인증 시스템**으로 확장 가능
-  matching 결과를 feature로 활용하여 **손 인증 classifier 훈련** 가능
-  

---


##  결론

이 프로젝트는 단순한 손 사진 비교를 넘어서 딥러닝 기반의 이미지 매칭 + 자동 클러스터링 까지 경험할 수 있는 실습형 과제입니다.

💡 실습을 통해 다음을 체험할 수 있었습니다:

- 이미지 특징점 추출 및 정렬

- GNN 기반 매칭 모델 적용

- 실시간 시각화 및 분류 판단

- 유사도 기반 클러스터링 기술

---

## 👤 프로젝트 정보

- 이름: 이먓판
- 학번: 23101201  
- 소속: 컴퓨터공학과  
- 과제명: 컴퓨터비전 기말 프로젝트










