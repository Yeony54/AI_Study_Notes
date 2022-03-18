# Titanic

단계 1 : Data Set을 사용해 모델을 분석하고, 주석으로 설명해보자

## 참고

kaggle : https://www.kaggle.com/c/titanic/overview

dacon : https://dacon.io/competitions/open/235539/codeshare

누구나 파이썬 머신러닝 완벽 가이드 2장 6번 131페이지 예제



data download : https://www.kaggle.com/c/titanic/data



---



kaggle api 사용법 https://velog.io/@skyepodium/Kaggle-API-%EC%82%AC%EC%9A%A9%EB%B2%95

- kaggle 데이터파일 다운로드, 제출 가능





---





```
 1) 데이터 분석 (1.4 판다스 p39~)
 - 데이터프레임 출력 확인하기
 - 데이터 분석 관련 표 3개 이상 출력해서 데이터 확인해보기
 - 학습 데이터, 테스트 데이터 분할 (머신러닝 모델에 넣기 위한 준비)

 2) 모델 사용
 - 사이킷런 모델을 사용해서 Accuracy 구하기. 적어도 2개 이상 모델 사용하기

 3) 결론 및 질문
 - 모델을 사용해보고서 느낀점이나 궁금증. 혹은 분석 결과 정리하여 문서화 완료.
```



- [이보연님](https://github.com/minsuk-heo/kaggle-titanic/blob/master/titanic-solution.ipynb)

  [reference](https://github.com/minsuk-heo/kaggle-titanic/blob/master/titanic-solution.ipynb)

  - Mr, Mrs 칭호별로 분류
  - 클래스 별로 나이 중위값으로 채워줌
  - 상관계수 : 
    - 0.3, 0.5만 넘겨도 유의하다
    - 1에가깝다...?
  - 사이킷런 교차검증 cross_Val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
    - clf : 머신러닝 넣은 객체 객체넣는거
    - train_data : train 할 데이터
    - target_data : 정답 데이터
    - cv : 검증 횟수
    - n_jobs : 리소스 가져오는 개념,,?, cpu 코어
    - scoring : accuracy 정확도

- [김인후님](https://github.com/InhuKim/study/blob/main/ML%20example/titanic_prediction/Titanic.ipynb)

  - SVM 100% 성능

- [이희경님](https://github.com/JellyJoa/kaggle/blob/master/MLexample/Titanic%20example/Titanic.ipynb)

  - 여러가지 Model 시도

- [김창현님1](https://github.com/kch8906/Study/blob/master/07.kaggle/1.%20Binary%20classification%20-%20Tabular%20data/1st%20Titanic.%20Machine%20Learning%20from%20Disaster/20220313%20-%20%ED%83%80%EC%9D%B4%ED%83%80%EB%8B%891.ipynb)  [김창현님2](https://github.com/kch8906/Study/blob/master/07.kaggle/1.%20Binary%20classification%20-%20Tabular%20data/1st%20Titanic.%20Machine%20Learning%20from%20Disaster/20220315-%20%ED%83%80%EC%9D%B4%ED%83%80%EB%8B%892.ipynb)

  - missingno : 결측치 시각화 모듈
  - warning 무시 
  - Outline 없애려고 log를 시켰음
  - 원핫인코딩아니고 딕셔너리..?
  - 그래프 예쁜거많당
  - 사이킷런 method보다 pandas꺼로 하는게 편하지 않을까...라고하시네
  - 검증데이터 val로 뽑아서,,, ?
  - 피쳐 중요도

- [이보연님](https://github.com/leebydev/kaggle_study/blob/main/titanic/code/titanic.ipynb)
- jupyter 자동완성 탭
- jupyter 함수 상세정보 shift+tab
- 확장프로그램 번역
- 
