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



---



- Target label 확인

  Target label이 어떤 distribution을 가지고 있는지 확인해봐야한다.

  binary classification 문제의 경우에서 1과 0의 분포가 어떠하냐에 따라 모델의 평가 방법이 달라진다.

  ```python
  f, ax = plt.subplots(1, 2, figsize=(18, 8))
  
  df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1],
                                              autopct='%1.1f%%', ax=ax[0], shadow=True)
  ax[0].set_title('Pie plot - Survived')
  ax[0].set_ylabel('')
  sns.countplot('Survived', data=df_train, ax=ax[1])
  ax[1].set_title('Count plot - Survived')
  
  plt.show()
  ```

  

- crosstab : 교차표

  ```python
  pd.crosstab(train['Pclass'], train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
  ```

  행, 열 요인 기준 별로 빈도를 세어서 도수분포표(frequency table), 교차표(contingency table)를 만들어주는 기능



- seaborn factorplot, pointplot

  똑같은 그래프인데 factorplot은 figure-level이고, pointplot은 axes-level이다.

  

- Kernel Density Estimation (KDE) 커널밀도추정

  - https://darkpgmr.tistory.com/147

  - 밀도추정 (Density Estimation)

    얻어진(관측된) 데이터들의 분포로부터 원래 변수의 (확률) 분포 특성을 추정하고자 하는것

  - Parametric vs. Non-parametric 밀도추정

    Parametric : 정규분포등 pdf(probability density function)에 대한 모델을 정해놓고 데이터들로부터 모델의 파라미터만 추정하는 방식, 간단하다

    Non-parametric : 현실에서 모델이 미리 주어지는 경우는 많지 않으니 순수하게 관측된 데이터만으로 확률밀도함수를 추정하는것

  - histogram

    Non-parametric 밀도추정의 가장 간단한 형태가 히스토그램이다.

    즉, 관측된 데이터들로부터 히스토그램을 구한 후 구해진 히스토그램을 정규화하여 확률밀도함수로 사용하는 것이다.

  - Kernel Density Estimation (KDE) 커널밀도추정

    앞서 non-parametric의 밀도추정의 가장 단순한 형태가 히스토그램 방법이라고 했는데, 히스토그램 방법은 bin 의 경계에서 불연속성이 나타난다는점, bin의 크기 및 시작 위치에 따라서 히스토그램이 달라진다는점, 고차원(high dimension) 데이터에는 메모리 문제등으로 사용하기 힘들다는점 등의 문제를 갖는다.

    KDE 방법은  non-parametric 밀도추정 방법중 하나로서 커널함수를 이용하여 히스토그램 방법의 문제점을 개선한 방법이다.



- seaborn 

  - histplot
  - distplot

- 분석 tip

  - 비대칭데이터 문제

    - [DataScienceSchool](https://datascienceschool.net/03%20machine%20learning/14.02%20%EB%B9%84%EB%8C%80%EC%B9%AD%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EB%AC%B8%EC%A0%9C.html)

    - 각 클래스에 속한 데이터의 갯수의 차이에 의해 발생하는 문제

    - 데이터 클래스 비율이 너무 차이가 나면 (highly-imbalanced data) 단순히 우세한 클래스를 택하는 모형의 정확도가 높아지므로 모형의 성능판별이 어려워진다. 즉, 정확도(accuracy)가 높아도 데이터 갯수가 적은 클래스의 재현율(recall-rate)이 급격히 작아지는 현상이 발생할 수 있다.

    - 해결방법 1 - (클래스 면에서인듯?)

      비대칭 데이터는 다수 클래스 데이터에서 일부만 사용하는 언더 샘플링이나 소수 클래스 데이터를 증가시키는 오버 샘플링을 사용하여 데이터 비율을 맞추면 정밀도(precision)가 향상된다.

      - 오버샘플링(over-sampling)
      - 언더샘플링(under-sampling)
      - 복합샘플링(Combining over-and under-sampling)
      - https://m.blog.naver.com/jinty/221740980208

    - 해결방법 2 - log 등

      - Log Transform
      - Square Root Transform
      - Box-Cox Transform
      - Negative Skewed Data일 경우?
      - https://dining-developer.tistory.com/18
      - np.log(), np.log1p() https://suppppppp.github.io/posts/Why-Series-MDM-1/

- 기타함수

  - skew(왜도) : 분포의 비대칭도





https://lsjsj92.tistory.com/426
