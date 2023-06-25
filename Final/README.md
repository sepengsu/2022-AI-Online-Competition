2022 인공지능 온라인 경진대회
과제: [수치해석] 상수관로 누수감지 및 분류 문제          
서재원

1. 초기 디렉토리 및 파일 위치 구조
```
${PROJECT}
├── weight/
├── Models/
├── results/        
├── DATA
│       ├── 00_source/
│       │          └── train.csv
│       │          └── test.csv
│       ├── 01_split/
│       └── sample_submission.csv
├── README.hwp
├── train.ipynb
├── predict.ipynb
└── preprocess.ipynb
```
폴더

weight: 추론에 필요한 가중치 정보 파일 csv를 저장하는 폴더

Models: 학습된 모델을 저장하는 폴더 (h5파일로 모델 저장)

results: 최종 예측 데이터를 저장하는 폴더 

파일
- preprocess.ipynb: 데이터 전처리 시 실행하는 코드
- train.ipynb: 학습 시 실행하는 코드
- predict.ipynb: 추론 시 실행하는 코드
- train.csv: 제공되는 훈련 데이터
- test.csv: 제공되는 예측 데이터
- sample_submission.csv: 제공되는 제출 샘플 데이터로, test.csv 정렬을 위하여 필요
 
2. 파일코드 및 코드 실행과정 간단 설명

2-1. preprocess.ipynb
위 파일에서는 데이터 전처리를 진행한다. 
- train, test, sample_submission 데이터 가져오기
- test데이터를 sample_submission를 기준으로 정렬
- train 데이터 결측치 및 중복 데이터 제거
- k-Fold split하기
- 클래스를 배열 형태로 바꾸는 ANN Labeling하기
- k-fold 된 데이터, ANN Labeling과 정렬된 test 데이터를 DATA 폴더 하위 위치에 아래와 같이 저장하기
※ ANN Labeling은 ANN 모델 학습을 위하여 5가지 클래스 값을 1차원 배열로 바꾸었다.
```
DATA
├── 00_source/
│          └── train.csv
├── 01_split/
│         └── k_Fold_1.csv : preprocess.ipynb 실행 후 새로 저장되는 훈련데이터
│                  ... (k=10으로 10개 생성)
│         └── k_FoldTrainLabel_1.csv :  preprocess.ipynb 실행 후 새로 저장되는 ANN 라벨 데이터
│                  ... (k=10으로 10개 생성)
│         └── test.csv ... preprocess.ipynb 실행 후 다시 저장되는 예측데이터 
└── sample_submission.csv
```

2-2. train.ipynb
위 파일에서는 학습을 진행한다.
- k-fold split을 한 훈련데이터와 ANN Label 데이터를 가져온다.
- ann 모델 10개를 정의하고 하이퍼파라미터를 설정한다.
- k-fold 교차검증을 이용하여 ANN 모델 10개를 학습한다.
- 모델 가중치를 k-fold 교차검증 시 각 모델의 최종 accuracy로 설정한다.
- 가중치와 ANN 모델 10개를 아래와 같이 저장한다.
```
${PROJECT}
├── weight/
│       └── Weight.csv : train.ipynb 실행 후 새로 저장되는 가중치데이터
├── Models/
│       ├── ann_1.h5 : train.ipynb 실행 후 새로 저장되는 모델
         └──        ... (k=10으로 10개 생성)
```
2-3. predict.ipynb
위 파일에서는 예측을 한다.
- 모델 10개, 가중치 파일, test 데이터 가져오기
- 모델 10개 예측값 생성하기
- 각 예측값에 대하여 각 모델의 가중치를 곱한후 모두 더하여 최종 예측값 생성
- 최종 예측값을 이용하여 클래스 추론하기
- 예측한 데이터를 아래와 같은 위치에 저장하기
```
${PROJECT}
├── weight/
├── Models/
├── results/  
         └──predictions.csv : predict.ipynb 실행 후 새로 저장되는 예측결과데이터
```
3. 코드 전부 실행 후 디렉토리 및 파일 위치 구조
```
${PROJECT}
├── weight/
│       └── Weight.csv : train.ipynb 실행 후 새로 저장되는 가중치데이터
├── Models/
│       ├── ann_1.h5 : train.ipynb 실행 후 새로 저장되는 모델
│       └──        ... (k=10으로 10개 생성)
├── results/  
│       └──predictions.csv : predict.ipynb 실행 후 새로 저장되는 예측결과데이터
├──DATA
│	├── 00_source/
│	│          └── train.csv
│	├── 01_split/
│	│         └── k_Fold_1.csv : preprocess.ipynb 실행 후 새로 저장되는 훈련데이터
│	│                  ... (k=10으로 10개 생성)
│	│         └── k_FoldTrainLabel_1.csv :  preprocess.ipynb 실행 후 새로 저장되는 ANN 라벨 데이터
│	│                  ... (k=10으로 10개 생성)
│	│         └── test.csv ... preprocess.ipynb 실행 후 다시 저장되는 예측데이터 
│	└── sample_submission.csv
├── README.hwp
├── train.ipynb
├── predict.ipynb
└── preprocess.ipynb

4. 인공지능 사용 방법론
4-1. DNN
이번 인공지능 모델에서 은닉층을 2개이상으로 늘린 DNN을 활용하였다. 그리고 DNN에서 배치 정규화, 드롭아웃, 활성화 함수, Nadam을 사용하였다.
모델의 구조는 아래와 같다.
입력층 -> 은닉층 1-> 은닉층 2-> 출력층 
그리고 모델의 1회 학습 순서는 다음과 같다.
입력층(활성화함수 = relu)-> 배치 정규화 -> 드롭아웃(1/4) -> 은닉층1(활성화함수 = 'elu')-> 배치 정규화 -> 
은닉층2(활성화함수 = relu)-> 배치 정규화 -> 드롭아웃(1/2) -> 출력층(함수 = softmax) 
또한 overfitting을 방지하기 위하여 옵티마이저를 Nadam으로 설정하였다.

생성 모델 함수 코드는 아래와 같다.
def ANN_model(input_data,Initial,NoOfNeuron):
    model = keras.Sequential()
    model.add(keras.layers.Dense(units = NoOfNeuron, activation = 'relu',input_shape =(input_data,))) 
    model.add(BatchNormalization()) # 배치 정규화
    model.add(keras.layers.Dropout(1/4)) # dropout 
    model.add(keras.layers.Dense(units = NoOfNeuron, activation = 'elu')) #은닉층 1,
    model.add(BatchNormalization()) # 배치 정규화
    model.add(keras.layers.Dense(units = NoOfNeuron, activation = 'relu')) # 은닉층 2
    model.add(BatchNormalization()) # 배치 정규화
    model.add(keras.layers.Dropout(1/2)) # dropout     
    model.add(keras.layers.Dense(units = 5,  activation = 'softmax')) # 출력층
    Optimizer=keras.optimizers.Nadam(learning_rate=Initial) # 학습률 최적화
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=Optimizer) 
    return model

4-2. k-fold 앙상블
k-fold 교차검증은 원래 과적합을 막기 위하여 행하는 것으로 k개의 데이터 폴드 세트를 만들어서 k번 만큼 아래 과정과 같이 학습과 검증 평가를 반복하는 방법이다.

또한 앙상블은 weak learner들이 모여 voting를 통해 더욱 더 강력한 strong learner를 구성하는 것이다. voting의 종류는 hard voting과 soft voting이 있는데 이 중 아래 그림과 같이 soft voting을 사용하였다. 소프트 보팅은 weak learner들의 예측 확률값의 평균이나 가중치 합으로 예측을 진행하는 방법이다.

이 두가지를 합쳐서 인공지능을 진행하였다. k-fold에서 k번 모델을 학습하는데 이 때 k개의 모델을 앙상블의 weak model로 설정하였다. 이렇게 한 이유는 모든 데이터를 학습에 사용하기 위해서이다.
4-1에서 우리는 일부 데이터를 validation으로 사용해야 한다. 즉, 학습에 사용할 수 없다. 그렇게 된다면 데이터 부족으로 인하여 overfitting이 발생할 수 있다. 따라서 모든 데이터를 학습에 사용하기 위해서 앙상블과 k-fold 모델들을 사용하였다.
그리고 weak voting에서 예측 확률값 x 가중치(확률 평균)을 사용하였다.
가중치를 곱한 이유는 가중치를 모델의 성능으로 판단, 더 좋은 모델의 확률값이 더 큰 의사결정권을 가질 수 있도록 하기 위해서이다.
