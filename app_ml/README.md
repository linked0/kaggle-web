### 적용할 것
- 아래 <### plan> 부분 잊지 말기

- 구글 Design 문서에 적어놓은 아이디어 확인 필요

- LabelEncoder사용하기(Text Attribute)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]       <--- 이런식으로 개별적으로 끊어놓고 처리함 
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded

- Imputer 이용하기(hands-on machine learning 책)
housing_num = housing.drop("ocean_proximity", axis=1) <-- Imputer 사용하려면 텍스트 attribute는 잠깐 제거해야함 
X = imputer.transform(housing_num)
The result is a plain Numpy array containing the transformed features. If you want to put it back into a Pandas DataFrame, it’s simple:
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

- One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot
<16513x5 sparse matrix of type '<class 

==> from sklearn.preprocessing import LabelBinarizer가 한방에 끝낼 수 있어서 좋음

- 프로팅을 이런식으로..
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=4,
            s=housing['population']/100, label='population',
            c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
 Parameters
 |      ----------
 |      x, y : label or position, optional
 |          Coordinates for each point.
 |      s : scalar or array_like, optional
 |          Size of each point.
 |      c : label or position, optional
 |          Color of each point.
 |      **kwds : optional
 |          Keyword arguments to pass on to :py:meth:`pandas.DataFrame.plot`.
             
- corr
corr_matrix = housing.corr()

- 새로운 필드를 만들어야 함
housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room']= housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households']

- Missing Values 처리 방법
housing.dropna(subset=["total_bedrooms"])    # option 1
housing.drop("total_bedrooms", axis=1)       # option 2
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median)     # option 3
median은 기억하고 있어야 함 You will need it later to replace missing values in the test set when you want to evaluate your 
system, and also once the system goes live to replace missing values in new data.



### 필드
- PassengerId: int, 1부터 시작되는 승객 아이디
- Survived: int, 생존여부, 예측의 대상이 되는 y 값
- Pclass: int, 승객 입장권 클래스, 1이 가장 높고, 3이 가장 낮음
    > Fare 평균은 1: , 2:, 3: 
- Name: str
- Sex: str, 0:male/1:femaie
- Age: int
- SibSp: int, Number of Siblings/Spouses Aboard
- Parch: int, Number of Parents/Children Aboard
- Ticket: str, 의미없음
- Fare: float, 값이 0인경우가 15건이 됨.
- Cabin: str, 의미없음 
- Embarked: char, Southampton, Cherbourg, Queenstown
    > Fare 평균은 C: 59.95, S: 27.08, Q: 13.28

### 작업 순서
1. 필드 의미 파악
    - dt.head()로 데이터의 일부를 본다.
    - 각 필드의 데이터 타입 확인 
    - 다음의 같이 간단하게 추가의미를 파악한다.
        > dt[dt.Pclass == 1].Fare.mean()
    - 인터넷 검색등을 통해서 정보 알아내기.
2. missing data check 및 처리 (당연히 전체 데이터로 처리)
    - NaN도 확인해야하지만 값이 0인 경우도 확인해야 함. 이런 경우, isnull함수로 확인 불가능
    - Age: mean이 아닌 median을 사용함.
    - Fare: 클래스별 평균으로 입력한다.
        > missing 개수: Pclass 1 -> 5개, Pclass 2 -> 6, Pclass 3 -> 4
        > 클래스별 Fare 평균: 1 -> 86.15, 2 -> 21.36, 3 -> 13.79
    - Embarked: S: 644, C: 168, Q: 77, nan: 2
        > 제일 많은 S로 처리함. 
3. 모든 데이터를 numeric으로 바꾼다.
4. udacity MNIST 예제처럼 overlapping되는 샘플을 찾는다. 다음의 문제를 확인 잘 할 것
    Problem 5
    By construction, this dataset might contain a lot of overlapping samples, including training data that's also contained in the validation and test set! Overlap between training and test can skew the results if you expect to use your model in an environment where there is never an overlap, but are actually ok if you expect to see training samples recur when you use it. Measure how much overlap there is between training, validation and test samples.
    Optional questions:
    What about near duplicates between datasets? (images that are almost identical)
    Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
4. 결과 값을 one hot encoding으로 바꾼다.
5. train/test set 분리
6. preprocessing
    - standardization 수행 
    - 다른 필드와의 covariance 파악 
7. 각 모델에 대한 모듈(kernel_svm 등)으로 각각의 최적 hyperparameter를 선택하기
8. RandomizedSearchCV를 이용하는 param_search 모듈로 최적의 hyperparameter 찾아보기
9. GridSearchCV 이용하기 -> 아직 GridSearchCV가 동작하지 않는 문제 있음

--------------------------------------------------------------------
### 트레이닝 중간 결과
1. PCA의 feature 개수를 2로 했을때 전체보다 더 나쁨.
- Logistic Regression with PCA:False
    > Test Accuracy: 0.816
- Logistic Regression with PCA:True
    > Test Accuracy: 0.709
- Logistic Regression (cross validation) with PCA: False
    > CV accuracy: 0.785 +/- 0.041
- Logistic Regression (cross validation) with PCA: True
    > CV accuracy: 0.679 +/- 0.059
    
2. RandomizedSearchCV로 테스트했을때의 성능
- ## 중간 결론: parameter와 RandomizedSearchCV의 n_iter값을 어떻게 주느냐에 따라 값이 많이 차이남.
- RandomForestClassifier
    > params
    param_dist = {"max_depth": [3, None],
                "max_features": sp_randint(1, 7),
                "min_samples_split": sp_randint(1, 11),
                "min_samples_leaf": sp_randint(1, 11),
                "bootstrap": [True, False],
                "criterion": ["gini", "entropy"]}
    > 결과
    best score: 0.832865
    best estimator: RandomForestClassifier(bootstrap=False, class_weight=None,
            criterion='entropy', max_depth=None, max_features=2,
            max_leaf_nodes=None, min_samples_leaf=9, min_samples_split=3,
            min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
    best parameters: {'bootstrap': False, 'min_samples_leaf': 9, 'min_samples_split': 3, 'criterion': 'entropy', 'max_features': 2, 'max_depth': None}

    > 다른 하이퍼파라미터를 확인해봐야 함.
    
- SVC
    > params
    param_dist = {
          'clf__C': [0.01, 0.1, 1, 10, 100, 1000],
          'clf__gamma': [0.01, 0.1, 1, 10, 100, 1000],
          'clf__kernel': ['rbf', 'linear'],
    }
    > 결과
    best score: 0.824438
    best estimator: Pipeline(steps=[('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), ('clf', SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape=None, degree=3, gamma=0.1, kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False))])
    best params: {'clf__gamma': 0.1, 'clf__C': 1, 'clf__kernel': 'rbf'}

3. Random Forest옴(160719)
    > High Variance 발생
        Train accuracy: 0.982
        Test accuracy: 0.838
    > learning_curve를 실행시켜보면 확실하게 보임
    > 어떻게 해결한다?


### plan

매일
하루에 한개씩 리팩토링/서브작업
설계서 및 UML 작성

목표 / 나중에 할 것
여기서의 내용이 문서나 블로그로 정리되어야 함.(github으로 정리해도 좋음
여기서 짜는 코드와 프로세스가 프레임워크가 되어야 함. 
guideml이라는 파이선 스크립트를 만들고 프로세스를 이해할 수 있도록 하기 
pythonanywhere에 페이지 구성 필요.
AWS에 사이트를 만들필요가 있음.
교육 사이트로 확대 
예상 소프트웨어를 만들어야 함: 예를들어 디지털 시그널 처리 등등
코맨드 라인도 사용할 수 있도록 한다.
심플 UI과 전문가 UI가 선택할 수 있도록 하기

LANSAC, seaborn, heatmap적용 (히트맵 클릭하면 시본 나오도록)
kaggle kernel(Code or Notebook) 
    https://www.kaggle.com/daryadedik/titanic/could-the-titanic-have-been-saved
    http://nbviewer.jupyter.org/github/agconti/kaggle-titanic/blob/master/Titanic.ipynb
UI 적용: https://github.com/CalumJEadie/part-ii-individual-project-dev.git
Tensorflow 책의 Logistic Regression적용해보기 
http://www.mat.ucsb.edu/~g.legrady/academic/courses/11w259/schneiderman.pdf 적용
Feature Selection & Dimensionality Reduction: 지속적으로 봐야 함.
seaborn적용 
output 뷰 만들기
Tensorflow hidden Layer 적용
Regularization & Dropout
체계적인 UI, TensorBoard 같은 UI
SGDClassifier 적용
pythonanywhere를 사용하면서 사용법 동영상 만들어야함.
UI 적용: https://github.com/CalumJEadie/part-ii-individual-project-dev.git
NN - kaggle Test: 79.9%, not mnist Test: 89%

TensorFlow추가(+TensorBoard)
RANSAC 추가
Pipeline에 PCA 적용
Xgboost, Lasso, Ridge
SVM, Naive Bayes
Test Score 적용
PyML (Scikit-Learn) 적용
CNN 적용
GridSearch 적용
GraphViz program 적용
저장 및 로드 기능
탭바 적용
다양한 종류의 데이터 처리(MNIST)
대용량 처리
리포트 기능(웹 기능 정의, 황대리에게 묻기)
언어분석 기능
Recommender system 적용
프리프로세싱 처리
Import Hook 적용

12월
강화학습
Minecraft 
pygame 적용
캐글 종료일: Sat 31 Dec 2016
QTDesigner적용

<리팩토링>
데이터 구조 문서화
쓰레드 처리를 해서 중단이 가능하게 하고 프로그래스를 보여줘야 함.
TrainSettingModelSelection.py 클래스명 리팩토링 
self에 모든 변수를 세팅하는 것은 지양해야할 수 있음.
어떤 파라미터를 먼저 처리했느냐에 따라서 그래프 변화추이가 다르다.
Signal과 MLB stat도 처리해보자.
run_kernel_svm과 validate_kernal_svm의 KernelSVM 클래스 생성 코드 공통 패키지에 넣기 
Validation graph 부분
EventListener 만들기
각 알고리즘 클래스화 하기 
do_model_selection이 맞나?
show_learning_curve와 show_validation_curve는 공통으로 사용할 수 있음.
preprocess_data는 set_data는 한번만 호출하도록 하기
데이터도 selection해야함. 
데이터 선택하는 탭을 하다 만들어야 함.
hyper_param -> hyperparam 변경
TrainConfigHyperparamView 좀 길지 않나 싶음
TrainConfigHyperparamLayout 제거하기 
from 구문 정리 필요
from common import control_view as gplt -> import common.control_view as gilt


<서브 작업 I>
Mastering Python Data Visualization 적용
모든 알고리즘 결과 보관하기
메뉴 만들기
하나의 계정으로 성공하고 지영이 계정으로 데모를 보여주는 것.
기능 별로 잘라야 함
일단 train_test_split 함수 쓰고 나중에 KFold로 다시 정리하기 
분산모델을 만들어서 적용해야함.
SGDClassifier (Stochastic Gradient Descent Classifier)를 쓸 수 있는 옵션을 만들어야 함.
ppn = SGDClassifier(loss='perceptron')
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')
mlb의 stat같은 물음표 기능을 넣자.(예를 들어, C는 1/lambda를 의미함.
mlb stat으로 머신러닝 해보자. 모든 솔루션을 다 만들어서 팔아버리자. 소스까지 다 넘기기. 한번에 1000만원씩해서 넘기면...  그동안 기술사따서 CTO자리를 알아보는 것도 굿.
plot_decision_regions 적용
mlxtend의 FeatureSelection 나머지 적용
ExhaustiveFeatureSelector
ColumnSelector
sklearn.ensemble.VotingClassifier(MajorityVoteClassifier)
cov plotting
Ridge
GridSearchCV용
    > Wisconsin의 유방암 데이터와는 잘 맞고 타이타닉 데이터와는 안 맞는 것 같음.
클러스터링후에 라벨링한 것과 사람이 라벨링한 것과의 차이를 보자(예측률 100% 달성후)
각 feature사이의 상관관계


<서브 작업 II>
KFold 관련
    > The standard value for k in k-fold cross-validation is 10, which is typically a reasonable choice for most applications. However, if we are working with relatively small training sets, it can be useful to increase the number of folds. If we increase the value of k, more training data will be used in each iteration, which results in a lower bias towards estimating the generalization performance by averaging the individual model estimates. However, large values of k will also increase the runtime of the cross-validation algorithm and yield estimates with higher variance since the training folds will be more similar to each other. On the other hand, if we are working with large datasets, we can choose a smaller value for k, for example, K-fold cross-validation, and still obtain an accurate estimate of the average performance of the model while reducing the computational cost of refitting and evaluating the model on the different folds.
    > A special case of k-fold cross validation is the leave-one-out (LOO) cross-validation method. In LOO, we set the number of folds equal to the number of training samples (k = n) so that only one training sample is used for testing during each iteration. This is a recommended approach for working with very small datasets
hj-comment 확인
아래 페이지의 Example(Examples using sklearn.svm.SVC) 코드 적용해 보기
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.set_params
winsconsin 대학의 Malignant/Benign cancer의 비율이 4:6인데도 StratifiedKFlod를 사용함 
PC의 성능때문이라도 모든 데이터를 가지고 트레이닝하는 것이 아니라 일부 데이터로 1차로 모델을 만들어서 검증해볼 수 있다.
Binary Reverance관련 참고해서 적용할 만함: https://stackoverflow.com/questions/38826221/difference-between-binary-relevance-and-one-hot-encoding
Train, Validation, Test set 비율 UI
train num, eta 선택 UI