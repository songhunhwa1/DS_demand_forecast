## 들어가며
- 하이퍼파라메터 최적화 문제는 모든 예측모델에 해당하는 문제
  - 예) 수요예측모델, 추천/랭킹모델, 최적화 모델 등
- 최적의 가중치를 산출하여 랭킹,추천 모델 개발에도 적용 가능
  - 변수별 가중치가 최적화 타깃이며, 랭킹 결과에 핵심적인 역할
- MAB에 적용된 개념(e-greedy, 톰슨샘플링 등) 피드백을 빠르게 반영하여 제공/노출 시안이나 상품을 빠르게 업데이트
  - 출처: [MAB (Multi-Armed Bandits)](https://soobarkbar.tistory.com/135)
  - 1) 여러번 시도를 통해 분포 획득
  - 2) 배너별 확률밀도함수

## Bayesian Optimization 핵심 내용
- [Bayesian Optimization 개요: 딥러닝 모델의 효과적인 hyperparameter 탐색 방법론 (1) - 블로그 | 코그넥스](https://www.cognex.com/ko-kr/blogs/deep-learning/research/overview-bayesian-optimization-effective-hyperparameter-search-technique-deep-learning-1)
- 핵심 개념
  - Surrogate 모델(e.g 가우시안 프로세스)
  - Acquisition Function (e.g Expected Improvement 등)
> 어느 입력값 x를 받는 미지의 목적 함수(objective function) f를 상정하여, 그 함숫값 f(x)를 최대로 만드는 최적해 x∗를 찾는 것을 목적으로 합니다. 보통은 목적 함수의 표현식을 명시적으로 알지 못하면서(i.e. black-box function), 하나의 함숫값 f(x)를 계산하는 데 오랜 시간이 소요되는 경우를 가정합니다. 이러한 상황에서, 가능한 한 적은 수의 입력값 후보들에 대해서만 그 함숫값을 순차적으로 조사하여, f(x)를 최대로 만드는 최적해 x∗를 빠르고 효과적으로 찾는 것이 주요 목표라고 할 수 있습니다.

|방법론|장점|단점|
|---|---|---|
|Grid Search|주어진 범위내 모든 조합 탐색, 많은 경우의 수 탐색|긴 소요시간, 이산(Discrete), 비효율 (불필요 영역 탐색)|
|Random Search|빠르게 결과 도출 가능, Continuous 한 접근|최적화 지점 놓칠 가능성, Random Sample → 오차 존재, 비효율 (불필요 영역 탐색)|
|Bayesian Optimization |기존 실험 결과를 기반으로 추정, 비교적 빠르게 결과 도출 가능|Surrogate, Acquisition Function 선정에 따라 상이한 결과, 과정에 대한 설명/이해가 쉽지 않음|

## 실제 코드 및 테스트 결과 (1) - 수요예측 Regression
- 전일 조별 주문수 예측 (Regression)
- 데이터셋 형상


## LGBM Default 하이퍼파라메터 성능

## Random Search  성능 테스트
```python
%%time
param_fin = pd.DataFrame()
i = 0
while i <= 5:    
    i += 1
    print(i)
    # Random Grid Search
    params = {} #initialize parameters
    params['learning_rate'] = np.random.uniform(0, 1)
    params['boosting_type'] = np.random.choice(['gbdt', 'dart', 'goss'])
    params['objective'] = 'regression'
    params['metric'] = 'mae'
    params['sub_feature'] = np.random.uniform(0, 1)
    params['num_leaves'] = np.random.randint(20, 300)
    params['min_data'] = np.random.randint(10, 100)
    params['max_depth'] = np.random.randint(5, 200)
    params['n_estimators'] = np.random.randint(100, 2000)
        
    # 조별로 필터링후 학습/예측
    pred_res = pd.DataFrame()
    rgn_list = df.region_group_code.drop_duplicates()        
    target_encoder = ce.TargetEncoder(cols=['biz_hour'])
    drop_cols = ['ord_cnt', 'region_group_code', 'ord_ratio_ar', 'ord_ratio', 'ord_sum']

    # 특정 과거 소급시 날짜 입력후 가능                                   
    for date in pd.period_range(target_min_date, target_date):
        logging.info(f"{date}: model train & predict is started")                        
        for rgn in rgn_list:

            # subset 생성
            subset = df.query("region_group_code == @rgn")

            # 예측수행일 날짜 제외하고 어제까지 데이터로만 학습
            ## todo: 최대 21일까지 예측 -> 이슈가 null 메꾸고 써야함. 우선 당일예측으로 집중
            train = subset[subset.index < str(date-1)] # 예측수행일 미만만 학습
            pred = subset[subset.index == str(date)] # 예측대상 기간

            X_train = train.drop(drop_cols, axis=1)            
            X_train['biz_hour_encode'] = target_encoder.fit_transform(train['biz_hour'], train['ord_cnt'])
            y_train = train[['ord_cnt']]

            X_pred = pred.drop(drop_cols, axis=1)    
            X_pred['biz_hour_encode'] = target_encoder.transform(pred['biz_hour'])                               
            y_pred = pred[['biz_hour', 'region_group_code', 'ord_cnt']] 

            #todo: 파라메터 최적화 필요
            model = LGBMRegressor(**params)            
            model.fit(X_train, y_train)
            y_pred['pred'] = model.predict(X_pred).astype(int)
            y_pred['pred_date'] = str(date-1) # 예측수행일  
            pred_res = pred_res.append(y_pred)        

        logging.info(f"{date}: model train & predict is finished")      
    pred_res = pred_res.reset_index()
    pred_res['biz_ymd'] = pd.to_datetime(pred_res['biz_ymd'])
    pred_res['pred_date'] = pd.to_datetime(pred_res['pred_date'])
    pred_res1 = pred_res.groupby(['biz_ymd', 'region_group_code'])[['ord_cnt', 'pred']].sum().reset_index()
    
    param_df = pd.DataFrame.from_dict(params, orient='index').T
    param_df['mdape'] = 1-(abs(pred_res1['pred']-pred_res1['ord_cnt'])/pred_res1['ord_cnt']).median()
    param_fin = param_fin.append(param_df)
%%time
param_fin = pd.DataFrame()
i = 0
while i <= 5:    
    i += 1
    print(i)
    # Random Grid Search
    params = {} #initialize parameters
    params['learning_rate'] = np.random.uniform(0, 1)
    params['boosting_type'] = np.random.choice(['gbdt', 'dart', 'goss'])
    params['objective'] = 'regression'
    params['metric'] = 'mae'
    params['sub_feature'] = np.random.uniform(0, 1)
    params['num_leaves'] = np.random.randint(20, 300)
    params['min_data'] = np.random.randint(10, 100)
    params['max_depth'] = np.random.randint(5, 200)
    params['n_estimators'] = np.random.randint(100, 2000)
        
    # 조별로 필터링후 학습/예측
    pred_res = pd.DataFrame()
    rgn_list = df.region_group_code.drop_duplicates()        
    target_encoder = ce.TargetEncoder(cols=['biz_hour'])
    drop_cols = ['ord_cnt', 'region_group_code', 'ord_ratio_ar', 'ord_ratio', 'ord_sum']

    # 특정 과거 소급시 날짜 입력후 가능                                   
    for date in pd.period_range(target_min_date, target_date):
        logging.info(f"{date}: model train & predict is started")                        
        for rgn in rgn_list:

            # subset 생성
            subset = df.query("region_group_code == @rgn")

            # 예측수행일 날짜 제외하고 어제까지 데이터로만 학습
            ## todo: 최대 21일까지 예측 -> 이슈가 null 메꾸고 써야함. 우선 당일예측으로 집중
            train = subset[subset.index < str(date-1)] # 예측수행일 미만만 학습
            pred = subset[subset.index == str(date)] # 예측대상 기간

            X_train = train.drop(drop_cols, axis=1)            
            X_train['biz_hour_encode'] = target_encoder.fit_transform(train['biz_hour'], train['ord_cnt'])
            y_train = train[['ord_cnt']]

            X_pred = pred.drop(drop_cols, axis=1)    
            X_pred['biz_hour_encode'] = target_encoder.transform(pred['biz_hour'])                               
            y_pred = pred[['biz_hour', 'region_group_code', 'ord_cnt']] 

            #todo: 파라메터 최적화 필요
            model = LGBMRegressor(**params)            
            model.fit(X_train, y_train)
            y_pred['pred'] = model.predict(X_pred).astype(int)
            y_pred['pred_date'] = str(date-1) # 예측수행일  
            pred_res = pred_res.append(y_pred)        

        logging.info(f"{date}: model train & predict is finished")      
    pred_res = pred_res.reset_index()
    pred_res['biz_ymd'] = pd.to_datetime(pred_res['biz_ymd'])
    pred_res['pred_date'] = pd.to_datetime(pred_res['pred_date'])
    pred_res1 = pred_res.groupby(['biz_ymd', 'region_group_code'])[['ord_cnt', 'pred']].sum().reset_index()
    
    param_df = pd.DataFrame.from_dict(params, orient='index').T
    param_df['mdape'] = 1-(abs(pred_res1['pred']-pred_res1['ord_cnt'])/pred_res1['ord_cnt']).median()
    param_fin = param_fin.append(param_df)
'''

## Bayesian Optimization 성능 테스트
'''python
## Bayesian Optimization
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score

drop_cols = ['ord_cnt', 'region_group_code', 'ord_ratio_ar', 'ord_ratio', 'ord_sum']
train = df[df.index < '2023-01-15']
X_train = train.drop(drop_cols, axis=1)            
y_train = train[['ord_cnt']]

def lgb_evaluate(n_estimators, num_leaves, max_depth, learning_rate):
    
    model = LGBMRegressor(n_estimators=int(n_estimators), 
                          num_leaves=int(num_leaves), 
                          max_depth=int(max_depth),
                          learning_rate=learning_rate
                         )    
    model.fit(X_train, y_train)
    scores = mean_absolute_error(model.predict(X_train), y_train)    
    return np.mean(scores)

def bayesOpt(X_train, y_train):
    lgbBO = BayesianOptimization(lgb_evaluate, {
                                               'n_estimators': (100, 2000),
                                                'num_leaves': (31, 200),
                                                'max_depth': (10, 200),
                                                'learning_rate': (0.001, 1)
                                               })

    lgbBO.maximize(init_points=2, n_iter=20)
    print(lgbBO.res)
    return lgbBO.max['params']
'''

## 실제 코드 및 테스트 결과 (2) -  랭킹모델

- 임의 데이터를 생성하여 테스트 진행
  - 초기 결과 (임의 가중치: 0.64, 0.23, 0.13 -> Score39.53
- Ranking 모델에 BO 적용 결과 
  - 최적화 가중치: 0.791, 0.3128, 0.1055) -> Score: 39.72
'''python
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

base = {
     'product_name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], 
     'discount_rate': np.random.uniform(0.1, 0.4, 10),
     'sales_count': np.random.randint(10, 1000, 10),
     'likes_count': np.random.randint(1, 200, 10)
     }

base = pd.DataFrame(base).set_index("product_name")
click_cnt = {'A': 100, 'B': 280, 'C': 30, 'D': 160, 'E':30, 'F':120, 'G':50, 'H':29, 'I':65, 'J': 8}

# BayesianOptimization
def objective_func(x, y, z):
        
    df = pd.DataFrame(MinMaxScaler().fit_transform(base), columns=base.columns, index=base.index)        
    df['score'] = df['discount_rate']*x + df['sales_count']*y + df['likes_count']*z
    df['ranks'] = df['score'].rank(ascending=False)
    
    df = df.sort_values('ranks').reset_index()    
    df['click'] = df['product_name'].replace(click_cnt)
    
    ranks_list = df['ranks'].values
    click_list = df['click'].values
    
    score = get_score(ranks_list, click_list)
    return score
    
bayes_optimizer = BayesianOptimization(objective_func, {'x': (.1, .9), 'y': (.1, .9), 'z': (.1, .9)}, random_state=42)    
bayes_optimizer.maximize(init_points=4, n_iter=50)
'''

## 마무리
- 1차로 생각해볼만한 프로젝트
- 랭킹/추천 모델 → 미지 함수, 로그 데이터 활용, 유저 행동 분포 획득, 빈번한 업데이트, 하이퍼파라메터 자동화 for 지표 극대화 등

## 참고자료
- Bayesian Optimization 개요: 딥러닝 모델의 효과적인 hyperparameter 탐색 방법론 (1) - 블로그 | 코그넥스 
- Bayesian Optimization 개요: 딥러닝 모델의 효과적인 hyperparameter 탐색 방법론 (2) - 블로그 | 코그넥스 
- Practical Bayesian Optimization of Machine Learning Algorithms (NIPS 2012) - README
- [ML] Bayesian Optimization으로 파라미터 튜닝하기 
- Bayesian optimization - Martin Krasser's Blog 
- Practical Bayesian Optimization of Machine Learning Algorithms


