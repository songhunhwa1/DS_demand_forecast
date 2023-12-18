import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime, date, timedelta
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
warnings.filterwarnings("ignore")
from common.io.read import main_read
from common.util.aargparser import get_aargparser
from common.data.region_mapper import RegionMapper
from common.batch.dataprocessor import GCPDataProcessor
from common.io.read import AWSRead, GCPRead
from common.data.cookie import read_ar_goal_info_by_ondo_temp

class OrderPredictByCenter(GCPDataProcessor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.start_date = self._date_parse("2022-10-01")
        self.end_date = self.target_date
        self.target_min_date = self.target_date - timedelta(days=0) # 과거 예측기간
        self.rgn_mapper = RegionMapper(self.target_date)
        self.opt_param = False # True 경우 파라메터 최적화 진행
        self.param_space = {
                            'learning_rate': [0.01, 0.05, 0.1],
                            'n_estimators': [100, 200, 300],
                            'max_depth': [3, 5, 10],
                            'num_leaves': [20, 30, 40],
                            'min_child_samples': [1, 5, 10],
                            'subsample': [0.5, 0.7, 0.9],
                            'colsample_bytree': [0.5, 0.7, 0.9]
                            }        
        self.is_save = True
        self.table_name = "mkdf_center_predict_final_v3" 
        self.save_idx = ['biz_ymd', 'biz_hour', 'center_dlvy', 'pred_date', 'pred_hour']
        self.storage_middle_path = "data"
        self.sql_path = "g_sql"


    @staticmethod
    def _type_change(df):
        
        """
        df 받아서 pandas 형태의 컬럼 형식으로 numpy로 변경한후 df return
        """
        float_cols = df.select_dtypes(include=['Float64', 'Int64']).columns
        df[float_cols] = df[float_cols].astype(float)        
        ord_cols = [c for c in df.columns if "ord" in c]
        df[ord_cols] = df[ord_cols].astype(float)
        df['biz_hour'] = df['biz_hour'].astype(int)
        
        return df             
    
            
        
    def load_data(self):

        """
        과거 조별 실제 주문수 추출하고
        당일 미래 발생될 인덱스 추가
        """
        df = GCPRead().read_table_with_sql_file(sql_file_name='ord_cnt_by_dlvy_daily_v6.sql',
                                                start_date=self.start_date.to_date, end_date=self.end_date.to_date,
                                                SQL_PATH=self.sql_path)
        df['biz_ymd'] = pd.to_datetime(df['biz_ymd'])
        df.rename(columns={'region_group_cd': 'region_group_code'}, inplace=True)        
        
        print("start_date & end_date", self.start_date.to_date, self.end_date.to_date)
        logging.info("load the basic dataset is finished")
        return df
    
    def add_reserved(self):

        # 함께구매 추출
        together = GCPRead().read_table_with_sql_file(
                'ord_cnt_by_together.sql'
                , start_date = (self.end_date - pd.Timedelta(days=21)).to_date
                , end_date = self.end_date.to_date
                , SQL_PATH='g_sql'
                , use_storage = True
                , check_zero =False)\
                .rename(columns={'ord_cnt':'together_ord_cnt', 'unit_cnt':'together_unit_cnt'})
        
        # '내일 날짜 = 예약배송 날짜 = 오늘 물류 처리'인 것만 추출
        together = together[together['reserved_ymd'] == (self.end_date + pd.Timedelta(days = 1)).to_date]
        together['center_cd'] = 'CC03'
        together['dlvy_type'] = 'NON_BASIC'
                
        # 가장 마지막 시간인 23시 50분에 예약배송 건수를 더해줌
        together['biz_ymd'] = pd.to_datetime(self.end_date.to_date)
        together['biz_hour'] = 0
        together['center_dlvy'] = together['center_cd'] + '_' + together['dlvy_type']
        df_together = together[['biz_ymd','biz_hour','center_dlvy','together_ord_cnt','together_unit_cnt']]

        return df_together
                        
    # todo: 보정 로직 검토 필요    
    def apply_conti_ord_cnt(self, df, quantile=0.9):
        
        """
        컨티가 발생된 경우 보정치로 변경 > 과보정으로 임시 홀드
        """
        
        # 컨티값 보정: 샛별배송(BASIC)만 컨티 시간대를 평균 주문수로 대치
        conti = GCPRead().read_table_with_sql_file(sql_file_name='continuity_rgn_model_fe.sql',
                                                   start_date=self.start_date.to_date, end_date=self.end_date.to_date,
                                                   SQL_PATH=self.sql_path)
        conti['biz_ymd'] = pd.to_datetime(conti['biz_ymd'])

        # 센터의 권역들을 받아와 처리        
        basic_regions = self.rgn_mapper.BASIC
        conti = conti[conti['region_group_code'].isin(basic_regions)]

        # 토요일 택배의 경우 주문수를 0으로 수정
        df['dayofweek'] = df['biz_ymd'].dt.day_name()
        df['ord_cnt'] = np.where((df['region_group_code'].isin(self.rgn_mapper.NON_BASIC)) & (df['dayofweek'] == 'Saturday'), 0, df['ord_cnt'])

        # 컨티 대상 시간대는 분포고려하여 90퍼센타일 수치로 변환
        df = df.merge(conti, on=['biz_ymd', 'biz_hour', 'region_group_code'], how='left').fillna(0)
        df['ord_cnt_mean'] = df.groupby(['region_group_code', 'dayofweek', 'biz_hour'])['ord_cnt'].transform(lambda x: x.quantile(quantile))
        df['ord_sales_mean'] = df.groupby(['region_group_code', 'dayofweek', 'biz_hour'])['ord_sales'].transform(lambda x: x.quantile(quantile))

        df['ord_cnt'] = np.where(df['is_continuity'] == 1, df['ord_cnt_mean'].astype(int), df['ord_cnt'].astype(int))
        df['ord_sales'] = np.where(df['is_continuity'] == 1, df['ord_sales_mean'].astype(int), df['ord_sales'].astype(int))

        df = df.drop(['ord_cnt_mean', 'ord_sales_mean', 'is_continuity', 'dayofweek'], axis=1)
        logging.info("apply_conti_ord_cnt is finished")
        return df
        
        
        
    def change_to_center_dlvy(self, df):
        
        """
        조별 집계단위를 센터-배송유형 단위로 변경
        region mapper로 센터별 소급
        택배의 경우 전일 컨티주문수를 당일에 합산(캐파부족과 상관 없는 추가 주문으로 판단)
        """
        
        # 센터, 배송유형 소급처리
        df['center_code'] = df['region_group_code'].replace(self.rgn_mapper.get_center_code_per_region_group_code())
        df['dlvy_type'] = df['region_group_code'].replace(self.rgn_mapper.get_dlvy_type_per_region_group_code())        
        df = df[df['center_code'].isin(self.rgn_mapper.center_code)]

        df['center_dlvy'] = df['center_code']+'_'+df['dlvy_type']        
        df = df.groupby(['biz_ymd', 'center_dlvy', 'biz_hour'])[['ord_cnt', 'ord_sales']].sum().reset_index()
        
        ## 택배지역: 전일 택배 컨티 주문수 추출하여 주문수에 합산
        sql = f"""
            select
                biz_ymd,
                delivery_type as dlvy_type,
                center_code,        
                sum(ord_cnt) as cont_ord_cnt,
                sum(sales) as cont_ord_sales
            from bq-datafarm.data_science.continuity_ord_cnt_by_rgn_1d
            where 1=1
                and biz_ymd > '{self.start_date.to_date}'
                and biz_ymd <= '{self.end_date.to_date}'
                and delivery_type = 'NON_BASIC'
            group by
                biz_ymd,
                delivery_type,
                center_code        
            """

        conti_non = GCPRead().read_sql_query(sql=sql, check_zero=False)
        conti_non['biz_ymd'] = pd.to_datetime(conti_non['biz_ymd'])
        conti_non['center_dlvy'] = conti_non['center_code']+'_'+conti_non['dlvy_type']
        conti_non['biz_hour'] = 0

        # 택배의 경우 전일 컨티를 집계에 합산
        df = df.merge(conti_non, on=['biz_ymd', 'biz_hour', 'center_dlvy'], how='left').fillna(0)
        df['ord_cnt'] = df['ord_cnt']+df['cont_ord_cnt']    
        df['ord_sales'] = df['ord_sales']+df['cont_ord_sales'] 

        logging.info("apply_non_basic_conti_ord is finished")
        return df.drop(['dlvy_type', 'center_code', 'cont_ord_cnt', 'cont_ord_sales'], axis=1)
    
    
    
    def get_pure_hit_cnt(self, df):
        
        """
        AR에 제공하는 경우 16시 이후 푸시로 인한 증분을 제거해야함.
        """
        
        camp = GCPRead().read_table_with_sql_file(sql_file_name='push_pure_cnt_by_center.sql',
                                                   start_date=self.start_date.to_date,
                                                   SQL_PATH=self.sql_path)
        
        camp['biz_ymd'] = pd.to_datetime(camp['biz_ymd'])
        camp['center_dlvy'] = camp['center_code']+'_'+camp['dlvy_type']
        camp = camp.drop(['center_code', 'dlvy_type'], axis=1)
        camp['pure_hit_cnt'] = np.where(camp['center_dlvy'] == '0_0', camp['pure_hit_cnt']*0.5, camp['pure_hit_cnt'])

        camp0 = camp.query("center_dlvy == '0_0'")
        camp0['center_dlvy'] = 'CC02_BASIC'
        camp['center_dlvy'] = np.where(camp['center_dlvy'] == '0_0', 'CC01_BASIC', camp['center_dlvy'])
        camp = pd.concat([camp, camp0])
        
        df = df.merge(camp, on=['biz_ymd', 'center_dlvy', 'biz_hour'], how='left')
        df['pure_hit_cnt'] = df['pure_hit_cnt'].fillna(df.groupby(['biz_ymd', 'center_dlvy'])['pure_hit_cnt'].ffill())
        df['pure_hit_cnt'] =  df.groupby(['biz_ymd', 'center_dlvy'])['pure_hit_cnt'].apply(lambda x: x / x.notnull().sum())
        df['pure_hit_cnt'] = df['pure_hit_cnt'].fillna(0)
        df['ord_cnt_purehit'] = (df['ord_cnt']-df['pure_hit_cnt']).astype(int)
        
        logging.info("get camp dataset is finished")
        return df.drop('pure_hit_cnt', axis=1)
    
    
    
    def make_future_dataset(self, df):
        
        """
        미래 예측 대상이 되는 인덱스 생성
        """        
        biz_hour_list = df['biz_hour'].drop_duplicates().to_frame()
        center_dlvy_list = df['center_dlvy'].drop_duplicates().to_frame()

        future_df = pd.merge(biz_hour_list, center_dlvy_list, how='cross')
        future_df['biz_ymd'] = self.target_date.to_date
        future_df['biz_ymd'] = pd.to_datetime(future_df['biz_ymd'])

        df = df.merge(future_df, on=['biz_ymd', 'center_dlvy', 'biz_hour'], how='outer') 

        logging.info("make_future_dataset is finished")
        return df.sort_values(['biz_ymd','center_dlvy','biz_hour'])

    
    
    def get_ar_goal(self):

        """
        AR 목표 주문건수: 최대 90일 추출
        """         
        # CHECK: GCP 에서도 함수 작동?
        ar_goal = read_ar_goal_info_by_ondo_temp(start_date=self.start_date.to_date, end_date=self.end_date.to_date, target_date=self.target_date.to_date) 
        ar_goal['biz_ymd'] = pd.to_datetime(ar_goal['biz_ymd'])

        ar_goal = ar_goal[['biz_ymd', 'center_cd', 'dlvy_type', 'ar_cnt']]        
        ar_goal.rename(columns = {'center_cd':'center_code'}, inplace=True)
        ar_goal['center_dlvy'] = ar_goal['center_code']+'_'+ar_goal['dlvy_type']

        logging.info("ar goal dataset is done")
        return ar_goal.drop(['center_code', 'dlvy_type'], axis=1)
    
    
    
    def add_fe_ar_goal(self, df, ar):

        """
        AR 목표값 추출해서 피처로 활용
        """
        
        df = df.merge(ar, on=['biz_ymd', 'center_dlvy'], how='left')

        ## AR 기반으로 카운트 예측시도하여 피처로
        df['ord_ratio_ar'] = df['ord_cnt'] / df['ar_cnt']
        df['ord_ratio_ar_w1'] = df.groupby(['center_dlvy', 'biz_hour'])['ord_ratio_ar'].shift(7)
        df['ord_ratio_ar_w2'] = df.groupby(['center_dlvy', 'biz_hour'])['ord_ratio_ar'].shift(14)
        df['ord_ratio_ar_w3'] = df.groupby(['center_dlvy', 'biz_hour'])['ord_ratio_ar'].shift(21)

        df['ord_ratio_ar_w2_mean'] = df[['ord_ratio_ar_w1', 'ord_ratio_ar_w2']].mean(axis=1)
        df['ord_ratio_ar_w3_mean'] = df[['ord_ratio_ar_w1', 'ord_ratio_ar_w2', 'ord_ratio_ar_w3']].mean(axis=1)

        ## AR 베이스 * 2, 3주 평균 비중을 피처로 활용
        df['ord_pred_cnt_ar_w2'] = df['ord_ratio_ar_w2_mean'] * df['ar_cnt']
        df['ord_pred_cnt_ar_w3'] = df['ord_ratio_ar_w3_mean'] * df['ar_cnt']

        logging.info("add_fe_ar_goal is finished")
        return df
    
    
    
    def add_fe_daterelated(self, df):

        """
        날짜 관련 피처 추가
        """
        df['biz_ymd'] = pd.to_datetime(df['biz_ymd'])
        df["month"] = df['biz_ymd'].dt.month #월
        df["week_no"] = df['biz_ymd'].dt.isocalendar()['week'].astype('int')
        df['week_num'] = np.ceil((df['biz_ymd'].dt.to_period('M').dt.to_timestamp().dt.weekday + df['biz_ymd'].dt.day) / 7.0).astype(int)  # 월별주차
        df["dayofweek"] = df['biz_ymd'].dt.day_name() # 요일

        df = pd.concat([df, pd.get_dummies(data=df['dayofweek'], prefix='dayofweek')], axis=1)  # 요일 one-hot encoding
        df["weekend_yn"] = np.where(df["dayofweek"].isin(["Saturday", "Sunday"]), 1, 0)  # 주말여부
        df["dayofyear"] = df['biz_ymd'].dt.dayofyear  # 연간 일
        df['dayofweek'] = LabelEncoder().fit_transform(df['dayofweek'])

        logging.info("add_fe_daterelated is finished")
        return df
    
    
    
    def add_fe_holiday(self, df):

        """
        휴일 및 마케팅/프로모션 관련 변수 추가
        """
        ## TODO: SQL 안에 테이블 수정 DONE
        holiday_df = GCPRead().read_table_with_sql_file(sql_file_name='holiday_info_v2.sql',
                                                        start_date=self.start_date.to_date,
                                                        use_athena=False,
                                                        SQL_PATH=self.sql_path)
        holiday_df.rename(columns={'ymd': 'biz_ymd'}, inplace=True)
        holiday_df['biz_ymd'] = pd.to_datetime(holiday_df['biz_ymd'])

        # 컬리 이벤트와 공휴일을 구분
        holiday_df_kurly = holiday_df[holiday_df.holiday.str.contains('컬리')].groupby(['biz_ymd'])['holiday'].apply(lambda x: '&'.join(x)).reset_index()
        holiday_df_public = holiday_df[~holiday_df.holiday.str.contains('컬리', na=True)].groupby(['biz_ymd'])['holiday'].apply(lambda x: '&'.join(x)).reset_index()

        # 공휴일변수
        df = df.merge(holiday_df_public, 'left', on = 'biz_ymd')
        df['traditional_holiday_yn'] = 0; df['public_holiday_yn'] = 0
        df.loc[df.holiday.str.contains('설날|추석|성탄절|기독탄신일', na=False), 'traditional_holiday_yn'] = 1
        df.loc[~df.holiday.str.contains('설날|추석|성탄절|기독탄신일', na=True), 'public_holiday_yn'] = 1
        df.drop(columns='holiday', inplace=True)

        # 컬리 이벤트 변수
        df = df.merge(holiday_df_kurly, 'left', on = 'biz_ymd')
        df['dlvy_stop_yn'] =0; df['purple_day_hr_yn'] = 0; df['cj_coupon_yn'] = 0; df['kurly_special_festa'] = 0
        df.loc[df.holiday.str.contains('컬리_퍼플아워|컬리_퍼플데이|컬리_퍼플위크', na=False), 'purple_day_hr_yn'] = 1
        df.loc[df.holiday.str.contains('컬리_플렉스위크|컬리_세이브위크|컬리_뷰티위크|컬리_라이브커머스|컬리_블랙위크|컬리_플러스위크|컬리_라이브방송', na=False), 'kurly_special_festa'] = 1
        df.loc[df.holiday.str.contains('샛별불가', na=False), 'dlvy_stop_yn'] = 1
        df.loc[df.holiday.str.contains('컬리_CJ쿠폰', na=False), 'cj_coupon_yn'] = 1
        df.drop(columns='holiday', inplace=True)

        # traditional holiday 이전, 이후 7일
        df['before_holiday'] = np.nan; df['after_holiday'] = np.nan
        for day in df[df.traditional_holiday_yn == 1].biz_ymd.unique():
            day = pd.to_datetime(day)
            bf14 = (day - pd.Timedelta(days=14)).strftime(self.date_format)
            bf1 = (day - pd.Timedelta(days=1)).strftime(self.date_format)
            af7 = (day + pd.Timedelta(days=7)).strftime(self.date_format)
            af1 = (day + pd.Timedelta(days=1)).strftime(self.date_format)
            df.loc[(df.biz_ymd >= bf14) & (df.biz_ymd <= bf1) & (df.traditional_holiday_yn == 0), 'before_holiday'] = 1
            df.loc[(df.biz_ymd >= af1) & (df.biz_ymd <= af7) & (df.traditional_holiday_yn == 0), 'after_holiday'] = 1

        df[['before_holiday', 'after_holiday']] = df[['before_holiday', 'after_holiday']].fillna(0)
        logging.info("add_fe_holiday is finished")
        return df    
    
    
    
    def add_fe_laggging(self, df, max_lagging_n=30):

        """
        lagging 피처 및 rolling mean 피처 추가
        """
        # 누적주문수
        df['ord_cnt_cum'] = df.groupby(['biz_ymd', 'center_dlvy'])['ord_cnt'].cumsum()

        # 이전 동시간대 주문수 피처
        for n in np.arange(1, max_lagging_n):

            # lagging values
            df[f'ord_cnt_d{n}'] = df.groupby(['center_dlvy', 'biz_hour'])['ord_cnt'].shift(n)
            df[f'ord_cnt_cum_d{n}'] = df.groupby(['center_dlvy', 'biz_hour'])['ord_cnt_cum'].shift(n)
            df[f'ord_cnt_roll_d{n}'] = df.groupby(['center_dlvy', 'biz_hour'])['ord_cnt_d1'].transform(lambda x: x.rolling(n,1).mean())
            df[f'ord_cnt_ewm_d{n}'] = df.groupby(['center_dlvy', 'biz_hour'])['ord_cnt_d1'].transform(lambda x: x.ewm(span=n, adjust=False).mean())

        logging.info("add_fe_laggging is finished")
        return df
    
    
    
    def add_fe_change_ratio(self, df, max_change_n=14):

        """
        증감율 피처 추가
        """
        for n in np.arange(1, max_change_n):

            df[f'ord_cnt_change_d1{n}'] = (df[f'ord_cnt_d{n}']-df[f'ord_cnt_d{n+1}'])/df[f'ord_cnt_d{n+1}']
            df[f'ord_cnt_change_d2{n}'] = (df[f'ord_cnt_d{n+1}']-df[f'ord_cnt_d{n+2}'])/df[f'ord_cnt_d{n+2}']
            df[f'ord_cnt_change_d3{n}'] = (df[f'ord_cnt_d{n+2}']-df[f'ord_cnt_d{n+3}'])/df[f'ord_cnt_d{n+3}']
            df[f'ord_cnt_change_d4{n}'] = (df[f'ord_cnt_d{n+3}']-df[f'ord_cnt_d{n+4}'])/df[f'ord_cnt_d{n+4}']

            df[f'ord_cnt_cum_change_d1{n}'] = (df[f'ord_cnt_cum_d{n}']-df[f'ord_cnt_cum_d{n+1}'])/df[f'ord_cnt_cum_d{n+1}']
            df[f'ord_cnt_cum_change_d2{n}'] = (df[f'ord_cnt_cum_d{n+1}']-df[f'ord_cnt_cum_d{n+2}'])/df[f'ord_cnt_cum_d{n+2}']
            df[f'ord_cnt_cum_change_d3{n}'] = (df[f'ord_cnt_cum_d{n+2}']-df[f'ord_cnt_cum_d{n+3}'])/df[f'ord_cnt_cum_d{n+3}']
            df[f'ord_cnt_cum_change_d4{n}'] = (df[f'ord_cnt_cum_d{n+3}']-df[f'ord_cnt_cum_d{n+4}'])/df[f'ord_cnt_cum_d{n+4}']


        df[f'ord_cnt_change_w1'] = (df[f'ord_cnt_d2']-df[f'ord_cnt_d7'])/df[f'ord_cnt_d7']
        df[f'ord_cnt_change_w2'] = (df[f'ord_cnt_d7']-df[f'ord_cnt_d14'])/df[f'ord_cnt_d14']
        df[f'ord_cnt_change_w3'] = (df[f'ord_cnt_d14']-df[f'ord_cnt_d21'])/df[f'ord_cnt_d21']

        df[f'ord_cnt_cum_change_w1'] = (df[f'ord_cnt_cum_d2']-df[f'ord_cnt_cum_d7'])/df[f'ord_cnt_cum_d7']
        df[f'ord_cnt_cum_change_w2'] = (df[f'ord_cnt_cum_d7']-df[f'ord_cnt_cum_d14'])/df[f'ord_cnt_cum_d14']
        df[f'ord_cnt_cum_change_w3'] = (df[f'ord_cnt_cum_d14']-df[f'ord_cnt_cum_d21'])/df[f'ord_cnt_cum_d21']

        logging.info("add_fe_change_ratio is finished")
        return df    

    
    
    def add_fe_rgn_ratio(self, df):

        """
        조별 과거 주문비중 컬럼을 생성해서 피처로 이용
        """

        # 학습/예측시에는 삭제해야함
        df['ord_sum'] = df.groupby(['biz_ymd', 'center_dlvy'])['ord_cnt'].transform(sum)
        df['ord_ratio'] = (df['ord_cnt']/df['ord_sum'])

        n_list = [1, 2, 3, 5, 7, 10, 11, 14, 21]
        for n in n_list:

            df[f'ord_sum_d{n}'] = df.groupby(['center_dlvy', 'biz_hour'])['ord_sum'].shift(n)
            df[f'ord_ratio_d{n}'] = df.groupby(['center_dlvy', 'biz_hour'])['ord_ratio'].shift(n)        
            df[f'ord_cnt_by_ratio_d{n}'] = df[f'ord_sum_d{n}']*df[f'ord_ratio_d{n}']

            df[f'ord_sum_roll_d{n}'] = df.groupby(['center_dlvy', 'biz_hour'])['ord_sum_d1'].transform(lambda x: x.rolling(n,1).mean())
            df[f'ord_ratio_roll_d{n}'] = df.groupby(['center_dlvy', 'biz_hour'])['ord_ratio_d1'].transform(lambda x: x.rolling(n,1).mean())

        df['ord_ratio_w2_mean'] = df[['ord_ratio_d7', 'ord_ratio_d14']].mean(axis=1)
        df['ord_ratio_w3_mean'] = df[['ord_ratio_d7', 'ord_ratio_d14', 'ord_ratio_d21']].mean(axis=1)

        logging.info("add_fe_rgn_ratio is finished")
        return df    
    
    
    
    def add_fe_promotion(self, df):

        """
        프로모션 정보를 피처에 추가
        dataio 올라오는건 일특, 라이브커머스 등이며, PAPO는 기획전과 같이 계획성이 업로드됨.
        각각 추출후에 합친후에 피처로 활용. mkrs_gss_schema 변경 필요함.
        """
    
        promo = GCPRead().read_table_with_sql_file(sql_file_name='promo_by_region_fe.sql', 
                                                   start_date=self.start_date.to_date, 
                                                   SQL_PATH=self.sql_path)
        promo = promo.drop_duplicates()

        # 프로모션 별 시작, 끝 날짜밖에 없으므로 별도로 df 생성
        promo_df = pd.DataFrame()
        for i in promo.index:
            subset = promo.loc[i]
            start = str(subset.loc['period_from'])
            end = str(subset.loc['period_to'])

            sub_df = pd.Series(pd.date_range(start, end)).to_frame()
            sub_df.columns = ['promo_date']
            sub_df['promo_type'] = subset.loc['type']
            sub_df['promo_id'] = subset.loc['event_code']

            promo_df = promo_df.append(sub_df)

        promo_df = promo_df.groupby(['promo_date', 'promo_type'])['promo_id'].nunique().unstack().reset_index().fillna(0)
        promo_df.rename(columns={'promo_date':'biz_ymd'}, inplace=True)

        df = df.merge(promo_df, on=['biz_ymd'], how='left')

        logging.info("add_fe_promotion is finished")
        return df    
    
    
    
    def add_fe_mkt_event(self, df):

        """
        AR에서 별도로 입력하는 마케팅 변수 추가
        """
        sql = f"""
                select
                    cast(cast(ymd as timestamp) as date) as biz_ymd,
                    marketing_event as ar_mkt_yn,
                    dlvy_stop
                from bq-datafarm.data_science.gss_kurly_event_info_ar
                where cast(ymd as timestamp) >= '{self.start_date}'
                """

        mkt = GCPRead().read_sql_query(sql=sql)
        mkt['biz_ymd'] = pd.to_datetime(mkt['biz_ymd'])
        mkt = mkt.sort_values("biz_ymd")

        mkt['dlvy_stop'] = np.where(mkt['dlvy_stop'].str.contains("택배불가"), 1, 0)
        mkt['ar_live_yn'] = np.where(mkt['ar_mkt_yn'].str.contains("라이브"), 1, 0)
        mkt['ar_week_yn'] = np.where(mkt['ar_mkt_yn'].str.contains("위크"), 1, 0)
        mkt['ar_mkt_yn'] = np.where(mkt['ar_mkt_yn'] != '', 1, 0)

        df = df.merge(mkt, on='biz_ymd', how='left')
        logging.info("add_fe_mkt_event is finished")
        return df    
    
    
    def add_fe_push_laggging(self, df, max_lagging_n=15):
        
        """
        AR 푸시 기록을 피처로 추가        
        lagging 피처 및 rolling mean 피처 추가
        """
        # todo: 백업테이블 이관후 적용 필요
        push = GCPRead().read_table_with_sql_file(sql_file_name='push_cnt_by_center.sql',
                                                    start_date=self.start_date.to_date, 
                                                    end_date=self.end_date.to_date,
                                                    SQL_PATH=self.sql_path)

        push['biz_ymd'] = pd.to_datetime(push['biz_ymd'])                
        df = df.merge(push, on=['biz_ymd', 'biz_hour'], how='left')
        df['push_cnt'] = df['push_cnt'].fillna(0)
        
        # 누적 Push 주문수
        df['push_cnt_cum'] = df.groupby(['biz_ymd', 'center_dlvy'])['push_cnt'].cumsum()

        # days on the same hour
        for n in np.arange(1, max_lagging_n):

            # lagging values
            df[f'push_cnt_d{n}'] = df.groupby(['center_dlvy', 'biz_hour'])['push_cnt'].shift(n)
            df[f'ord_cnt_cum_d{n}'] = df.groupby(['center_dlvy', 'biz_hour'])['ord_cnt_cum'].shift(n)
        
        logging.info("add_fe_push_laggging is finished")
        return df
                 
        
        
    def train_predict(self, df):
                        
        """
        target: ord_cnt, ord_sales 
        type: AR 제공용(17시 이후 푸시X), 푸시 관련 없이 예측(일반적인 예측)
        TODO: 푸시 관련 테이블이 BQ이관 되기 전에는 AR 제공용은 제외
        """
    
        # 인덱스 지정
        df = df.set_index("biz_ymd")                

        # 센터별로 필터링후 학습/예측
        pred_res = pd.DataFrame()
        center_list = df.center_dlvy.drop_duplicates()
        target_encoder = ce.TargetEncoder(cols=['biz_hour'])
        drop_cols = ['ord_cnt', 'ord_cnt_purehit', 'center_dlvy', 'ord_ratio_ar', 'ord_ratio', 'ord_sum', 'ord_sales', 'ord_cnt_cum']        
        #drop_cols = ['ord_cnt', 'ord_cnt_purehit', 'center_dlvy', 'ord_ratio_ar', 'ord_ratio', 'ord_sum', 'ord_sales', 'ord_cnt_cum', 'push_cnt', 'push_cnt_cum']        
        
        # 날짜별, 센터별로 반복적으로 학습/예측 수행
        for date in pd.period_range(self.target_min_date.to_date, self.target_date.to_date):
            logging.info(f"{date}: model train & predict is started")

            for center in center_list:
                subset = df.query("center_dlvy == @center")
                train = subset[subset.index < str(date)] # ~학습기간
                pred = subset[subset.index == str(date)] # 예측대상 기간

                X_train = train.drop(drop_cols, axis=1)
                X_train['biz_hour_encode'] = target_encoder.fit_transform(train['biz_hour'], train['ord_cnt'])
                y_train_cnt = train[['ord_cnt']] # 주문수 예측
                y_train_sales = train[['ord_sales']] # 매출액 예측
                y_train_cnt_ar = train[['ord_cnt_purehit']] # 주문수(푸시제거) 예측

                X_pred = pred.drop(drop_cols, axis=1)
                X_pred['biz_hour_encode'] = target_encoder.transform(pred['biz_hour'])
                y_pred = pred[['biz_hour', 'center_dlvy', 'ord_cnt', 'ord_sales', 'ord_cnt_purehit']]

                # todo: 파라메터 최적화 -> 우선 기본 파라메터로 예측
                # 주문수 예측
                lgbm_model_cnt = LGBMRegressor()
                lgbm_model_cnt.fit(X_train, y_train_cnt)        
                y_pred['pred_cnt'] = lgbm_model_cnt.predict(X_pred).astype(int)
                y_pred['pred_cnt'] = np.where(y_pred['pred_cnt'] < 0, 0, y_pred['pred_cnt'])                                        

                # 매출액 예측
                lgbm_model_sales = LGBMRegressor()
                lgbm_model_sales.fit(X_train, y_train_sales)        
                y_pred['pred_sales'] = lgbm_model_sales.predict(X_pred).astype(int)
                y_pred['pred_sales'] = np.where(y_pred['pred_sales'] < 0, 0, y_pred['pred_sales'])                                    

                # 주문수 예측 (push 제거)
                lgbm_model_cnt_ar = LGBMRegressor()
                lgbm_model_cnt_ar.fit(X_train, y_train_cnt_ar)        
                y_pred['pred_cnt_purehit'] = lgbm_model_cnt_ar.predict(X_pred).astype(int)
                y_pred['pred_cnt_purehit'] = np.where(y_pred['pred_cnt_purehit'] < 0, 0, y_pred['pred_cnt_purehit'])                

                y_pred['pred_date'] = self.target_date.to_date # 예측수행일                    
                y_pred['pred_hour'] = self.target_date.hour # 예측수행 시간
                pred_res = pred_res.append(y_pred)

               
            logging.info(f"{date}: model train & predict is finished")            
        pred_res = pred_res.reset_index()
        pred_res['biz_ymd'] = pd.to_datetime(pred_res['biz_ymd'])
        pred_res['pred_date'] = pd.to_datetime(pred_res['pred_date'])

        logging.info("train_predict is done")  
        return pred_res
        
    
    def preprocess(self, **kwargs):
        
        # 예측 target 3개 필요
        ## 1)푸시포함 주문수 및 2)매출액, 3)미포함 주문수(AR)
        df = self.load_data()
        #df = self.apply_conti_ord_cnt(df) # 컨티 보정 임시 홀드
        df = self.change_to_center_dlvy(df)            
        df = self.get_pure_hit_cnt(df)                        
        df = self.make_future_dataset(df)
        ar = self.get_ar_goal()
        df = self.add_fe_ar_goal(df, ar)
        df = self.add_fe_daterelated(df)
        df = self.add_fe_holiday(df)
        df = self.add_fe_laggging(df)
        df = self.add_fe_change_ratio(df)
        df = self.add_fe_rgn_ratio(df)
        df = self.add_fe_promotion(df)
        df = self.add_fe_mkt_event(df)       
        df = self.add_fe_push_laggging(df) # todo: bak_빅쿼리 이관중. 임시로 3개월만 이용
        df = self._type_change(df)            
        pred_res = self.train_predict(df) 
         
        logging.info("preprocess is done")            
        return {'df': df, 'pred_res': pred_res}
                
        
    def postprocess(self, **kwargs): 
        
        """
        센터 구분 및 일부 수치 보정/보간하고 슬랙 메시지 송부
        """
        df = kwargs['df']
        pred_res = kwargs['pred_res']
                
        # 택배 토요일 0으로 처리
        pred_res = self.add_fe_mkt_event(pred_res)        
        pred_res = pred_res.drop(['ar_mkt_yn', 'ar_live_yn', 'ar_week_yn'], axis=1)        
        cols_to_zero = ['pred_cnt', 'ord_cnt', 'pred_cnt_purehit', 'ord_cnt_purehit', 'pred_sales', 'ord_sales']        
        for cols in cols_to_zero:            
            pred_res[cols] = np.where((pred_res['dlvy_stop'] == 1) & (pred_res['center_dlvy'] == 'CC03_NON_BASIC'), 0, pred_res[cols])
            
        # 예약 건수 추가 
        # df_together = self.add_reserved()
        # print('함께구매건수')
        # print(df_together)
        # pred_res = pred_res.rename(columns={'pred_cnt':'pred_cnt_temp'})
        # pred_df = pred_res.merge(df_together, how = 'left', on =['biz_ymd','biz_hour','center_dlvy'])
        # pred_df['together_ord_cnt'] = pred_df['together_ord_cnt'].fillna(0)
        # pred_df['together_unit_cnt'] = pred_df['together_unit_cnt'].fillna(0)
        # pred_df['pred_cnt'] = pred_df['pred_cnt_temp'] + pred_df['together_ord_cnt']
        # pred_df['ord_cnt'] = pred_df['ord_cnt'] + pred_df['together_ord_cnt']
        # pred_df = pred_df[['biz_ymd', 'biz_hour', 'center_dlvy', 'ord_cnt', 'ord_sales', 'ord_cnt_purehit', 'pred_cnt', 'pred_sales', 'pred_cnt_purehit', 'pred_date', 'pred_hour', 'dlvy_stop']]
        
        print(pred_res)
        logging.info("postprocess is done") 
        return pred_res

    
if __name__ == "__main__":

    parser = get_aargparser()
    args = parser.parse_args()
    # self = OrderPredictByCenter(**args.__dict__)
    cls = OrderPredictByCenter(**args.__dict__)
    cls.process()
