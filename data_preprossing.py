import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 5 #input
hidden_size = 500
num_classes = 2 #output
num_epochs = 5
batch_size = 100
learning_rate = 0.001




#1. data 가져오기
#train data가져오기
train = pd.read_csv("train.csv",  parse_dates=["기준_분기_코드"], engine='python', encoding = 'cp949') 
train.shape
#train=train.drop(['상권_구분_코드_명','상권_코드_명','서비스_업종_코드_명'], axis=1)

#test data가져오기
test = pd.read_csv("test.csv",  parse_dates=["기준_분기_코드"], engine='python', encoding = 'cp949')
test.shape
test=test.drop(['상권_구분_코드_명','상권_코드_명','서비스_업종_코드_명'], axis=1)




#2. 평균 매출 구하기
#train 주중
weekly_avg=train['주중_매출_금액']/train['점포수']
train['주중_매출_금액']=weekly_avg

#train 주말
weekday_avg=train['주말_매출_금액']/train['점포수']
train['주말_매출_금액']=weekday_avg

#test 주중
weekly_avg_t=test['주중_매출_금액']/test['점포수']
test['주중_매출_금액']=weekly_avg_t

#test 주말
weekday_avg_t=test['주말_매출_금액']/test['점포수']
test['주말_매출_금액']=weekday_avg_t




#3. min-max scaling - train

#상권구분코드
train.replace('A',1,inplace=True)
train.replace('D',2,inplace=True)
train.replace('R',3,inplace=True)
train.replace('U',4,inplace=True)

normalized_df=(train['상권_구분_코드']-train['상권_구분_코드'].min())/(train['상권_구분_코드'].max()-train['상권_구분_코드'].min())
train['상권_구분_코드']=normalized_df


#기준_분기_코드

normalized_bungi=(train['기준_분기_코드']-train['기준_분기_코드'].min())/(train['기준_분기_코드'].max()-train['기준_분기_코드'].min())
train['기준_분기_코드']=normalized_bungi


#상권_코드
normalized_sang=(train['상권_코드']-train['상권_코드'].min())/(train['상권_코드'].max()-train['상권_코드'].min())
train['상권_코드']=normalized_sang


#서비스_업종_코드
out_cs=train['서비스_업종_코드'].str.strip("CS") #서비스 코드 앞 CS제거
train['서비스_업종_코드']=out_cs

train['서비스_업종_코드']=pd.to_numeric(train['서비스_업종_코드']) #str -> int
train['서비스_업종_코드'].dtypes

normalized_service=(train['서비스_업종_코드']-train['서비스_업종_코드'].min())/(train['서비스_업종_코드'].max()-train['서비스_업종_코드'].min())
train['서비스_업종_코드']=normalized_service



#점포수 : num
normalized_num=(train['점포수']-train['점포수'].min())/(train['점포수'].max()-train['점포수'].min())
train['점포수']=normalized_num



#주중_매출_금액 : weekly -> 최대 1억, 최소 0
normalized_weekly=(train['주중_매출_금액']-0)/(10000000-0)
train['주중_매출_금액']=normalized_weekly


#주말_매출_금액 : weekday -> 최대 1억, 최소 0
normalized_weekday=(train['주말_매출_금액']-0)/(10000000-0)
train['주말_매출_금액']=normalized_weekday





#4. min-max scaling - test

#상권구분코드
test.replace('A',1,inplace=True)
test.replace('D',2,inplace=True)
test.replace('R',3,inplace=True)
test.replace('U',4,inplace=True)

normalized_df_t=(test['상권_구분_코드']-test['상권_구분_코드'].min())/(test['상권_구분_코드'].max()-test['상권_구분_코드'].min())
test['상권_구분_코드']=normalized_df_t


#기준_분기_코드
normalized_bungi_t=(test['기준_분기_코드']-test['기준_분기_코드'].min())/(test['기준_분기_코드'].max()-test['기준_분기_코드'].min())
test['기준_분기_코드']=normalized_bungi_t


#상권_코드
normalized_sang_t=(test['상권_코드']-test['상권_코드'].min())/(test['상권_코드'].max()-test['상권_코드'].min())
test['상권_코드']=normalized_sang_t


#서비스_업종_코드
out_cs_t=test['서비스_업종_코드'].str.strip("CS") #서비스 코드 앞 CS제거
test['서비스_업종_코드']=out_cs_t

test['서비스_업종_코드']=pd.to_numeric(test['서비스_업종_코드']) #str -> int
test['서비스_업종_코드'].dtypes

normalized_service_t=(test['서비스_업종_코드']-test['서비스_업종_코드'].min())/(test['서비스_업종_코드'].max()-test['서비스_업종_코드'].min())
test['서비스_업종_코드']=normalized_service_t

#점포수 : num
normalized_num_t=(test['점포수']-test['점포수'].min())/(test['점포수'].max()-test['점포수'].min())
test['점포수']=normalized_num_t


#주중_매출_금액 : weekly
normalized_weekly_t=(test['주중_매출_금액']-0)/(10000000-0)
test['주중_매출_금액']=normalized_weekly_t

#주말_매출_금액 : weekday
normalized_weekday_t=(test['주말_매출_금액']-0)/(10000000-0)
test['주말_매출_금액']=normalized_weekday_t


train.head()
test.head()

train.to_csv("train_scaling.csv",mode="w",encoding='euc-kr')
test.to_csv("test_scaling.csv",mode="w",encoding='euc-kr')
