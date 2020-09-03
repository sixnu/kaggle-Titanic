import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
#模型
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb
#模型评估
from sklearn.metrics import roc_auc_score,mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,train_test_split
import sklearn.preprocessing as preprocessing

#读取数据
data_train=pd.read_csv('./train.csv')
data_test=pd.read_csv(./test.csv')

#合并数据
data_train['type']='train'
data_test['type']='test'
data_all=pd.concat([data_train,data_test],axis=0,ignore_index=True,sort=False)

#查看头部数据
data_all.head(1).T

#查看列名称
data_all.columns

#查看数据特征
data_all.info()

#查看数据类型
data_all.dtypes.sort_values()

#查看缺失
data_all.isnull().sum()

#查看训练集生存率
data_all['Survived'].sum()
#0.383838

#查看不同性别人员的生存率
data_all.groupby('Sex')['Survived'].mean().plot(kind='bar')

#查看不同舱位生存率
data_all.groupby(data_all.Cabin.str[0])['Survived'].mean().plot(kind='bar')


#数据清洗
#先对文本数据进行清洗
#Cabin 处理文法 
Cabin_mapping = {0: 0, 'A': 1,'B' : 2, 'C': 3,'D' : 4, "E": 5, "F": 6, "G":7, "T": 8 }
data_all['Cabin']=data_all['Cabin'].str[0].map(Cabin_mapping).fillna(0)
data_all=data_all.join(pd.get_dummies(data_all['Cabin']).add_prefix('Cabin_'))


#对Ticket进行处理
data_all.Ticket.str[0].value_counts().sort_index()

Ticket_mapping = {"A":1, 
                 "C": 2, "F": 3,
                 "L": 4, "P": 5,"S": 6, "W": 8,
                }
data_all['Ticket']=data_all['Ticket'].str[0].map(Ticket_mapping).fillna(0)

data_all=data_all.join(pd.get_dummies(data_all['Ticket']).add_prefix('Ticket_'))

#Embarked
#缺失只有一个
data_all['Embarked'].mode()
data_all['Embarked']=data_all['Embarked'].fillna('C')
#对数据进行分列
data_all=data_all.join(pd.get_dummies(data_all['Embarked']).add_prefix('Embarked_'))


#对性别的数据转换
data_all['Sex']=data_all.Sex.str.contains('female').apply(lambda x:1 if x  else 0 )

#对Name的数据转换
#提取姓名中称谓
data_all['Name']=data_all['Name'].map(lambda name:name.split(",")[1].split(".")[0].strip())

#直接转换成数字效果不佳，下一步转换为字母，并进行0-1数据转换
Name_mapping = {"Mr": 0, 
                 "Miss": 1, 
                 "Mrs": 2, 
                 "Master": 3,
                 "Lady": 4,  "Ms": 4,
                  "Dr": 4, "Rev": 4, "Col": 4, "Major": 4, "Mlle": 4,"the Countess": 4,
                 "Jonkheer": 4, "Don": 4, "Dona" : 4, "Mme": 4,"Capt": 4,"Sir": 4 }
data_all['Name']=data_all['Name'].map(Name_mapping).fillna(0)
#查看生存率
data_all.groupby('Name')['Survived'].size().plot(kind='bar')
#分列
data_all=data_all.join(pd.get_dummies(data_all['Name']).add_prefix('Name_'))


#Fare缺失值处理
#用'Cabin','Pclass' 聚合数据填充缺失
data_all.Fare.fillna(data_all.groupby(['Cabin','Pclass'])['Fare'].transform('median'),inplace=True)
#查看Fare数据累计百分比分布，对Fare数据进行分段
data_all.Fare.quantile([.1, .2, .3, .4, .5, .6, .7, .8,.9])
data_all.Fare.describe()

def fare_category(fr): 
    if fr <= 10:
        return 1
    elif fr <=10 and fr > 15:
        return 2
    elif fr <= 25 and fr > 15:
        return 3
    elif fr <= 35 and fr > 25:
        return 4
    elif fr <= 75 and fr >35:
        return 5
    elif fr <= 150 and fr > 75:
        return 6
    return 7

data_all['Fare_cat'] = data_all['Fare'].apply(fare_category) 

data_all=data_all.join(pd.get_dummies(data_all['Fare_cat']).add_prefix('Fare_cat_'))
#查看数据分布和生存率
data_all.groupby('Fare_cat')['Survived'].size().plot(kind='bar')
data_all.groupby('Fare_cat')['Survived'].mean().plot(kind='bar')


#填充缺失值Age  用机器学习预测值填充
#创建XAge_train训练集，取有age的数据
X_Age=data_all.loc[(data_all['Age'].notnull()),:]#提取非缺失的数据
X_Age=X_Age.drop(['Age','type','Survived','Embarked'],axis=1)
yAge_train=data_all.loc[data_all['Age'].notnull()==True,'Age']
print(X_Age.shape)
print(yAge_train.shape)

#创建XAge_test测试集，取没有AGE的数据
XAge_test=data_all.loc[data_all['Age'].isnull()==True,:]
XAge_test=XAge_test.drop(['Age','type','Survived','Embarked'],axis=1)
print(XAge_test.shape)

import xgboost as xgb
# 用Xgboost填充缺失值
def RFmodel(X_train,y_train,X_test):
    model_xgb= xgb.XGBRegressor(max_depth=5, colsample_btree=0.1, learning_rate=0.2, n_estimators=32, min_child_weight=2)
    model_xgb.fit(X_train,y_train)
    y_pre=model_xgb.predict(X_test)
    return y_pre

y_pred=RFmodel(X_Age,yAge_train,XAge_test)

#填充 age缺失
data_all.loc[data_all['Age'].isnull(),'Age']=y_pred.astype(int)

#年龄分段
data_all['Age'].quantile([.1, .2, .3, .4, .5, .6, .7, .8,.9])

def age_category(age):
    if age <= 16:
        return 1
    elif age <= 24 and age > 16:
        return 2
    elif age <= 30 and age > 24:
        return 3
    elif age <= 40 and age > 30:
        return 4
    elif age <= 55 and age > 40:
        return 5
    return 6
data_all['Age_cat'] = data_all['Age'].apply(age_category)

data_all=data_all.join(pd.get_dummies(data_all['Age_cat']).add_prefix('Age_'))


#生成新变量 家庭成员数量
data_all['FamilySize'] = data_all['SibSp'] + data_all['Parch'] + 1
data_all['FamilySize'].value_counts()

data_all.groupby('FamilySize')['Survived'].mean().plot(kind='bar')

def family_category(n):
    if n == 1:
        return 1
    elif n == 2 :
        return 2
    elif n == 3  :
        return 3
    elif n == 4  :
        return 4
    elif n >= 5  :
        return 5
data_all['FamilySize_cat'] = data_all['FamilySize'].apply(family_category) 
data_all=data_all.join(pd.get_dummies(data_all['FamilySize_cat']).add_prefix('FamilySize_cat_'))

#是否为单身
data_all['Alone']=data_all['FamilySize_cat'].apply(lambda x:1 if x==1 else 0)

#变量扩充
data_all['FareCat_Sex'] = data_all['Fare_cat']*data_all['Sex']
data_all['Pcl_Sex'] = data_all['Pclass']*data_all['Sex']
data_all['Pcl_Name'] = data_all['Pclass']*data_all['Name']
data_all['Age_cat_Sex'] = data_all['Age_cat']*data_all['Sex']
data_all['Age_cat_Pclass'] = data_all['Age_cat']*data_all['Pclass']
data_all['Title_Sex'] = data_all['Name']*data_all['Sex']
data_all['Age_Fare'] = data_all['Age_cat']*data_all['Fare_cat']


# 性别、舱位、兄弟和生存率特征
Sex_and_Survived_mean=data_all.groupby('Sex')['Survived'].mean()
Pclass_and_Survived_mean=data_all.groupby('Pclass')['Survived'].mean()
SibSp_and_Survived_mean=data_all.groupby('SibSp')['Survived'].mean()
Age_and_Survived_mean=data_all.groupby('Age_cat')['Survived'].mean()
Name_and_Survived_mean=data_all.groupby('Name')['Survived'].mean()
Parch_and_Survived_mean=data_all.groupby('Parch')['Survived'].mean()
Ticket_and_Survived_mean=data_all.groupby('Ticket')['Survived'].mean()
Embarked_and_Survived_mean=data_all.groupby('Embarked')['Survived'].mean()

data_all['Sex_and_Survived_mean']=data_all.loc[:,'Sex'].map(Sex_and_Survived_mean)
data_all['Pclass_and_Survived_mean']=data_all.loc[:,'Pclass'].map(Pclass_and_Survived_mean)
data_all['SibSp_and_Survived_mean']=data_all.loc[:,'SibSp'].map(SibSp_and_Survived_mean)

data_all['Age_and_Survived_mean']=data_all.loc[:,'Age_cat'].map(Age_and_Survived_mean)
data_all['Name_and_Survived_mean']=data_all.loc[:,'Name'].map(Name_and_Survived_mean)

data_all['Ticket_and_Survived_mean']=data_all.loc[:,'Ticket'].map(Ticket_and_Survived_mean)
data_all['Embarked_and_Survived_mean']=data_all.loc[:,'Embarked'].map(Embarked_and_Survived_mean)


#特征选择，选择相关系数在0.3及以上的变量 
#CustomCorrelationChooser 
from sklearn.base import TransformerMixin,BaseEstimator
class CustomCorrelationChooser(TransformerMixin,BaseEstimator):
    def __init__(self,response,cols_to_keep=[],threshold=None):
        self.response=response
        #保存响应变量
        self.threshold=threshold
        #保存阈值
        self.cols_to_keep=cols_to_keep 
        #初始化变量，并保存特征名
        
    def transform(self,X):
        return X[self.cols_to_keep] 
    #转换会选择合适的列
    
    def fit(self,X,*_):
        df=pd.concat([X,self.response],axis=1)
        self.cols_to_keep=df.columns[df.corr()[df.columns[-1]].abs()>self.threshold]
        self.cols_to_keep=[c for c in self.cols_to_keep if c in X.columns]
        return self


#创建训练集

#删除文本变量
data_all=data_all.drop(['type','Embarked','Cabin'],axis=1)

X=data_all.loc[0:890,:]
X=X.drop(['Survived'],axis=1)
Y=data_all.loc[0:890,'Survived']

#特征选择 相关系数0.3以上的变量
ccc = CustomCorrelationChooser(threshold=0.3,response=Y)
ccc.fit(X)
ccc.cols_to_keep
ccc.transform(X).head()

#特征扩充
from sklearn.preprocessing import PolynomialFeatures
poly= PolynomialFeatures(degree=2,include_bias=False,interaction_only=False)


#归一化
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = MinMaxScaler()
#scaler2 = StandardScaler()

#数据转换
X= scaler.fit_transform(poly.fit_transform(ccc.transform(X)))
X_test_mn=data_all.loc[891:,:].drop(['Survived'],axis=1)[ccc.cols_to_keep]
X_test_mn=scaler.fit_transform(poly.fit_transform(X_test_mn))

#划分测试和验证集
from sklearn.model_selection import cross_val_score,train_test_split                                        
X_train,X_test,y_train,y_test=train_test_split(X,Y,train_size=0.8)

print(",训练数据特征:",X_train.shape,
      ",验证数据特征:",X_test.shape,
     ",测试数据特征:",X_test_mn.shape)
print(",训练数据标签:",y_train.shape,
     ',验证数据标签:',y_test.shape)
     
#建模
import lightgbm as lgb

#数据转换
lgb_train = lgb.Dataset(X_train,label=y_train)
lgb_valid = lgb.Dataset(X_test,label=y_test, reference=lgb_train)

num_round = 150
params = {
    'boosting_type': 'gbdt',
    'objective':'binary',
    'metric': 'auc',
    'num_leaves': 60,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'sigmoid':1,
    'verbose': 0,
    'subsample': 0.8, 
    'n_estimators':2000
    
}

results = {}

lgbm = lgb.train(params,
                lgb_train, 
                num_boost_round= num_round, 
                valid_sets=[lgb_valid,lgb_train],
                valid_names=('validate','train'),
                early_stopping_rounds =100,
                evals_result= results,
                )
Y_pred=lgbm.predict(X_test.astype('float64'))

#查看验证集得分
accuracy_score(y_test, np.array(Y_pred)>0.5)

#预测结果
lgbm_ypred = lgbm.predict(X_test_mn)

# 测试集输出
sub = pd.DataFrame()
sub['PassengerId'] = data_test.PassengerId
sub['Survived']=lgbm_ypred
sub['Survived']=sub['Survived'].apply(lambda x: 1 if x>0.5 else 0)
sub.to_csv('lgbm_predictions.csv', index=False)
print(sub['Survived'].describe())
print(sub['Survived'].sum())
#提交后的成绩0.8066
