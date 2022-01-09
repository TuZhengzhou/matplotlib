import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn import impute



titles = [
        ['1.sklear自带数据集————鸢尾花iris'],
        ['2.大数据血糖含量']
        ]

def self_JBtest(y):
    # 样本规模n
    n = y.size
    y_ = y - y.mean()
    """
    M2:二阶中心钜
    skew 偏度 = 三阶中心矩 与 M2^1.5的比
    krut 峰值 = 四阶中心钜 与 M2^2 的比
    """
    M2 = np.mean(y_**2)
    skew =  np.mean(y_**3)/M2**1.5
    krut = np.mean(y_**4)/M2**2

    """
    计算JB统计量，以及建立假设检验
    """
    
    JB = n*(skew**2/6 + (krut-3 )**2/24)
    print(skew, krut, JB)

    pvalue = 1 - stats.chi2.cdf(JB,df=2)
    print("偏度：",stats.skew(y),skew)
    print("峰值：",stats.kurtosis(y)+3,krut)
    print("JB检验：",stats.jarque_bera(y))
    return np.array([JB,pvalue])

# ************************************************
# 1.sklear自带数据集————鸢尾花iris
# ************************************************
# from sklearn.datasets import load_iris
# iris = load_iris()
# data = iris.data
# target = iris.target
# print(data)

# ************************************************
# 2.大数据血糖含量
# ************************************************
df = pd.read_csv('d_train.csv', encoding='gbk')
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(copy = False)

for feature in df.columns.values:
    if df[feature].dtypes == object:
        continue
    print(feature, df[feature].isnull().sum())
    # imputer.fit_transform(df[feature].values.reshape(-1,1))

    index = df[df[feature].notnull() == True].index
    # print(index)
    mean = df.loc[index, feature].mean()
    std  = df.loc[index, feature].std()

    # # 这两行代码解决 plt 中文显示的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.hist(df.loc[index, feature], bins=500, rwidth=0.618)
    plt.title(feature)
    plt.show()

    # print(df.loc[index, feature])
    __, v1 = stats.shapiro(df.loc[index, feature])
    print(v1)
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # data = scaler.fit_transform(df.loc[index, feature].values.reshape(-1,1))
    # print(data)
    # jb, pvalue = self_JBtest(data)
    # print(jb, pvalue)
    # __, v1 = stats.kstest(sorted(df.loc[index, feature]), 'norm', alternative='less', args=(mean, std))
    # __, v2 = stats.kstest(sorted(df.loc[index, feature]), 'norm', alternative='greater', args=(mean, std))
    # pvalue = v1
    # if v2 > v1:
    #     pvalue = v2
    # if(pvalue>0.01):
    #     print(f'{feature}的正态检验')
    #     print(pvalue)
