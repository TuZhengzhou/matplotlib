# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# from scipy import stats
# from sklearn import impute
# # x = np.linspace(-15, 15, 9)
# # x = [-15, 15, -3, -6, -9, -12, 3, 6, 9, 12, 0]
# x = np.linspace(-15, 15, 11)
# print(x)
# print(stats.kstest(x, 'norm'))
# # (0.44435602715924361, 0.038850142705171065)

##生成1000个服从N（0,1）的随机数
import numpy as np
np.random.seed(0)
norm_Data = np.random.normal(0,1,100000)
#生成100个服从lambda=10的指数分布exp(10)
exp_Data = np.random.exponential(scale=0.1, size=100000) #scale=1/lambda
 
 
import scipy.stats as stats
print(stats.shapiro(norm_Data))
##输出(统计量JB的值,P值)=(0.28220016508625245, 0.86840239542814834)，P值>指定水平0.05,接受原假设，可以认为样本数据在5%的显著水平下服从正态分布
 
print(stats.shapiro(exp_Data)) 
##输出(统计量JB的值,P值)=(1117.2762482645478, 0.0)，P值<指定水平0.05,拒绝原假设，认为样本数据在5%的显著水平下不服从正态分布