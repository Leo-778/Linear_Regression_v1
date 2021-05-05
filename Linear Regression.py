import pandas as pd
import seaborn as sns
sns.set(context="notebook", style="whitegrid", palette="dark")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
'''
            在本练习的这一部分中，您将使用一个变量实现线性回归，以预测食品卡车的利润。
            假设你是一家餐饮连锁店的首席执行官，正在考虑在不同的城市开设一家新的分店。
            这个连锁店已经在不同的城市有卡车，你有来自这些城市的利润和人口数据。您希望
            使用此数据帮助您选择下一个要扩展到的城市。文件ex1data1.txt包含线性回归问
            题的数据集。第一栏是一个城市的人口，第二栏是该城市一辆食品车的利润。利润为
            负值表示亏损。
            ex1data1.txt  人口  利润

'''

'''
part1:导入数据
绘制散点图
'''
data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])#读取数据并赋予列名

data.plot(kind='scatter', x='population', y='profit', figsize=(12,8))#kind为图像种类scatter为散点图,x为横坐标数据，y为纵坐标数据，figsize为图像大小

#df = sns.lmplot(x='population',y='profit',data=data)
plt.show()
'''
part2:实现
1、计算cost
2、梯度下降
'''
data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])#读取数据并赋予列名,data为列表
data.insert(0, 'Ones', 1)#在data列表第0列插入一列命名one 数据全为1
cols = data.shape[1]#输出数组的行和列数,0表示行数,1表示列数 cols=3

x=data.iloc[:,:-1]#x是data除最后一列外所有列(population)
y = data.iloc[:,cols-1:cols]#y是data最后一列

X = np.matrix(x.values)#x转化为矩阵
y = np.matrix(y.values)#y转化为矩阵
theta = np.matrix(np.array([0,0]))#初始化theta
alpha = 0.01#学习速率
iters = 1500#迭代次数

def computeCost(X, y, theta):
    J=np.power(((X * theta.T) - y), 2)
    return np.sum(J)/(2*len(X))

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost
g, cost = gradientDescent(X, y, theta, alpha, iters)

computeCost(X, y, g)
print("g=",g)

predict1 = [1,3.5]*g.T
print("predict1:",predict1)
predict2 = [1,7]*g.T
print("predict2:",predict2)


x = np.linspace(data.population.min(), data.population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.population, data.profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()