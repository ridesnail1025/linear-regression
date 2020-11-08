import numpy as np
import matplotlib.pyplot as plt#导入数据

x_data = np.asarray([0,0.2222222222222222,0.24074074074074073,0.3333333333333333,0.37037037037037035,0.4444444444444444,0.4962962962962963,
0.5740740740740741,0.9259259259259259,1])
y_data = np.asarray([0,0.22330097087378642,0.5825242718446602,0.19902912621359223,0.5485436893203883,0.3883495145631068,0.5388349514563107,0.529126213592233,1,0.6067961165048543])
#散点图
plt.scatter(x_data,y_data)


# 设置学习率
lr = 0.001
# 截距相当于xita0
b = 0
# 斜率k,相当于xita1
k = 0
# 最大迭代次数
epochs = 10000


# 自小二乘法,看代价函数（损失函数）定义的形式

def compute_error(b, k, x_data, y_data):
    totalError = 0
    for i in range(0, len(x_data)):
        # 差平方
        totalError += (y_data[i] - (k * x_data[i] + b)) ** 2
        pass
    return totalError / float(len(x_data)) / 2.0


# 梯度下降法
def gradient_descent_runner(x_data, y_data, b, k, lr, epochs):
    # 计算总数量
    m = float(len(x_data))
    # 循环epochs
    for i in range(epochs):
        b_grad = 0
        k_grad = 0
        # 计算梯度综合再求平均
        for j in range(0, len(x_data)):
            # 分别对西塔0和西塔1求骗到后的函数，
            b_grad += -(1 / m) * (y_data[j] - ((k * x_data[j]) + b))
            k_grad += -(1 / m) * x_data[j] * (y_data[j] - ((k * x_data[j]) + b))
        # 更新b和k
        b = b - (lr * b_grad)
        k = k - (lr * k_grad)

        # 没迭代5次，输出一次图像
    #         if i%5==0:
    #             print("epochs",i)
    #             plt.plot(x_data,y_data,'b.')
    #             plt.plot(x_data,k*k_data+b,'r')
    #             plt.show()
    #             pass
    #         pass
    return b, k
# gradient_descent_runner(x_data,y_data,b,k,lr,epochs)
print("Starting b={0},k={1},error={2}".format(b, k, compute_error(b, k, x_data, y_data)))
print("Running")
b, k = gradient_descent_runner(x_data, y_data, b, k, lr, epochs)

print("After{0} iterations b={1},k={2},error={3}".format(epochs, b, k, compute_error(b, k, x_data, y_data)))
p=k*0.2+b
print(p)
# 画图
plt.plot(x_data, y_data, 'b.')
# 也就是y的值k*x_data+b
plt.plot(x_data, x_data * k + b, 'r')

plt.show()