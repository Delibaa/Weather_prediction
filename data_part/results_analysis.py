import matplotlib.pyplot as plt
import numpy as np

# for example
x = np.arange(1, 17, 1)
y = np.array([4.00, 6.40, 8.00, 8.80, 9.22, 9.50, 9.70, 9.86, 10.00, 10.20, 10.32, 10.42, 10.50, 10.55, 10.58, 10.60])
z1 = np.polyfit(x, y, 3)  # 用3次多项式拟合，输出系数从高到0
p1 = np.poly1d(z1)  # 使用次数合成多项式
yvals = p1(x)

plt.plot(x, y, '*')
plt.plot(x, yvals)
plt.show()
