import matplotlib.pyplot as plt
import numpy as np

trainmetrics = [-7.827264757844233, -6.539122052193318, -5.46885741580931, -4.860724952141639]
testmetrics = [-7.624574522874662, -6.11743100622369, -5.002220748941359, -4.422560242520135]

trainmetrics = np.round(trainmetrics, decimals=3)
testmetrics = np.round(testmetrics, decimals=3)

print (trainmetrics, testmetrics)

# x = [1,2,3,4]
x = np.linspace(0,1,100)
y = x**2

# plt.plot(x, trainmetrics, color='b', marker='^')
# plt.plot(x, testmetrics, color='r', marker='o')
# plt.xlabel('Log-Scale 10 # Sequences')
# plt.ylabel('Average Log-Likelihood')
# plt.legend(('Train LL', 'Test LL'))
# plt.title('#Sequences vs Avg. Log-Likelihood')
plt.plot(x,x)
plt.plot(x,y)
plt.show()