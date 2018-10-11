from sklearn import neighbors
import numpy as np

X = np.array([[2.2,3.4], [3.9,2.9], [3.7,3.6], [4,4], [2.8,3.5], [3.5,1], [3.8,4], [3.1,2.5]])
y = np.array([45,55,91,142,88,2600,163,67])

X1 = np.array([[3.5,3.6]])

# for i in range(1, 15):
clf = neighbors.KNeighborsRegressor(3,p=2)
clf.fit(X,y)
Z = clf.predict(X1)
# score = clf.score(X1)
print(Z)


# a = 
# b = np.array([3.5,3.6])

# for i in range(0,len(a)):
#     c = np.linalg.norm(a[i]-b)
#     print(c)