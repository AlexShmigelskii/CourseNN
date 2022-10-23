import urllib
from urllib import request
import numpy as np

# fname = input()  # read file name from stdin
fname = 'https://stepic.org/media/attachments/lesson/16462/boston_houses.csv'
f = urllib.request.urlopen(fname)  # open file from URL
data = np.loadtxt(f, delimiter=',', skiprows=1)  # load data to work with

Y = data[:, 0]
# X = np.delete(data, 0, axis=1)

n, m = data.shape

X0 = np.ones((n, 1))
X = np.hstack((X0, np.delete(data, 0, axis=1)))

B = np.linalg.inv(X.T @ X) @ X.T @ Y

print(*B)
