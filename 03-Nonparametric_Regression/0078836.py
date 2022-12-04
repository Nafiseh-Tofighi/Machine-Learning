import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# .csv file direction:
dir="./hw03_data_set.csv"

# read data into memory
data_set = np.genfromtxt(dir, delimiter = ",")

# get x and y values
x_train = data_set[1:151, 0]
x_test = data_set[151:, 0]

y_train = data_set[1:151, 1].astype(int)
y_test = data_set[151:, 1].astype(int)
N = data_set.shape[0]

bin_width = 0.37
minimum_value =  1.5
maximum_value = 1.5 + (0.37*10)
data_interval = np.linspace(minimum_value, maximum_value, 1601)


left_borders = np.arange(start = minimum_value,
                         stop = maximum_value,
                         step = bin_width)
right_borders = np.arange(start = minimum_value + bin_width,
                          stop = maximum_value + bin_width,
                          step = bin_width)

# regressogram
p_hat1=[]
for i in range(len(left_borders)):
  y=0
  b=0
  for idx,xi in enumerate(x_train):
    if (left_borders[i] < xi) & (xi <= right_borders[i]):
      b+=1
      y+=y_train[idx]
  
  p_hat1.append(y/b)
  
p_hat1=np.asarray(p_hat1)

plt.figure(figsize = (10, 6))
plt.plot(x_test, y_test,
         "r.", markersize = 10)
plt.plot(x_train,y_train,
         "b.", markersize = 10)
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [p_hat1[b], p_hat1[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [p_hat1[b], p_hat1[b + 1]], "k-")    
plt.show()


s=0
for i in range(len(left_borders)):
  for idx,xi in enumerate(x_test):
    if (left_borders[i] < xi) & (xi <= right_borders[i]):
      s+=(y_test[idx]-p_hat1[i])**2
rmse=math.sqrt(s/len(x_test))
print(f'Regressogram => RMSE is {rmse} when h is {bin_width}')


# mean smoother
w=0
y=0
p_hat2=[]

for x in data_interval:
  for idx,xi in enumerate(x_train):
    if abs((x-xi)/bin_width)<0.5:
      y+=y_train[idx]
      w+=1

  p_hat2.append(y/w)
  y=0
  w=0

p_hat2=np.asarray(p_hat2)

plt.figure(figsize = (10, 6))
plt.plot(x_test, y_test,
         "r.", markersize = 10)
plt.plot(x_train,y_train,
         "b.", markersize = 10)
plt.plot(data_interval, p_hat2, "k-")
plt.show()

p_hat_test2=[]

# sort the test data
indices = np.argsort(x_test)
x_test = x_test[indices]
y_test = y_test[indices]

# interpolate the test data
for idx, xi in enumerate(x_test):
  test_idx=np.where(data_interval<xi)[0][-1]
  y=p_hat2[test_idx+1]
#   if test_idx==len(data_interval)-1:
#     y=p_hat2[-1]
#   else:
#     m=(p_hat2[test_idx+1]-p_hat2[test_idx+1])/(data_interval[test_idx+1]-data_interval[test_idx])
#     y = m * (xi - data_interval[test_idx]) + p_hat2[test_idx]
  p_hat_test2.append(y)


# calculate the RMSE of test data
s=0
for idx,xi in enumerate(x_test):
    s+=(y_test[idx]-p_hat_test2[idx])**2
rmse=math.sqrt(s/len(y_test))
print(f'Regressogram => RMSE is {rmse} when h is {bin_width}')


#  kernel smoother
w=0
y=0
p_hat3=[]

for x in data_interval:
  for idx,xi in enumerate(x_train):
    u=(x-xi)/bin_width
    k=(1/math.sqrt((2*math.pi)))*math.exp((-1*(u**2))/2)
    y+=y_train[idx]*k
    w+=k

  p_hat3.append(y/w)
  w=0
  y=0

p_hat3=np.asarray(p_hat3)

plt.figure(figsize = (10, 6))
plt.plot(x_test, y_test,
         "r.", markersize = 10)
plt.plot(x_train,y_train,
         "b.", markersize = 10)
plt.plot(data_interval, p_hat3, "k-")
plt.show()

p_hat_test3=[]

# interpolate the test data
for idx, xi in enumerate(x_test):
  test_idx=np.where(data_interval<xi)[0][-1]
  # y=p_hat[test_idx+1]
  if test_idx==len(data_interval)-1:
    y=p_hat3[-1]
  else:
    m=(p_hat3[test_idx+1]-p_hat3[test_idx+1])/(data_interval[test_idx+1]-data_interval[test_idx])
    y = m * (xi - data_interval[test_idx]) + p_hat3[test_idx]
  p_hat_test3.append(y)


# calculate the RMSE of test data
s=0
for idx,xi in enumerate(x_test):
    s+=(y_test[idx]-p_hat_test3[idx])**2
rmse=math.sqrt(s/len(y_test))
print(f'Regressogram => RMSE is {rmse} when h is {bin_width}')