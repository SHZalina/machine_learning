'''
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
#только x -> t
#x, t = fetch_california_housing(return_X_y=True)
#print(x)
#print(t)
#Загрузка набора данных
dataset = load_boston()
X = dataset.data
T = dataset.target
'''
'''
x, t = sklearn.datasets.fetch_california_housing(return_X_y=True) # получаем датасет
''' 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Создание базисных функций (так не работает:(
def basis_functions(X):
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))
    poly = X
    for i in range(2, 9):
        poly = np.hstack((poly, np.power(X, i)))
    #trig = np.hstack((np.cos(X), np.sin(X)))
    #log = np.exp(X)
    return poly #np.hstack((poly, trig, log))

# Генерируем матрицу плана добавляя к матрице выборки столбец с 1
def matrix_plan(x): 
    mp = np.ones((np.size(x[:, 0]), np.size(x[0]) + 1))
    for i in range(np.size(x[:, 0])):
        for j in range(1, np.size(x[0]) + 1):
            mp[i][j] = x[i][j - 1]
    return mp

# Функция для вычисления среднеквадратичной ошибки
def mse(F, t, ves, alpha): 
    y = F @ ves 
    return 1/(F.shape[0]) * (np.sum((t - y) ** 2))

# Функция для вычисления градиента c регуляризацией
def gradient(F, t, ves, alpha): 
    return -(t.T @ F).T + (ves.T @ (F.T @ F)).T + alpha * ves.T

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
x = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
t = raw_df.values[1::2, 2]

# Стандартизация данных
for i in range (x.shape[1]):
    mean = np.mean(x[:, i], axis=0)
    std = np.std(x[:, i], axis=0)
    x[:, i] = (x[:, i] - mean) / std

# Разделение набора данных на тренировочные и тестовые части
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.2, shuffle = True)

F_train = matrix_plan(x_train)
F_test = matrix_plan(x_test)

# Инициализация начального приближения
w = np.random.normal(loc=0, scale=0.1, size=F_train.shape[1])

# Коэффициент регуляризации
#alpha = 10**(-10)
alpha = 0.1
#Шаг в градиентном спуске
learning_rate = 0.0001

# Градиентный спуск
num_iter = 1000 
error_history = []  #np.zeros(num_iter)
for i in range(num_iter): 
    grad = gradient(F_train, t_train, w, alpha) 
    w = w - learning_rate * grad 
    #error_history[i] = mse(F_train, t_train, w, alpha)
    error_history.append(mse(F_train, t_train, w, alpha))
# Критерий остановки по норме градиента 
#    if np.linalg.norm(grad) < 1e-4: 
#       break 
# Критерий остановки по норме разности последовательных приближений 
    if i > 0 and np.abs(error_history[i] - error_history[i-1]) < 1e-4: 
        break


# Вывод результата на экран
print("Ошибка на обучающей выборке:", mse(F_train, t_train, w, alpha)) 
print("Ошибка на тестовой выборке:", mse(F_test, t_test, w, alpha))

# График зависимости ошибки от номера итерации
plt.plot(np.arange(i+1), error_history[:i+1]) 
plt.xlabel("Номер итерации") 
plt.ylabel("Ошибка на обучающей выборке") 
plt.show()