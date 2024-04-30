import numpy as np
import matplotlib.pyplot as plt
import random

# ПОЛИНОМЫ ДО СТЕПЕНИ M
def plan_matrix(x, M : int):
    n = len(x)
    F = np.ones((n,1))
    x = np.array((x))[:, np.newaxis]
    for i in range(M):
        F = np.concatenate((F, x), axis=1)
    powers = range(M + 1)
    F = np.power(F, np.array([powers]))

    return F

#ОПРЕДЕЛЯЕМ БАЗИСНЫЕ ФУНКЦИИ (буду использовать полиномы до 10-ей степени, sin, cos, tan, exp, sqrt )
#И СОЗДАНИЕ ОБЩЕЙ МАТРИЦЫ ПЛАНА
def common_plan_matrix(x):
    F = plan_matrix(x, 10)
    length = len(x)
   
    sinX = np.sin(x).reshape((length, 1)) #(1000, ) --> (1000, 1)
    cosX = np.cos(x).reshape((length, 1))
    tanX = np.tan(x).reshape((length, 1))
    
    exp = np.exp(x).reshape((length, 1))
    sqrt = np.sqrt(x).reshape((length, 1))

    return np.concatenate([
        F, sinX, cosX, tanX,     
        exp, sqrt, 
    ], axis=1)

#ИЗ ОБЩЕЙ МАТРИЦЫ ПЛАНА ВЫБИРАЕМ СЛУЧАЙНЫЕ СТОЛБЦЫ И ФОРМИРУЕМ НОВУЮ МАТРИЦУ
def random_plan(F, M):
    new_matrix = F[:,0]
    new_matrix = new_matrix.reshape((F.shape[0], 1)) 

    index_func = [0] # [1 ... 1].T

    indexs = list(range(1, F.shape[1]))

    for i in range(M):
        index = random.choice(indexs) 
        new_matrix = np.concatenate([new_matrix, F[:, index].reshape((F.shape[0], 1))], axis=1)
        index_func.append(index)
        indexs.remove(index)
    
    return [new_matrix, index_func]


#ДЕЛЕНИЕ ДАННЫХ на train, validation и test части (0.6, 0.2, 0.2)
#делаем перестановку элементов массива, далее с помощью срезов разделяем массив на 3 части
def data_sets(x, t):
    #подсмотрела отсюда: https://stackoverflow.com/questions/67579166/shuffling-numpy-arrays-keeping-the-respective-values
    state = np.random.get_state()
    x_random = np.copy(x)  
    np.random.shuffle(x_random)
    
    np.random.set_state(state)
    t_random = np.copy(t)  
    np.random.shuffle(t_random)

    length = len(x)
    step1 = int(length * 0.6)
    step2 = int(length * 0.8)

    return [ x_random[:step1],  x_random[step1:step2], x_random[step2:], t_random[:step1], t_random[step1:step2], t_random[step2:]] # train, validation, test
    
#РЕГУЛЯРИЗАЦИЯ ОЦЕНОК
def func_w(F, alpha=0):
    f = F.T @ F
    I = np.eye (len(f))
    W = np.linalg.inv(f + alpha * I.T ) @ F.T
    return W

N = 1000
x = np.linspace(0, 1, N)
z = 20*np.sin(2*np.pi * 3 * x) + 100*np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

x_power = [f"x ** {i}" for i in range(1, 11)]

func_str = np.array([
    "1", * x_power, 
    "sinX", "cosX", "tanX", "exp", "sqrt"
])


train_x, val_x, test_x, train_t, val_t, test_t = data_sets(x, t) #делим набор данных

l = [1e-07, 1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000] #коэффициенты регуляризации

error0 = float('inf')
M0 = None
func0 = None
lambda0 = None
F0 = None
w0 = None

for M in range(2, len(func_str)):
    for alpha in l:
        try:
            F, funcs = random_plan(common_plan_matrix(train_x), M)
            w = func_w(F, alpha) @ train_t
            
            y = common_plan_matrix(val_x)[:,funcs]  @ w
            error = 1/len(val_x) * np.sum((val_t - y) ** 2)

            if error < error0:
                error0 = error
                M0 = M
                func0 = funcs
                lambda0 = alpha
                w0 = w

        except:
            ...

print('Количество лучших базисных ф-ий:', M0)
print('Набор лучших базисных функций:', func_str[func0][1:])
print('Лучший коэффициент регурялизации:', lambda0)
print('Ошибка на validation наборе данных:')
print(error0)

F_test = common_plan_matrix(test_x)[:,func0]
w_test = func_w(F_test, lambda0) @ test_t
y_test = F_test @ w0
print("Ошибка на test наборе данных:")
error = 1/len(test_x) * np.sum(np.abs(test_t - y_test) ** 2)
print(error)

F = common_plan_matrix(x)[:,func0]

W = w0
y0 = F @ W

plt.plot(x, z, 'g', label=('z(x)'))
plt.legend(loc="upper left")

plt.scatter(x, t,  s=1, color=(1, 0, 0, 0.5), label=(' t(x)'))
plt.legend(loc="upper left")

plt.plot(x, y0, label=('регрессия')) #график регрессии на лучших параметрах модели: используется лучший коэффициент регурялизации и набор лучших базисных функций
plt.legend(loc="upper left")

plt.show()
