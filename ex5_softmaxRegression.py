
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

class SoftmaxRegression:
    def __init__(self, lr, c, epochs):
        self.lr = lr
        self.c = c
        self.epochs = epochs
        
    def fit(self, X, Xv, y, yv):
        # X --> данные
        # y --> истинное/целевое значение
        # lr --> скорость обучения
        # c --> количество классов
        # эпох --> количество итераций

        m, n = X.shape   # m, n --> количество обучающих примеров, количество признаков 
        y_val_hot = self.one_hot(yv, self.c)
        
        # Инициализация весов и смещения: нормальное распределение с нулевым средним и небольшой дисперсией N(0, sigma)
        self.w = np.random.normal(loc=0, scale=0.1, size=(n, self.c))
        self.b = np.random.normal(loc=0, scale=0.1, size=self.c)

        prev_w = self.w 
        # Пустой списки для хранения потерь и точностей
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        for epoch in range(self.epochs):
            # Расчет гипотезы/прогноза
            z = X @ self.w + self.b
            y_hat = self.softmax(z)

            # One-hot encoding y
            y_hot = self.one_hot(y, self.c)

            # Расчет градиента потерь относительно w и b
            w_grad = (1/m) * np.dot(X.T, (y_hat - y_hot)) 
            b_grad = (1/m) * np.sum(y_hat - y_hot)

            # Обновление параметров
            curr_w = prev_w - self.lr * w_grad
            self.b = self.b - self.lr * b_grad

            # Расчет потерь, точности для обучающей выборки и добавление их в список
            #train_loss =  -np.mean(np.log(y_hat[np.arange(len(y)), y]))
            train_loss = self.cross_entropy(y_hat, y_hot, epsilon = 1e-9)
            self.train_losses.append(train_loss)
            train_acc = np.mean(y == np.argmax(y_hat, axis=1)) 
            self.train_accs.append(train_acc)

            # Расчет потерь, точности для валидационной выборки и добавление их в список
            z1 = Xv @ self.w + self.b
            val_y_hat = self.softmax(z1)
            # val_loss = -np.mean(np.log(val_y_hat[np.arange(len(yv)), yv]))
            val_loss = self.cross_entropy(val_y_hat, y_val_hot, epsilon = 1e-9)
            self.val_losses.append(val_loss)
            val_acc = np.mean(yv == np.argmax(val_y_hat, axis=1)) 
            self.val_accs.append(val_acc) 
            
            # Значение потери на каждой сотой итерации
            if epoch % 100 == 0:
               print(f"Epoch {epoch} ==> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f},  Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")
    
            # Критерий остановки по норме градиента 
            if np.linalg.norm(w_grad) < 1e-10: 
              break 
            
            # Критерий остановки по норме значений между последовательными приближениями меньше определенного значения
            if prev_w is not None and np.linalg.norm(curr_w - prev_w) <  1e-4:
              break
            
            prev_w = curr_w
            self.w =  prev_w
                
    # Функция кросс-энтропии: y выступает в роли своего рода переключателя, сохраняющего одну из частей выражения, и обнуляющего другую
    # Добавлена константа в логарифм для вычислительной устойчивости
    def cross_entropy(self, probs, y, epsilon = 1e-9):
        n = probs.shape[0]
        ce = -np.sum(y * np.log(probs + epsilon)) / n
        return ce
     
    # Функция предсказания: возвращается номер класса с наибольшей вероятностью
    def predict(self, X):
        z = X @ self.w + self.b
        y_hat = self.softmax(z)
        return np.argmax(y_hat, axis=1)

    # One-hot encoding
    @staticmethod
    def one_hot(y, c):
        # y--> метки из target-a
        # c--> количество классов
        y_hot = np.zeros((len(y), c))
        y_hot[np.arange(len(y)), y] = 1 # Ставим 1 для столбца, где стоит метка, и иcпользуем многомерное индексирование 
        return y_hot
    
    # Функция softmax
    # В результате работы получим матрицу m x c, где каждая строка — это прогнозные/вероятностые значения принадлежности одного наблюдения к каждому из с классов
    @staticmethod
    def softmax(z):
        exp = np.exp(z - np.max(z)) # вычитаем максимум z для избежания переполнения (см. пункт вычисления задания №5)
        for i in range(len(z)):
            exp[i] /= np.sum(exp[i])
        return exp
    
# Функция для подсчета точности
def accuracy(y, y_hat):
    return np.sum(y == y_hat) / len(y)


digits = load_digits()
x = digits.data
y = digits.target

# Стандартизация данных и удаление столбцов с нулевым стандартным отклонением
cols_to_delete = []
for i in range(x.shape[1]):
    mean = np.mean(x[:, i], axis=0)
    std = np.std(x[:, i], axis=0)
    if std == 0:
        cols_to_delete.append(i)
    else:
        x[:, i] = (x[:, i] - mean) / std

# Удаление столбцов с нулевым стандартным отклонением
if cols_to_delete:
    x = np.delete(x, cols_to_delete, axis=1)

# Разделение набора данных на тренировочные и валидационные части
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle = True)

# Создание объекта SoftmaxRegression, обучение модели
model = SoftmaxRegression(lr=0.9, c=10, epochs=1000)
model.fit(X_train, X_val, y_train, y_val)

# Предсказания для валидационной выборки
val_preds = model.predict(X_val)

# Подсчет точности для валидационной выборки
val_acc = accuracy(y_val, val_preds)

print("Validation accuracy:", val_acc)

fig = plt.figure()
plt.plot(model.train_losses, label="Train Loss")
plt.plot(model.val_losses, label="Val Loss")
plt.legend()
plt.show()

plt.plot(model.train_accs, label="Train Acc")
plt.plot(model.val_accs, label="Val Acc")
plt.legend()
plt.show()

