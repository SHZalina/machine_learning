import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import random 
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

'''
Используется рекурсивный подход в алгоритме:
1. Берем данные Х
2. Проходимся в цикле по всем признакам (атрибутам)
3. В каждом признаке (атрибуте) итерация по всем значениям
4. Для каждой пары признак/значение вычисляем энтропию и прирост информации
5. Находим лучшую пару, которая дает нам наибольший прирост информации
6. Разделяем данные по этой паре
7. Передаем разделенные данные в левый и правый узел дерева
8. Повтор
'''

def train_test_split(X, y, test_size=0.2, random_state=None):
    # получаем количество примеров в выборке
    n_samples = len(X)  

    # вычисляем количество примеров в тестовой выборке
    n_test = int(n_samples * test_size)  

    # если задано значение для random_state
    if random_state is not None:  
        # устанавливаем seed для генератора случайных чисел
        random.seed(random_state)  

    # создаем список индексов всех примеров
    indices = list(range(n_samples)) 
    # перемешиваем индексы случайным образом 
    random.shuffle(indices)  

    # выбираем первые n_test индексов для тестовой выборки
    test_indices = indices[:n_test] 
    # выбираем оставшиеся индексы для обучающей выборки
    train_indices = indices[n_test:]  

    X_train = X[train_indices]  # создаем массив признаков для обучающей выборки
    X_test = X[test_indices]  # создаем массив признаков для тестовой выборки
    y_train = y[train_indices]  # создаем массив меток классов для обучающей выборки
    y_test = y[test_indices]  # создаем массив меток классов для тестовой выборки

    # возвращаем четыре массива: признаки и метки для обучения и тестирования
    return X_train, X_test, y_train, y_test  

class DecisionTreeClassifier:
    # Конструктор класса
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, threshold=0.01):
        # Максимальная глубина дерева
        self.max_depth = max_depth
        # Минимальное количество объектов в узле, необходимое для его разделения
        self.min_samples_split = min_samples_split
        # Минимальное количество объектов в листе
        self.min_samples_leaf = min_samples_leaf
        # Порог для остановки рекурсии
        self.threshold = threshold
        # Дерево решений
        self.tree = None

    # Вычисление энтропии
    def entropy(self, y):
        # Получаем уникальные значения целевой переменной и их количество, аналог np.bincount(y)
        _, counts = np.unique(y, return_counts=True)
        # Вычисляем вероятности каждого класса
        p = counts / len(y)
        return -np.sum(p * np.log2(p))

    # Вычисление прироста информации
    def information_gain(self, X, y, feature_idx, threshold):
        # Вычисляем энтропию родительского узла
        parent_entropy = self.entropy(y)
        # Получаем индексы объектов, которые попадут в левое поддерево
        left_idx = X[:, feature_idx] < threshold
        # Получаем целевую переменную для левого поддерева
        left_y = y[left_idx]
        # Получаем целевую переменную для правого поддерева
        right_y = y[~left_idx]
        # Если в левом или правом поддереве нет объектов, то прирост информации равен 0
        if len(left_y) == 0 or len(right_y) == 0:
            return 0
        # Вычисляем энтропию для левого и правого поддеревьев
        left_entropy = self.entropy(left_y)
        right_entropy = self.entropy(right_y)
        # Вычисляем взвешенную сумму энтропий для левого и правого поддеревьев
        child_entropy = (len(left_y) / len(y)) * left_entropy + (len(right_y) / len(y)) * right_entropy
        # Вычисляем прирост информации
        return parent_entropy - child_entropy

    # Разделение узла
    def split_node(self, X, y):
        # Индекс лучшего признака
        best_feature_idx = None
        # Лучший порог
        best_threshold = None
        # Лучший прирост информации
        best_information_gain = -np.inf
        # Проходим по всем признакам
        for feature_idx in range(X.shape[1]):
            # Получаем уникальные значения признака
            thresholds = np.unique(X[:, feature_idx])
            # Проходим по всем порогам
            for threshold in thresholds:
                # Вычисляем прирост информации
                ig = self.information_gain(X, y, feature_idx, threshold)
                # Если прирост информации больше текущего лучшего, то обновляем значения
                if ig > best_information_gain:
                    best_information_gain = ig
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        # Получаем индексы объектов, которые попадут в левое поддерево
        left_idx = X[:, best_feature_idx] < best_threshold
        # Получаем индексы объектов, которые попадут в правое поддерево
        right_idx = ~left_idx
        # Возвращаем индекс лучшего признака, лучший порог, индексы объектов для левого и правого поддеревьев
        return best_feature_idx, best_threshold, left_idx, right_idx

    # Построение дерева решений
    def build_tree(self, X, y, depth=0):
        # Если достигнута максимальная глубина дерева или в узле меньше минимального количества объектов для разделения
        # или все объекты в узле относятся к одному классу, то создаем лист и возвращаем его значение
        if depth == self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()
        # Разделяем узел на два поддерева
        feature_idx, threshold, left_idx, right_idx = self.split_node(X, y)
        # Если в левом или правом поддереве меньше минимального количества объектов в листе, то создаем лист и возвращаем его значение
        #if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
        #    return np.bincount(y).argmax()
        # Создаем узел дерева
        tree = {}
        # Записываем индекс лучшего признака
        tree['feature_idx'] = feature_idx
        # Записываем лучший порог
        tree['threshold'] = threshold
        # Рекурсивно строим левое поддерево
        tree['left'] = self.build_tree(X[left_idx], y[left_idx], depth+1)
        # Рекурсивно строим правое поддерево
        tree['right'] = self.build_tree(X[right_idx], y[right_idx], depth+1)
        # Возвращаем узел дерева
        return tree

    # Обучение модели
    def fit(self, X, y):
        # Строим дерево решений
        self.tree = self.build_tree(X, y)

    # Предсказание классов
    def predict(self, X):
        # Если дерево решений не обучено, то выбрасываем исключение
        if self.tree is None:
            raise ValueError('Дерево решений не было обучено')
        # Создаем массив для предсказанных классов
        y_pred = np.zeros(len(X))
        # Проходим по всем объектам
        for i, x in enumerate(X):
            # Начинаем с корня дерева
            node = self.tree
            # Пока не достигнут лист, спускаемся по дереву
            while isinstance(node, dict):
                if x[node['feature_idx']] < node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            # Записываем предсказанный класс
            y_pred[i] = node
        # Возвращаем предсказанные классы
        return y_pred
    
    # Предсказание вероятностей классов
    def predict_proba(self, X):
        # Если дерево решений не обучено, то выбрасываем исключение
        if self.tree is None:
            raise ValueError('Дерево решений не было обучено')
        # Создаем массив для вероятностей классов
        proba = np.zeros((len(X), len(np.unique(y))))
        # Проходим по всем объектам
        for i, x in enumerate(X):
            # Начинаем с корня дерева
            node = self.tree
            # Пока не достигнут лист, спускаемся по дереву
            while isinstance(node, dict):
                if x[node['feature_idx']] < node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            # Получаем количество объектов каждого класса в листе
            counts = node.astype(int)
            # Вычисляем вероятности классов
            proba[i, :] = counts / np.sum(counts)
        # Возвращаем вероятности классов
        return proba
    
# Определяем функцию с именем accuracy_score, которая принимает два аргумента: y_true и y_pred.
def accuracy_score(y_true, y_pred):
    # Проверяем, равны ли i-ый элемент y_true и i-ый элемент y_pred.
    # Возвращаем отношение `correct` к общему количеству элементов в y_true.
    return np.sum(y_true == y_pred) / len(y_true)

# Определяем функцию с именем confusion_matrix, которая принимает три аргумента: y_true, y_pred и labels.
def confusion_matrix(y_true, y_pred, labels=None):
    # Проверяем, задан ли аргумент labels. Если нет, получаем список уникальных меток из y_true и сортируем его.
    if labels is None:
        labels = sorted(set(y_true))
    # Если labels был задан, сортируем его.
    else:
        labels = sorted(labels)
    # Вычисляем количество меток.
    n_labels = len(labels)
    # Инициализируем матрицу `counts` размером n_labels x n_labels, заполненную нулями.
    counts = [[0 for _ in range(n_labels)] for _ in range(n_labels)]
    # Цикл по индексам элементов списка y_true (предполагая, что длина y_true и y_pred одинакова).
    for i in range(len(y_true)):
        # Получаем истинную метку и предсказанную метку для i-го элемента.
        true_label = y_true[i]
        pred_label = y_pred[i]
        # Получаем индексы истинной метки и предсказанной метки в списке labels.
        true_idx = labels.index(true_label)
        pred_idx = labels.index(pred_label)
        # Увеличиваем на 1 соответствующий элемент матрицы `counts`.
        counts[true_idx][pred_idx] += 1
    # Возвращаем матрицу `counts` в виде массива NumPy.
    return np.array(counts)

# Загружаем набор данных digits
digits = load_digits()

# Извлекаем признаки и целевую переменную из набора данных
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание экземпляра классификатора дерева решений
dtc = DecisionTreeClassifier(max_depth=10, min_samples_split=2, min_samples_leaf=1, threshold=0.01)
# Обучение классификатора на тренировочных данных
dtc.fit(X_train, y_train)

# Предсказание меток классов на тренировочных данных:
y_train_pred = dtc.predict(X_train)

# Вычисление точности классификации на тренировочных данных:
train_accuracy = accuracy_score(y_train, y_train_pred)

# Предсказание меток классов на тестовых данных:
y_test_pred = dtc.predict(X_test)

# Вычисление точности классификации на тестовых данных:
test_accuracy = accuracy_score(y_test, y_test_pred)

# Вывод значения точности на обучающей и тестовой выборках
print("Точность на обучающей выборке:", train_accuracy)
print("Точность на тестовой выборке:", test_accuracy)

# Получение уникальных меток классов:
labels = np.unique(y_train)

# Вычисление матрицы ошибок на обучающей выборке:
cm = confusion_matrix(y_train, y_train_pred, labels=labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
disp.ax_.set_title("Матрица ошибок на обучающей выборке")
plt.show()

# Получение уникальных меток классов из тестовой выборки:
labels = np.unique(y_test)

# Вычисление матрицы ошибок на тестовой выборке:
cm = confusion_matrix(y_test, y_test_pred, labels=labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
disp.ax_.set_title("Матрица ошибок на тестовой выборке")
plt.show()

# Предсказание вероятностей принадлежности классам на тестовых данных:
y_test_prob = dtc.predict_proba(X_test)

# Создание маски для правильно предсказанных меток классов:
correct_mask = y_test == y_test_pred

# Выделение вероятностей правильно предсказанных меток классов:
correct_prob = y_test_prob[correct_mask]

# Выделение вероятностей неправильно предсказанных меток классов:
incorrect_prob = y_test_prob[~correct_mask]

# Создание фигуры с двумя графиками:
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Построение гистограммы вероятностей правильно предсказанных меток классов:
axs[0].hist(correct_prob.max(axis=1), bins=20, color='green')
axs[0].set_title('Гистограмма уверенностей правильно распознанных объектов')

# Построение гистограммы вероятностей неправильно предсказанных меток классов:
axs[1].hist(incorrect_prob.max(axis=1), bins=20, color='red')
axs[1].set_title('и ошибочных')

# Отображение графиков:
plt.show()