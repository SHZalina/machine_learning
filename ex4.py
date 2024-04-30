import numpy as np
import matplotlib.pyplot as plt

mu_0, sigma_0 = 175, 7.5 # параметры для распределения футболистов
mu_1, sigma_1 = 190, 7.5 # параметры для распределения баскетболистов

N = 1000 # количество сгенерированных спортсменов на класс

football_players_height = np.random.normal(mu_0, sigma_0, N)
basketball_players_height = np.random.normal(mu_1, sigma_1, N)

# бинарный классификатор на основе порога для классификации спортсменов на футболистов и баскетболистов. для этого задали пороговое значение и сравниваем рост каждого спортсмена с этим значением.
threshold = 187.5 # пороговое значение для классификатора
print("Наш порог: ", threshold)

football_predictions = np.zeros_like(football_players_height)
basketball_predictions = np.zeros_like(basketball_players_height)

for i in range(len(football_players_height)):
    if football_players_height[i] > threshold:
        football_predictions[i] = 1

for i in range(len(basketball_players_height)):
    if basketball_players_height[i] > threshold:
        basketball_predictions[i] = 1

'''
TP - это количество спортсменов, которые действительно являются баскетболистами и были правильно классифицированы как таковые. 
TN - это количество спортсменов, которые действительно являются футболистами и были правильно классифицированы как таковые. 
FP - это количество спортсменов, которые на самом деле являются футболистами, но были неправильно классифицированы как баскетболисты. 
FN - это количество спортсменов, которые на самом деле являются баскетболистами, но были неправильно классифицированы как футболисты.
'''
TP = np.sum(basketball_predictions == 1)
TN = np.sum(football_predictions == 0)
FP = np.sum(football_predictions == 1)
FN = np.sum(basketball_predictions == 0)

# Вычисляем метрики
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)

# Вычисляем ошибки первого и второго рода
alpha = FP / (TN + FP)
beta = FN / (TP + FN)

print("TP: ", TP)
print("TN: ", TN)
print("FP: ", FP)
print("FN: ", FN)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-Score: ", f1_score)

print("Ошибка 1-го рода (alpha): ", alpha)
print("Ошибка 2-го рода (beta): ", beta)

# Вычисляем значения TPR и FPR для ROC-кривой, меняем порог и считаем
tpr_list = [] 
fpr_list = [] 
thresholds = np.linspace(0, 220, 100)
 
for threshold in thresholds: 
    football_predictions = np.zeros_like(football_players_height) 
    basketball_predictions = np.zeros_like(basketball_players_height) 
 
    for i in range(len(football_players_height)): 
        if football_players_height[i] > threshold: 
            football_predictions[i] = 1 
 
    for i in range(len(basketball_players_height)): 
        if basketball_players_height[i] > threshold: 
            basketball_predictions[i] = 1 
 
    TP = np.sum(basketball_predictions == 1) 
    TN = np.sum(football_predictions == 0) 
    FP = np.sum(football_predictions == 1) 
    FN = np.sum(basketball_predictions == 0) 
     
    tpr = TP / (TP + FN) 
    fpr = FP / (FP + TN) 
     
    tpr_list.append(tpr) 
    fpr_list.append(fpr) 
 
# Вычисляем площадь под ROC-кривой, используя метод прямоугольников
auc = 0.0
print('tpr_list', tpr_list)
print('fpr_list', fpr_list)

tpr_list_reversed = list(reversed(tpr_list))
fpr_list_reversed = list(reversed(fpr_list))

prev_height = tpr_list_reversed[0]
for i in range(len(fpr_list) - 1): 
    cur_height = tpr_list_reversed[i + 1] 
    auc += (fpr_list_reversed[i+1] - fpr_list_reversed[i]) * cur_height

# Выводим площадь под ROC-кривой на экран 
print("Площадь под ROC-кривой (AUC): ", auc, '\n')

# Строим ROC-кривую
plt.plot(fpr_list, tpr_list, color = (0.4, 0, 1))
plt.plot([0, 1], [0, 1], linestyle='--', color = (1, 0.5, 0))
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.show()

accuracy2 = []
for threshold in thresholds:
    football_predictions = np.zeros_like(football_players_height) 
    basketball_predictions = np.zeros_like(basketball_players_height) 
 
    for i in range(len(football_players_height)): 
        if football_players_height[i] > threshold: 
            football_predictions[i] = 1 
 
    for i in range(len(basketball_players_height)): 
        if basketball_players_height[i] > threshold: 
            basketball_predictions[i] = 1 
 
    TP = np.sum(basketball_predictions == 1) 
    TN = np.sum(football_predictions == 0) 
    FP = np.sum(football_predictions == 1) 
    FN = np.sum(basketball_predictions == 0) 
     
    accuracy2.append((TP + TN) / (TP + TN + FP + FN))
 

max_accuracy_threshold = thresholds[accuracy2.index(max(accuracy2))]
print("Порог для максимальной Accuracy: ", max_accuracy_threshold)

football_predictions = (football_players_height > max_accuracy_threshold).astype(int)
basketball_predictions = (basketball_players_height > max_accuracy_threshold).astype(int)

TP = np.sum(basketball_predictions == 1)
TN = np.sum(football_predictions == 0)
FP = np.sum(football_predictions == 1)
FN = np.sum(basketball_predictions == 0)

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)
alpha_error = FP / (TN + FP)
beta_error = FN / (TP + FN)

print("TP: ", TP)
print("TN: ", TN)
print("FP: ", FP)
print("FN: ", FN)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-score: ", f1_score)
print("Alpha error: ", alpha_error)
print("Beta error: ", beta_error)


