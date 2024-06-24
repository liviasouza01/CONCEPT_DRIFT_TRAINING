#olp for concept drift

from skmultiflow.data import (SEAGenerator, FileStream)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from olp_ml import OLP

window_size = 1000
max_iterations = 10

#stream = SEAGenerator()
stream = FileStream('https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/covtype.csv')

pre_train_size = 100
X_pre_train, y_pre_train = stream.next_sample(pre_train_size)

model = OLP()
model.fit(X_pre_train, y_pre_train)

X_dsel, y_dsel = stream.next_sample(window_size - pre_train_size)
X_dsel = np.concatenate([X_pre_train, X_dsel])
y_dsel = np.concatenate([y_pre_train, y_dsel])

model.fit(X_dsel, y_dsel)

accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

for i in range(max_iterations):
    X_batch, y_batch = stream.next_sample(window_size) #new batch

    X_dsel = np.concatenate([X_dsel, X_batch])
    y_dsel = np.concatenate([y_dsel, y_batch])

    model.fit(X_dsel, y_dsel)

    print(i)
    if i == i:
        X_validation, y_validation = stream.next_sample(window_size)

        y_pred = model.predict(X_validation)

        accuracy = accuracy_score(y_validation, y_pred)
        precision = precision_score(y_validation, y_pred, average='weighted')
        recall = recall_score(y_validation, y_pred, average='weighted')
        f1 = f1_score(y_validation, y_pred, average='weighted')
        print(accuracy)
        
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        print(accuracy_list)
        print(i)

mean_accuracy = np.mean(accuracy_list)
mean_precision = np.mean(precision_list)
mean_recall = np.mean(recall_list)
mean_f1 = np.mean(f1_list)

print("Médias das Métricas de Avaliação:")
print(f"Acurácia: {mean_accuracy}")
print(f"Precisão: {mean_precision}")
print(f"Recall: {mean_recall}")
print(f"F1-Score: {mean_f1}")
