from sklearn.datasets import load_breast_cancer
from sklearn.datasets import  load_diabetes
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt


# carregamento do dataset de cancer
dataset_cancer = load_breast_cancer()
#print (dataset_cancer.feature_names)
#print (dataset_cancer.target_names)

# carregamento de dataset de codigos de diabetes
dataset_diabetes = load_diabetes()
#print ("\n******\n", dataset_diabetes.feature_names)
#print (dataset_diabetes.target)

X_train_can, X_test_can, y_train_can, y_test_can = train_test_split(dataset_cancer.data, dataset_cancer.target, stratify=dataset_cancer.target,random_state=42)

X_train_dia, X_test_dia, y_train_dia, y_test_dia = train_test_split(dataset_cancer.data, dataset_cancer.target, stratify=dataset_cancer.target, random_state=42)

training_accuracy = []
test_accuracy = []

for interception in [True, False]:
  regr = LinearRegression(fit_intercept=interception)
  regr.fit(X_train_dia, y_train_dia)
  training_accuracy.append(regr.score(X_train_dia, y_train_dia))
  test_accuracy.append(regr.score(X_test_dia, y_test_dia))

print(plt.plot(["Interc", "No Interc"],training_accuracy, label='Acuracia no conj. treino'))
print(plt.plot(["Interc", "No Interc"],test_accuracy, label='Acuracia no conj. teste'))
print(plt.ylabel('Acuracia'))
print(plt.xlabel('Fit Intercept'))
print(plt.legend())