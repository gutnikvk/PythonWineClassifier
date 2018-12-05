# -*- coding: utf-8 -*-

from pip._internal import main as pip
import importlib

def import_with_auto_install(package):
    try:
        return importlib.import_module(package)
    except ImportError:
        pip(['install', '--user', package])
    return importlib.import_module(package)

np = import_with_auto_install('numpy')
pandas = import_with_auto_install('pandas')

path = "../datasets/wine.csv"
data = pandas.read_csv(path, delimiter=",")
X = data.values[::, 1:14]
y = data.values[::, 0:1]

sklearn = import_with_auto_install('sklearn.model_selection')
X_train, X_test, y_train, y_test = sklearn.train_test_split(X, y, test_size=0.1)
test = data.values[-1:, 1:14] # Вариант для проверки, последняя строка из датасета

test = [[13.4, 3.91, 2.48, 23, 102, 1.8, .75, .43, 1.41, 7.3, .7, 1.56, 750]] # Вариант для проверки, строка из датасета (сделал просто чтобы понять, в каком виде нужно представлять данные)
test = [[13.05, 1.77, 2.1, 17, 107, 3, 3, .28, 2.03, 5.04, .88, 3.35, 885]] # Вариант для проверки, строка из датасета

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
print(clf.predict(test))

_ = input("Press enter to exit...")