from pip._internal import main as pip
import importlib
import scipy

def import_with_auto_install(package):
    try:
        return importlib.import_module(package)
    except ImportError:
        pip(['install', package])
    return importlib.import_module(package)

np = import_with_auto_install('numpy')
pandas = import_with_auto_install('pandas')

path = "D:\Projects\Python\Classifier\datasets\wine.csv"
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

# Чисто для красоты, график для двух параметров, с кавычками в начале и в конце не работает
'''
from sklearn.preprocessing import scale
X_train_draw = scale(X_train[::, 0:2])
X_test_draw = scale(X_test[::, 0:2])

clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf.fit(X_train_draw, y_train)

x_min, x_max = X_train_draw[:, 0].min() - 1, X_train_draw[:, 0].max() + 1
y_min, y_max = X_train_draw[:, 1].min() - 1, X_train_draw[:, 1].max() + 1

h = 0.02

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h))

pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
pred = pred.reshape(xx.shape)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.figure()
plt.pcolormesh(xx, yy, pred, cmap=cmap_light)
plt.scatter(X_train_draw[:, 0], X_train_draw[:, 1], 
            c=np.ravel(y_train), cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.title("Score: %.0f percents" % (clf.score(X_test_draw, y_test) * 100))
plt.show()
'''