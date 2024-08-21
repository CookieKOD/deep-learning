import joblib
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split

# Charger les données
digits = datasets.load_digits()
X, y = digits.data, digits.target

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entraîner le modèle
model = svm.SVC()
model.fit(X_train, y_train)

# Sauvegarder le modèle localement
joblib.dump(model, 'svm_digit_model.pkl')
