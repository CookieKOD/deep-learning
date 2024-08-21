import streamlit as st
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Charger le modèle sauvegardé
model = joblib.load('svm_digit_model.pkl')

# Charger les données des chiffres
digits = datasets.load_digits()

# Sélectionner les données et les étiquettes
X, y = digits.data, digits.target

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Interface utilisateur avec Streamlit
st.title("Classification des Chiffres avec un Modèle SVM")

# Prédiction des résultats du test
y_pred = model.predict(X_test)

# Afficher les résultats de la matrice de confusion
st.subheader("Matrice de Confusion")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prédiction')
plt.ylabel('Réalité')

# Afficher la matrice de confusion dans Streamlit
st.pyplot(plt)

# Option supplémentaire : Prédiction sur une image donnée
st.subheader("Prédiction sur une Image")

# Sélectionner une image aléatoire à partir des données de test
random_index = np.random.randint(0, len(X_test))
image = digits.images[random_index]

# Redimensionner l'image pour la rendre plus nette
# Convertir l'image en un format PIL
image_pil = Image.fromarray(np.uint8(image * 16))
# Redimensionner l'image avec une interpolation bicubique
image_resized = image_pil.resize((128, 128), Image.BICUBIC)

# Convertir en image RGB (en couleur)
image_rgb = image_resized.convert("RGB")

# Centrer l'image avec du HTML
st.markdown(f"<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
st.image(image_rgb, caption=f"Image de test aléatoire (Index: {random_index})", width=300)
st.markdown("</div>", unsafe_allow_html=True)

if st.button("Prédire sur cette image"):
    prediction = model.predict([X_test[random_index]])
    st.write(f"Prédiction du modèle : {prediction[0]}")
    st.write(f"Vraie étiquette : {y_test[random_index]}")
