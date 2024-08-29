import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv("Expresso_churn_dataset.csv" )

    # Afficher des informations générales sur l'ensemble de données
st.write("Aperçu des données")
st.write(data)

    # Créer un rapport de profilage de pandas
st.write("Rapport de profilage des données")



st.write("Rapport de profilage des données")
profile = ProfileReport(data, title="Profilage des données", explorative=True)
st_profile_report(profile)

    # Sauvegarder le rapport dans un fichier HTML
profile.to_file("rapport_profilage.html")
st.success("Rapport de profilage sauvegardé en tant que 'rapport_profilage.html'")


    # Gérer les valeurs manquantes
st.write("Valeurs manquantes")
st.write(data.isnull().sum())

    # Supprimer les doublons
st.write(data.drop_duplicates())

    # Encoder les caractéristiques catégorielles

    # Encoder les colonnes catégorielles
categorical_cols = data.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

    # Remplir les valeurs manquantes par la médiane pour les colonnes numériques
for col in data.select_dtypes(include=['float64']).columns:
    data[col].fillna(data[col].median(), inplace=True)

    # Afficher des informations sur les données après le prétraitement
st.write("Informations sur les données après le prétraitement")
st.write(data.info())
