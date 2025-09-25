import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, roc_auc_score, roc_curve
import random as rd
import email.utils
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
import spacy
import pandas as pd
import json
from openai import OpenAI

# Fonction pour générer et afficher deux nuages de mots (word clouds)
# Un pour les messages "ham" (non spam), un pour les messages "spam"
def wordcloudplot(ham_text, spam_text):
    # Génère le nuage de mots pour ham
    ham_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(ham_text)
    # Génère le nuage de mots pour spam
    spam_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_text)

    # Crée une figure avec deux sous-graphes côte à côte
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(ham_wordcloud, interpolation='bilinear')  # Affiche l’image du nuage
    plt.title('Word Cloud for Ham Messages')              # Titre pour ham
    plt.axis('off')                                       # Cache les axes

    plt.subplot(1, 2, 2)
    plt.imshow(spam_wordcloud, interpolation='bilinear') # Affiche le nuage spam
    plt.title('Word Cloud for Spam Messages')            # Titre pour spam
    plt.axis('off')                                       # Cache les axes

    plt.show()  # Affiche la figure complète

# Fonction de nettoyage et préparation linguistique d’un corpus de textes
def change_format(X):
    lemmatizer = WordNetLemmatizer()   # Initialise lemmatiseur NLTK
    corpus = []                       # Liste pour stocker les textes nettoyés
    for i in range(len(X)):
        # Supprime tout sauf lettres, remplace par espace
        review = re.sub('[^a-zA-Z]', ' ', X[i])
        # Met en minuscule
        review = review.lower()
        # Remplace les multiples espaces par un seul, supprime espaces aux extrémités
        review = re.sub(' +', ' ', review).strip()
        # Sépare le texte en mots
        review = review.split()
        # Lemmatisation et suppression des stopwords NLTK
        review = [lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
        # Rejoint les mots nettoyés en une chaîne de caractères
        review = ' '.join(review)
        corpus.append(review)
    return corpus  # Retourne la liste des textes nettoyés

# Fonction d'entraînement et d’évaluation multiples de modèles sur un train/test split
def train_model(models, X_train, y_train, X_test, y_test):
    for model_name, model in models.items():
        model.fit(X_train, y_train)          # Entraîne le modèle
        y_pred = model.predict(X_test)       # Prédit sur le test
        accuracy = accuracy_score(y_test, y_pred)  # Calcule la précision

        # Affiche le nom du modèle, précision, rapport classification et matrice confusion
        print(f"Model: {model_name}\n, Accuracy_Score: {accuracy}")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

# Fonction pour tracer une courbe ROC (Receiver Operating Characteristic)
def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")  # Courbe ROC
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label="Random Guess")  # Diagonale (hasard)
    plt.xlabel("False Positive Rate")  # Taux de faux positifs (axe x)
    plt.ylabel("True Positive Rate")   # Taux de vrais positifs (axe y)
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()  # Affiche la figure

# Fonction pour générer et afficher la courbe ROC et score AUC pour plusieurs modèles
def roc_plot(models, X_test, y_test):
    for model_name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]          # Prédiction des probabilités classe 1
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)    # Calcule taux faux positifs et vrais positifs
        roc_auc = roc_auc_score(y_test, y_pred_proba)             # Calcule AUC
        plot_roc_curve(fpr, tpr, roc_auc)                         # Trace la courbe ROC
        print(f"ROC-AUC Score for {model_name}: {roc_auc:.4f}")  # Affiche score AUC

# Nettoyage de texte simple : passage en minuscule et suppression des caractères non alphabétiques (sauf espaces)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.split()  # Retourne une liste de mots

# Fonction basique de détection de spam sur base de mots-clés
def is_spam(email_text, spam_keywords):
    
    
    for keyword in spam_keywords:
        # Recherche mot-clé entier dans le texte
        if re.search(rf"\b{keyword}\b", email_text):
            return 1  # Texte marqué spam si un mot-clé est trouvé
    return 0      # Sinon non spam
def contient_trop_de_non_alpha(texte, seuil=0.3):
    # Trouve tous les caractères non alphabétiques (tout sauf lettres a-z et A-Z)
    non_alpha = re.findall(r'[^a-zA-Z]', texte)
    proportion = len(non_alpha) / len(texte) if len(texte) > 0 else 0
    # Retourne True si la proportion dépasse le seuil
    return proportion > seuil

# Méthode heuristique de détection spam basée sur proportion de mots spam, longueur du message
def is_spam_heuristic(email_text, spam_keywords):
    words = email_text.split()

    # Compte combien de mots correspondent aux mots spam
    spam_word_count = sum(1 for word in words if word in spam_keywords)
    spam_ratio = spam_word_count / len(words) if words else 0

    # Si beaucoup de mots spam (>20%), ou texte très court (<20 chars), ou s'il y a trop de caractères spéciaux
    if spam_ratio > 0.1 or len(email_text) < 20 or contient_trop_de_non_alpha(email_text,0.1):
        return 1
    return 0

# Fonction de parsing date sécurisée, retourne None en cas d’échec
def safe_parse(date_str):
    try:
        return parsedate_to_datetime(date_str)  # Tente de parser la date email
    except Exception:
        return None  # Retourne None si erreur

# Tokenizer basé sur spaCy : lemmatisation, suppression stopwords, suppression tokens non alphabétiques
def spacy_tokenizer(text):
    nlp = spacy.load("en_core_web_sm")  # Charge le modèle spaCy (ang)
    doc = nlp(text)                     # Analyse le texte
    # Retourne la liste lemmatisée, minuscule, sans stopwords ni ponctuation, uniquement mots alphabétiques
    return [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]

'''
client = OpenAI(api_key="sk-ijklmnopabcd5678ijklmnopabcd5678ijklmnop")

def generate_augmented_ham_email(original_email, n_variants: int = 1):
    prompt = f"""
Génère {n_variants} email(s) réaliste(s) au format JSON
avec exactement ces champs :
- "sender": chaîne
- "receiver": chaîne
- "date": chaîne au format RFC 2822 (exemple : Tue, 05 Aug 2008 20:28:00 -1200)
- "subject": chaîne
- "body": chaîne
- "label": entier égal à 0
- "urls": entier

Ces emails doivent sembler rédigés par un humain, être non-spam,
et ne pas avoir l'air générés par IA.

Conserve le style et le contexte de l'email suivant :
{json.dumps(original_email, ensure_ascii=False)}

Réponds uniquement avec une LISTE JSON valide et complète.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.8,
        )
        return json.loads(response.choices[0].message.content.strip())
    except json.JSONDecodeError as e:
        print("Erreur parsing JSON :", e)
        return []
    except Exception as e:
        print("Erreur API :", e)
        return []

def generate_augmented_spam_email(original_email, n_variants=1):
    prompt = f"""
Génère {n_variants} email(s) réaliste(s) au format JSON
avec exactement ces champs :
- "sender": chaîne
- "receiver": chaîne
- "date": chaîne au format RFC 2822 (exemple : Tue, 05 Aug 2008 20:28:00 -1200)
- "subject": chaîne
- "body": chaîne
- "label": entier égal à 1
- "urls": entier

Ces emails doivent être du phishing ou spam mais sembler être des hams,
et ne pas avoir l'air générés par IA.

Conserve le style et le contexte de l'email suivant :
{json.dumps(original_email, ensure_ascii=False)}

Réponds uniquement avec une LISTE JSON valide et complète.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.8,
        )
        return json.loads(response.choices[0].message.content.strip())
    except json.JSONDecodeError as e:
        print("Erreur parsing JSON :", e)
        return []
    except Exception as e:
        print("Erreur API :", e)
        return []

def augment_dataset_full(df, n_aug=1):
    augmented_data = []
    for _, row in df.iterrows():
        date_rfc2822 = email.utils.format_datetime(
            datetime.now(timezone(timedelta(hours=2)))
        )
        original_email = {
            "sender": row.get("sender", "unknown@example.com"),
            "receiver": row.get("receiver", "client@example.com"),
            "date": date_rfc2822,
            "subject": row.get("subject", ""),
            "body": row["body"],
            "label": int(row["label"]),
            "urls": int(row.get("urls", 0))
        }
        augmented_data.extend(generate_augmented_ham_email(original_email, n_variants=n_aug))
        augmented_data.extend(generate_augmented_spam_email(original_email, n_variants=n_aug))

    df_aug = pd.DataFrame(augmented_data)
    df_aug["label"] = df_aug["label"].astype(int)
    df_aug["urls"] = df_aug["urls"].astype(int)
    return pd.concat([df, df_aug], ignore_index=True)
'''


