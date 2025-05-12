import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report,  accuracy_score, roc_auc_score,roc_curve
def wordcloudplot(ham_text,spam_text):
    ham_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(ham_text)
    spam_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_text)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(ham_wordcloud, interpolation='bilinear')
    plt.title('Word Cloud for Ham Messages')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(spam_wordcloud, interpolation='bilinear')
    plt.title('Word Cloud for Spam Messages')
    plt.axis('off')

    plt.show()

def change_format(X):
    lemmatizer=WordNetLemmatizer()
    corpus=[]
    for i in range(len(X)): 
        
        review = re.sub('[^a-zA-Z]',' ',X[i])
        review = review.lower()
        review = re.sub(' +', ' ', review).strip()
        review = review.split()
        review = [lemmatizer.lemmatize(word)for word in review if word not in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

def train_model(models,X_train,y_train,X_test,y_test):

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    
        print(f"Model: {model_name}\n, Accuracy_Score: {accuracy}")
        print(classification_report(y_test, y_pred))

def plot_roc_curve(fpr,tpr,roc_auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def roc_plot(models,X_test, y_test):

    for model_name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba) 
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        plot_roc_curve(fpr,tpr,roc_auc)
    
        print(f"ROC-AUC Score for {model_name}: {roc_auc:.4f}")