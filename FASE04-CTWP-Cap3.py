import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


df = pd.read_csv(
    "/mnt/data/seeds_dataset.txt",
    header=None,
    delim_whitespace=True
)

df.columns = [
    "Area", "Perimeter", "Compactness", "KernelLength", "KernelWidth",
    "AsymmetryCoeff", "GrooveLength", "Variety"
]

mapping = {1: "Kama", 2: "Rosa", 3: "Canadian"}
df["VarietyName"] = df["Variety"].map(mapping)

print(df.head())


print("\nESTATÍSTICAS DESCRITIVAS:\n")
print(df.describe())


df.hist(figsize=(12, 10), bins=20)
plt.suptitle("Distribuição das Features")
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="VarietyName", y="Area")
plt.title("Boxplot da Área por Variedade")
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(df.iloc[:, :7].corr(), annot=True, cmap="Blues")
plt.title("Matriz de Correlação entre Features")
plt.show()


X = df.drop(columns=["Variety", "VarietyName"])
y = df["Variety"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


modelos = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "RandomForest": RandomForestClassifier(),
    "NaiveBayes": GaussianNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

resultados = {}

def treinar_avaliar(nome, modelo, scaled=True):
    print(f"\nTreinando modelo: {nome}")

    if scaled:
        modelo.fit(X_train_scaled, y_train)
        pred = modelo.predict(X_test_scaled)
    else:
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, pred)
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, pred))
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, pred))

    resultados[nome] = acc



treinar_avaliar("KNN", modelos["KNN"], scaled=True)
treinar_avaliar("SVM", modelos["SVM"], scaled=True)
treinar_avaliar("RandomForest", modelos["RandomForest"], scaled=False)
treinar_avaliar("NaiveBayes", modelos["NaiveBayes"], scaled=False)
treinar_avaliar("LogisticRegression", modelos["LogisticRegression"], scaled=True)

print("\nRESULTADOS INICIAIS:", resultados)


print("\n==== OTIMIZAÇÃO DE HIPERPARÂMETROS ====\n")

parametros = {
    "KNN": {"n_neighbors": [3, 5, 7, 9]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    "RandomForest": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
}

melhores_modelos = {}

for nome in parametros.keys():
    print(f"\n>> GridSearch para {nome}")

    if nome == "RandomForest":
        base_model = RandomForestClassifier()
        grid = GridSearchCV(base_model, parametros[nome], cv=5)
        grid.fit(X_train, y_train)
        pred = grid.predict(X_test)
    else:
        base_model = KNeighborsClassifier() if nome == "KNN" else SVC()
        grid = GridSearchCV(base_model, parametros[nome], cv=5)
        grid.fit(X_train_scaled, y_train)
        pred = grid.predict(X_test_scaled)

    acc = accuracy_score(y_test, pred)

    melhores_modelos[nome] = {
        "best_params": grid.best_params_,
        "accuracy": acc
    }

    print("Melhores parâmetros:", grid.best_params_)
    print("Acurácia otimizada:", acc)


print("\n=== DESEMPENHO APÓS OTIMIZAÇÃO ===")
print(pd.DataFrame(melhores_modelos).T)
