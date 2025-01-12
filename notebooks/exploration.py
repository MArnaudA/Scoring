import marimo

__generated_with = "0.10.9"
app = marimo.App(width="full", app_title="Bank Marketing scoring project")


@app.cell
def marimo_import():
    import marimo as mo
    return (mo,)


@app.cell
def introduction(mo):
    mo.md(
        r"""
        #Bank Marketing scoring

        Bienvenue dans ce notebook concernant le scoring sur les clients d'une banque portugaise.

        Dans ce notebook, nous allons explorer les données du dataset "Bank Marketing" disponible sur le site du UC Irvine (https://archive.ics.uci.edu/dataset/222/bank+marketing).

        Les données proviennent d'une institution bancaire portugaise ayant mené des campagne publicitaire via des appels téléphoniques à des particuliers afin de savoir si ces clients seraient potentiellement intéressés par le produit : un dépôt à terme.

        Un dépôt à terme est un compte épargne bloqué (on ne peut pas accéder à l'argent déposé dessus) sur lequel est déposé un montant fixe au choix de l'utilisateur, et qui est valorisé avec un taux d'intérêt fixe après un certain temps.

        Le but de ce notebook sera de déterminer à partir des données si le client va souscrire à un dépôt à terme ou non.

        Nous allons d'abord explorer les données en décrivant les caratéristiques et les variables du jeu de données.  
        Ensuite nous effectuerons un nettoyage et traitement des données pour enfin entrainer un modèle de régression logistique.
        Nous proposerons un modèle simple que nous essaierons d'optimiser à travers les hyperparamètres du modèle.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1 : Analyse exploratoire

        ### 1-1 : Importation des données
        """
    )
    return


@app.cell
def data_importation():
    # Sur un notebook Jupyter, exécuter la commande suivante
    # !pip install ucimlrepo

    from ucimlrepo import fetch_ucirepo
    import pandas as pd

    # Récupération des données
    bank_marketing = fetch_ucirepo(id=222)

    # Récupération des données

    data = bank_marketing.data.original

    data
    return bank_marketing, data, fetch_ucirepo, pd


@app.cell
def _(mo):
    mo.md(r"""### 1-2 : Description des variables""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Nous allons décrire succintement les variables dont nous disposons. Les variables sont écrites sous la forme _VariableEnFrançais[_VariableEnAnglais_] avec le nom anglais de la variable étant celui du jeu de données d'origine.

        **1 - Age[age]** (Entier)  
        **2 - Job[job]** (Catégorielle) : Type de métier (Ex : admin, unemployed, blue-collar...)  
        **3 - Statut marital[marital]** (Catégorielle) : Statut marital de la personne (Ex : married, divorced, single)  
        **4 - Éducation[education]** (Catégorielle) : Niveau d'éducation (Ex: primary, secondary, tertiary, unknown)  
        **5 - Défaut[default]** (Catégorielle binaire) : Indique si la personne a un défaut de crédit (Ex : yes, no)  
        **6 - Solde[balance]** (Entier) : Moyenne annuelle du solde  
        **7 - Prêt habitation[housing]** (Catégorielle binaire) : Indique si la personne a pris un prêt habitation (Ex : yes, no)  
        **8 - Prêt personnel[loan]** (Catégorielle binaire) : Indique si la personne a pris un prêt personnel  
        **9 - Contact[contact]** (Catégorielle) : Type de la dernière communication avec la personne (Ex : telephone, cellular, unknown)  
        **10 - Jour[day]** (Entier) : Jour du mois du dernier contact de la personne  
        **11 - Mois[month]** (Entier) : Mois de l'année du dernier contact de la personne  
        **12 - Durée[duration]** (Entier) : durée du dernier contact en secondes  
        **13 - Nombre d'appels campagne[campaign]** (Entier) : Nombre de contacts effectués pour cette personne lors de cette campagne  
        **14 - pJour[pdays]** (Entier) : Nombre de jours entre le dernier appel de la précédente campagne publicitaire à laquelle a participé la personne, et le premier appel de la campagne actuelle  
        **15 - Précédent[previous]** (Entier) : Nombre de contacts effectués lors des campagnes précédentes  
        **16 - pRésultat[poutcome]** (Catégorielle) : Résultat de la précédente campagne (Ex : success, failure, other, unknown)  
        **17 - Résultat[y]** (Catégorielle binaire) : Résultat de cette campagne / Est-ce que le client a souscrit à un dépôt à terme ? (Ex : yes, no)
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""### 1-3 : Observation des valeurs manquantes et nettoyage des données""")
    return


@app.cell
def _(data):
    data.isna().sum()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        On remarque que la colonne poutcome contient beaucoup de valeurs manquantes. Il convient donc de ne pas la prendre en compte dans le jeu de données.  
        Il y a aussi un nombre de lignes significatifs pour lesquelles le "contact" n'est pas présent.  
        Pour nettoyer le jeu de données, nous allons :  
        - supprimer les variables "poutcome" et "contact",  
        - supprimer les lignes ayant une valeur nulle pour les variables "job" et "education".
        """
    )
    return


@app.cell
def _(data):
    # On en profite pour transformer les variables 'object' en variables catégorielles


    _data = data.replace({'yes':1, 'no':0})

    category_cols = _data.select_dtypes(include=['object']).columns

    _data_inter = _data.copy()
    _data_inter[category_cols] = _data[category_cols].astype('category')
    data_cleaned = (
        _data_inter
        .drop(columns = ['poutcome', 'contact'])
        .dropna()
        )

    data_cleaned
    return category_cols, data_cleaned


@app.cell
def _(mo):
    mo.md(
        """
        ### 1-5 : One-hot encoding

        Pour entrainer notre modèle, nous devons utiliser uniquement des variables numériques.  
        Nous devons donc procéder à un encodage des variables catégorielles. Nous allons passer par la méthode du one-hot encoding
        """
    )
    return


@app.cell
def _(data_cleaned, pd):
    _category_cols = data_cleaned.select_dtypes(include=['category']).columns
    _int64_cols = data_cleaned.select_dtypes(include=['int64']).columns

    _data_encoded = pd.get_dummies(data_cleaned[_category_cols])

    data_full = (pd
                 .concat([_data_encoded, data_cleaned[_int64_cols]], axis=1)
                )

    data_full
    return (data_full,)


@app.cell
def _(data_full):
    X = data_full.drop(columns = ['y'])
    y = data_full['y']
    return X, y


@app.cell
def _(mo):
    mo.md(r"""### 1-6 : Proportion de la variable cible dans le jeu de données et ajustement""")
    return


@app.cell
def _(y):
    nb_yes = y.value_counts()[1]
    nb_no = y.value_counts()[0]
    total = len(y)


    print("Nombre de 'yes' dans le jeu de données : ", nb_yes, "(", round(nb_yes/total*100, 2), "%)")
    print("Nombre de 'no' dans le jeu de données : ", nb_no, "(", round(nb_no/total*100, 2), "%)")
    return nb_no, nb_yes, total


@app.cell
def _(mo):
    mo.md(
        r"""
        On remarque un déséquilibre dans le jeu de données : 88,3% des personnes n'ont pas souscrit à l'offre proposée par la banque.  
        Avant de constituer nos ensembles d'entraînement et de test, nous allons essayer de voir si pallier ce déséquilibre améliore la performance du modèle. En effet, un déséquilibre de classe pourrait déteriorer la performance de la classe minoritaire. 
        Nous allons constituer deux jeux d'entrainement et de test : un jeu déséquilibré et un jeu rééquilibré avec la technique SMOTE  
        SMOTE (Synthetic Minority Over-sampling TEchnique) est une technique de sur-échantillonnage utilisée pour traiter le déséquilibre de classes dans les ensembles de données. Elle fonctionne comme suivant:  
        - **Identification des voisins les plus proches :**  Pour chaque exemple de classe minoritaire, SMOTE identifie ses k plus proches voisins (généralement k=5)  
        - **Création de nouveaux exemples :** Entre chaque exemple de la classe minoritaire et chacun de ses k plus proches voisins, SMOTE crée de nouveaux exemples synthétiques en interpolant aléatoirement le long du segment de ligne qui les relie
        """
    )
    return


@app.cell
def _(X, nb_no, nb_yes, y):
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE


    # Division en jeu d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sur-échantillonnage avec SMOTE
    smote = SMOTE(k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    nb_yes_smote = y_train_smote.value_counts()[1]
    nb_no_smote = y_train_smote.value_counts()[0]
    total_smote = len(y_train)

    print("Nombre de 'yes' dans le jeu d'entrainement déséquilibré : ", nb_yes)
    print("Nombre de 'no' dans le jeu d'entrainement déséquilibré : ", nb_no)
    print("Nombre de 'yes' dans le jeu d'entrainement traité avec SMOTE : ", nb_yes_smote)
    print("Nombre de 'no' dans le jeu d'entrainement traité avec SMOTE : ", nb_no_smote)
    return (
        SMOTE,
        X_test,
        X_train,
        X_train_smote,
        nb_no_smote,
        nb_yes_smote,
        smote,
        total_smote,
        train_test_split,
        y_test,
        y_train,
        y_train_smote,
    )


@app.cell
def _(X_train_smote):
    X_train_smote
    return


@app.cell
def _(X):
    X
    return


@app.cell
def _(X_test, X_train, X_train_smote, y_test, y_train, y_train_smote):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    model, model_smote = (LogisticRegression(max_iter=5000, tol=0.001, solver='saga', verbose=True),
                          LogisticRegression(max_iter=5000, tol=0.001, solver='saga', verbose=True)
                         )
    model.fit(X_train, y_train)
    model_smote.fit(X_train_smote, y_train_smote)

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)
    y_pred_smote = model_smote.predict(X_test)

    # Évaluation du modèle
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_smote = accuracy_score(y_test, y_pred_smote)
    print("Accuracy sans SMOTE", accuracy)
    print("Accuracy avec SMOTE", accuracy_smote)
    return (
        LogisticRegression,
        accuracy,
        accuracy_score,
        accuracy_smote,
        model,
        model_smote,
        y_pred,
        y_pred_smote,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        Nous remarquons ici que le modèle sans SMOTE obtient de meilleures performances comparé au modèle avec SMOTE. Cela est probablement dû au fait que les valeurs générées par SMOTE pour pallier le déséquilibre de classe introduisent du bruit.  
        Ce point montre donc toute l'importance de bien comprendre les données métier : générer de nouvelles données simplement à partir des plus proches voisins déteriore la performance.  
        Pour augmenter les performance du modèle, il faudrait introduire plus de données de personnes ayant souscrit au produit.
        """
    )
    return


@app.cell
def _(y_pred, y_test):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix


    confusion_mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')


    plt.xlabel('Prédictions')
    plt.ylabel('Vraies valeurs')
    plt.title('Matrice de Confusion')
    return confusion_mat, confusion_matrix, plt, sns


@app.cell
def _(confusion_matrix, plt, sns, y_pred_smote, y_test):
    confusion_mat_smote = confusion_matrix(y_test, y_pred_smote)
    sns.heatmap(confusion_mat_smote, annot=True, fmt='d', cmap='Blues')

    plt.xlabel('Prédictions')
    plt.ylabel('Vraies valeurs')
    plt.title('Matrice de Confusion')
    return (confusion_mat_smote,)


@app.cell
def _(mo):
    mo.md(r"""Bien que notre modèle sans SMOTE ait atteint une meilleure précision que le modèle avec SMOTE, on remarque en fait que ce premier parvient entièrement à classer les personnes n'ayant pas souscrit au produit, mais ne parvient quasiment pas à classer les personnes ayant souscrit au produit. Au contraire, malgré sa moins bonne performance générale, le deuxième modèle avec SMOTE parvient à classer la plupart des personnes ayant souscrit au produit.""")
    return


@app.cell
def _(accuracy, mo):
    mo.md(f"""Notre modèle de base atteint donc une précision de {round(accuracy*100,2)}%. Ce résultat est plutôt correct. Nous allons voir si l'on peut avoir de meilleurs résultats avec une modification des hyperparamètres.""")
    return


@app.cell
def _(LogisticRegression, X_test, X_train, y_train):
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    import numpy as np

    from sklearn.model_selection import cross_val_score

    # Nombre de folds pour la validation croisée
    k = 5

    _model = LogisticRegression(max_iter=5000, solver='saga')

    # Grille d'hyperparamètres à explorer
    param_grid = {'C': [0.001, 0.01, 0.1]}

    # Création du cross-validator
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Création du modèle avec recherche d'hyperparamètres
    grid_search = GridSearchCV(_model, param_grid, cv=skf)

    # Entraînement du modèle
    grid_search.fit(X_train, y_train)

    # Meilleurs hyperparamètres et meilleur score
    print("Meilleurs hyperparamètres :", grid_search.best_params_)
    print("Meilleur score :", grid_search.best_score_)

    # Utilisation du meilleur modèle pour faire des prédictions sur un nouvel ensemble de données (si nécessaire)
    best_model = grid_search.best_estimator_
    _y_pred = best_model.predict(X_test)
    return (
        GridSearchCV,
        StratifiedKFold,
        best_model,
        cross_val_score,
        grid_search,
        k,
        np,
        param_grid,
        skf,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        Nous n'obtenons pas de bien meilleures performances avec du fine-tuning sur une régression logistique (88,7% précision et C = 0.001)  
        Essayons un modèle Random Forest
        """
    )
    return


@app.cell
def _(
    StratifiedKFold,
    X_test,
    X_train,
    confusion_matrix,
    cross_val_score,
    plt,
    sns,
    y_test,
    y_train,
):
    from sklearn.ensemble import RandomForestClassifier

    # Créer un modèle Random Forest
    _model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Validation croisée stratifiée
    _cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    _model.fit(X_train, y_train)

    _y_pred = _model.predict(X_test)

    _scores = cross_val_score(_model, X_test, y_test, cv=_cv, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f)" % (_scores.mean(), _scores.std() * 2))

    _confusion_mat = confusion_matrix(y_test, _y_pred)
    sns.heatmap(_confusion_mat, annot=True, fmt='d', cmap='Blues')


    plt.xlabel('Prédictions')
    plt.ylabel('Vraies valeurs')
    plt.title('Matrice de Confusion')
    return (RandomForestClassifier,)


@app.cell
def _(mo):
    mo.md(
        r"""
        L'algorithme Random Forest performe mieux que la régression logistique.  
        En observant la matrice de confusion, on remarque en effet que plus de clients ayant réellement souscrit à l'offre (Vraies valeurs = 1) ont été correctement prédits.  
        En revanche, on obtient également plus d'erreur parmi les clients n'ayant pas souscrit à l'offre.

        Essayons d'améliorer la performance de cet algorithme avec du fine tuning.
        """
    )
    return


@app.cell
def _(
    GridSearchCV,
    RandomForestClassifier,
    StratifiedKFold,
    X_test,
    X_train,
    confusion_matrix,
    cross_val_score,
    grid_search,
    plt,
    sns,
    y_test,
    y_train,
):
    _param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }

    # Créer le modèle
    _skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    _rf = RandomForestClassifier()

    # Utiliser GridSearchCV pour trouver les meilleurs hyperparamètres
    _grid_search = GridSearchCV(estimator=_rf, param_grid=_param_grid, cv=_skf)
    _grid_search.fit(X_train, y_train)

    # Afficher les meilleurs hyperparamètres
    print(_grid_search.best_params_)
    _model = grid_search.best_estimator_

    _scores = cross_val_score(_model, X_test, y_test, cv=_skf, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f)" % (_scores.mean(), _scores.std() * 2))

    _y_pred = _model.predict(X_test)
    _confusion_mat = confusion_matrix(y_test, _y_pred)
    sns.heatmap(_confusion_mat, annot=True, fmt='d', cmap='Blues')


    plt.xlabel('Prédictions')
    plt.ylabel('Vraies valeurs')
    plt.title('Matrice de Confusion')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        En imposant des hyper-paramètres prédéfinis, on n'augmente pas la performance du modèle.  
        On obtient finalement une performance de 90% avec un Random Forest, ce qui est acceptable.
        """
    )
    return


if __name__ == "__main__":
    app.run()
