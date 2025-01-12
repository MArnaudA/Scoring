# Bonjour et bienvenue sur mon projet de scoring de potentiels clients de banque.

# But

Le but de ce projet est d'analyser un jeu de données contenant les caractéristiques des clients d'une banque portugaise ayant répondu à un sondage téléphonique. 
Le but de ce sondage était de déterminer si le client était intéressé par l'offre de la banque : un dépôt d'argent avec intérêt.  
Ainsi, avec ce jeu de données, nous allons essayer de déterminer en fonction de ses caractéristiques si la personne a souscrit a l'offre de dépôt d'argent.

# Résumé de la démarche

Le travail se résume en quelques points : 
- Importation et nettoyage du jeu de données
- Exploration et description des variables
- Test de quelques modèles et fine-tuning pour obtenir la meilleure performance.

# Spécificités techniques

## Package de notebook : marimo

Pour ce projet, j'ai utilisé le package `marimo` comme notebook python. Il s'agit d'un notebook réactif présentant plusieurs avantages comparés à un simple notebook Jupyter. Voici les inconvénients de Jupyter notebook : 
- Cellules ordonnées : L'ordre des cellules dans un notebook est significatif. Le déplacement ou la suprpression d'une cellule peut avoir un impact sur le résultat final.
- Structure de fichier JSON : Les notebooks Jupyter sont stockés dans un format JSON, ce qui rend difficile la comparaison directe des modifications entre deux versions, et donc le versioning dans Git.
- Collaboration difficile : Lorsque plusieurs persones travaillent sur le même notebook, les conflits de fusion peuvent être difficles à résoudre. 

En réponse à ces problèmes, `marimo` propose les avantages suivants :
- Reproductibilité et fiabilité : `marimo` utilise des DAG pour définir ses dépendances entre cellules, ce qui permet de garantir une exécution dans le bon ordre et la reproductibilité dans le notebook. Ainsi, toute modification dans une cellule entraîne la mise à jour automatique des cellules dépendantes, évitant ainsi les erreurs et les incohérences. (Ce qui peut arriver sur un notebook Jupyter lorsqu'on exécute une première cellule, puis une deuxième cellule dépendante de la première, et qu'on modifie la première cellule).
- Compatibilité avec Git : `marimo` est conçu pour s'intégrer parfaitement dans un flux de travail Git, ce qui facilite grandement le versioning et la collaboration en équipe 
- Déploiement : il est possible de transformern le notebook en exécutable, ou application web intéractive, ce qui facilite le partage.
- Visualisations et UI riches : `marimo` propose une interface utilisateur interactive permettant de créer des applications web directement à partir du notebook. De plus, il est possible d'intégrer des visualisations dynamiques pour une meilleure compréhension des données et des résultats.

La principale raison pour laquelle j'ai décidé d'explorer `marimo` est pour la possibilité de versioner son notebook avec Git. Cela rend l'expérience de codage agréable tout en s'assurant que l'on peut revenir à une version précédente du code.

## Package de machine learning : scikit-learn

Pour l'entraînement du modèle, j'ai opté pour le package classique de machine learning, `scikit-learn`. En effet, `scikit-learn` implémente toutes les opérations essentielles au traitement des jeux de données, notamment la séparation en "folds" pour effectuer une validation croisée, la séparation du jeu d'entraînement et de test, ainsi que les principaux algorithmes de machine learning tels que la régression logistique ou le Random Forest.

# Reproduction

Pour reproduire ce code, veuillez procéder comme suivant :
- Copier le lien git présent sur la page
- Exécuter la commande suivante sur Git Bash : `git clone lien/du/github`
- Installer les packages nécessaires et dépendances (de préférence avec un environnement virtuel) : `pip install -r requirements.txt`

Pour démarrer le notebook marimo, il existe plusieurs manières : 
- Pour exécuter le code et avoir un aperçu utilisateur : `marimo run exploration.py`
- Pour accéder au code et pouvoir le modifier : `marimo edit exploration.py`


