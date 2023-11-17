# Importation des bibliothèques
import pandas as pd
from elasticsearch import Elasticsearch

# Création d'une instance Elasticsearch
es = Elasticsearch(['http://localhost:9200'])

# Chargement des données du fichier "movies.dat"
movies_df = pd.read_csv('./data/movies.dat', sep='::', engine='python', names=['MovieID', 'Title', 'Genres'], encoding='latin-1')

# Suppression des lignes avec des valeurs manquantes
movies_df.dropna(inplace=True)

# Gestion des doublons pour les films
movies_duplicates = movies_df[movies_df.duplicated(subset=['MovieID'], keep=False)]
if not movies_duplicates.empty:
    for index, row in movies_duplicates.iterrows():
        movie_data = {
            'MovieID': row['MovieID'],
            'Title': row['Title'],
            'Genres': row['Genres'].split('|')
        }
        doc_id = f"{movie_data['MovieID']}_duplicate_{index}"
        try:
            es.index(index='movies', id=doc_id, body=movie_data)
        except Exception as e:
            print(f"Erreur lors de l'indexation du document avec ID {doc_id} (doublon) dans movies: {str(e)}")
            
# Création de l'index pour les films
index_name = 'movies'  # Nom de l'index
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, ignore=400)
else:
    print(f"L'index {index_name} existe déjà.")

# Transformation et indexation des données dans Elasticsearch
for index, row in movies_df.iterrows():
    movie_data = {
        'MovieID': row['MovieID'],
        'Title': row['Title'],
        'Genres': row['Genres'].split('|')
    }
    doc_id = movie_data['MovieID']
    try:
        es.index(index=index_name, id=doc_id, body=movie_data)
    except Exception as e:
        print(f"Erreur lors de l'indexation du document avec ID {doc_id} dans {index_name}: {str(e)}")

# Chargement des données du fichier "ratings.dat"
ratings_df = pd.read_csv('./data/ratings.dat', sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='latin-1')

# Suppression des lignes avec des valeurs manquantes
ratings_df.dropna(inplace=True)

# Gestion des doublons pour les notations (ratings)
ratings_duplicates = ratings_df[ratings_df.duplicated(subset=['UserID', 'MovieID'], keep=False)]
if not ratings_duplicates.empty:
    for index, row in ratings_duplicates.iterrows():
        rating_data = {
            'UserID': row['UserID'],
            'MovieID': row['MovieID'],
            'Rating': row['Rating'],
            'Timestamp': row['Timestamp']
        }
        doc_id = f"{rating_data['UserID']}_{rating_data['MovieID']}_duplicate_{index}"
        try:
            es.index(index='ratings', id=doc_id, body=rating_data)
        except Exception as e:
            print(f"Erreur lors de l'indexation du document avec ID {doc_id} (doublon) dans ratings: {str(e)}")

# Conversion du type de colonne 'Timestamp' en datetime si nécessaire
ratings_df['Timestamp'] = pd.to_datetime(ratings_df['Timestamp'], unit='s')

# Création de l'index pour les notations (ratings)
index_name = 'ratings'  # Nom de l'index
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, ignore=400)
else:
    print(f"L'index {index_name} existe déjà.")

# Transformation et indexation des données dans Elasticsearch
for index, row in ratings_df.iterrows():
    rating_data = {
        'UserID': row['UserID'],
        'MovieID': row['MovieID'],
        'Rating': row['Rating'],
        'Timestamp': row['Timestamp']
    }
    doc_id = f"{rating_data['UserID']}_{rating_data['MovieID']}"
    try:
        es.index(index=index_name, id=doc_id, body=rating_data)
    except Exception as e:
        print(f"Erreur lors de l'indexation du document avec ID {doc_id} dans {index_name}: {str(e)}")

# Chargement des données du fichier "users.dat"
users_df = pd.read_csv('users.dat', sep='::', engine='python', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], encoding='latin-1')

# Suppression des lignes avec des valeurs manquantes
users_df.dropna(inplace=True)

# Gestion des doublons pour les utilisateurs
users_duplicates = users_df[users_df.duplicated(subset=['UserID'], keep=False)]
if not users_duplicates.empty:
    for index, row in users_duplicates.iterrows():
        user_data = {
            'UserID': row['UserID'],
            'Gender': row['Gender'],
            'Age': row['Age'],
            'Occupation': row['Occupation'],
            'Zip-code': row['Zip-code']
        }
        doc_id = f"{user_data['UserID']}_duplicate_{index}"
        try:
            es.index(index='users', id=doc_id, body=user_data)
        except Exception as e:
            print(f"Erreur lors de l'indexation du document avec ID {doc_id} (doublon) dans users: {str(e)}")
            
# Création de l'index pour les utilisateurs
index_name = 'users'  # Nom de l'index
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, ignore=400)
else:
    print(f"L'index {index_name} existe déjà.")

# Transformation et indexation des données dans Elasticsearch
for index, row in users_df.iterrows():
    user_data = {
        'UserID': row['UserID'],
        'Gender': row['Gender'],
        'Age': row['Age'],
        'Occupation': row['Occupation'],
        'Zip-code': row['Zip-code']
    }
    doc_id = user_data['UserID']
    try:
        es.index(index=index_name, id=doc_id, body=user_data)
    except Exception as e:
        print(f"Erreur lors de l'indexation du document avec ID {doc_id} dans {index_name}: {str(e)}")

indices_info = es.cat.indices()
print(indices_info)