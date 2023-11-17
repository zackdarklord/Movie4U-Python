from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
import pandas as pd
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure secret key
es = Elasticsearch([{'host': 'elasticsearch', 'port': 9200}])  # Configuration de la connexion à Elasticsearch

occupations = {
    '0': 'Autre ou non précisé',
    '1': 'Universitaire/Éducateur',
    '2': 'Artiste',
    '3': 'Employé de bureau/Administrateur',
    '4': 'Étudiant/Étudiant diplômé',
    '5': 'Service client',
    '6': 'Médecin/Soignant',
    '7': 'Cadre/Manager',
    '8': 'Agriculteur',
    '9': 'Femme au foyer',
    '10': 'Élève de la maternelle à la 12e année',
    '11': 'Avocat',
    '12': 'Programmeur',
    '13': 'Retraité',
    '14': 'Ventes/Marketing',
    '15': 'Scientifique',
    '16': 'Indépendant',
    '17': 'Technicien/Ingénieur',
    '18': 'Commerçant/Artisan',
    '19': 'Chômeurs',
    '20': 'Écrivain',
}

valid_ages = {
    1: 'Moins de 18 ans',
    18: '18-24 ans',
    25: '25-34 ans',
    35: '35-44 ans',
    45: '45-49 ans',
    50: '50-55 ans',
    56: '56 ans ou plus'
}

def extract_genres(genres, all_genres):
    genre_list = genres.split('|')
    genre_dict = {genre: 1 if genre in genre_list else 0 for genre in all_genres}
    return pd.Series(genre_dict)

def extract_year_from_title(title):
    year_match = re.search(r'\((\d{4})\)', title)
    if year_match:
        return year_match.group(1)
    else:
        return None

def map_gender(gender):
    # Mapper le genre en 1 pour masculin et 0 pour féminin
    if gender == 'M':
        return 0
    else:
        return 1

def get_jeu_de_donnée():
    # Récupérez les données depuis les index Elasticsearch
    movies_data = es.search(index='movies', q='*:*', size=10000)
    users_data = es.search(index='users', q='*:*', size=10000)

    # Transformez les données Elasticsearch en DataFrames
    movies_df = pd.DataFrame([hit['_source'] for hit in movies_data['hits']['hits']])
    users_df = pd.DataFrame([hit['_source'] for hit in users_data['hits']['hits']])

    # Commencez une recherche de défilement
    scroll = es.search(index='ratings', q='Rating:[3 TO *]', scroll='2m', size=10000)

    # Initialisez une liste pour stocker les résultats
    ratings_data = []

    while len(scroll['hits']['hits']) > 0:
        ratings_data.extend(scroll['hits']['hits'])
        scroll = es.scroll(scroll_id=scroll['_scroll_id'], scroll='2m')

    # Transformez les données de notation en un DataFrame
    ratings_df = pd.DataFrame([hit['_source'] for hit in ratings_data])

    # Filtrer les notations supérieures ou égales à 3
    ratings_df = ratings_df[ratings_df['Rating'] >= 3]

    # Fusionner les DataFrames en utilisant les colonnes UserID et MovieID
    combined_data = pd.merge(ratings_df, movies_df, left_on='MovieID', right_on='MovieID')
    combined_data = pd.merge(combined_data, users_df, left_on='UserID', right_on='UserID')

    # Appliquez la fonction d'extraction de l'année pour créer la colonne 'Year'
    combined_data['Year'] = combined_data['Title'].apply(extract_year_from_title)

    # Utilisez la fonction `get_dummies` pour obtenir les colonnes de genre binaire
    genres = movies_df['Genres'].str.split('|')
    genre_dummies = pd.get_dummies(genres.apply(pd.Series).stack()).sum(level=0)
    combined_data = pd.concat([combined_data, genre_dummies], axis=1)

    # Appliquer la fonction de mappage du genre
    combined_data['Gender'] = combined_data['Gender'].apply(map_gender)

    # Sélectionnez les colonnes souhaitées dans le résultat final
    genre_columns = list(genre_dummies.columns)  # Obtenir les noms des colonnes de genre
    result_df = combined_data[['UserID', 'MovieID', 'Title', 'Year', 'Rating', 'Age', 'Gender', 'Occupation', 'Zip-code'] + genre_columns]

    # Trier le DataFrame
    result_df = result_df.sort_values(by=['Title', 'Year', 'Rating'], ascending=[True, True, True])

    return result_df

def get_movies():
    # Récupérez les données depuis les index Elasticsearch
    movies_data = es.search(index='movies', q='*:*', size=10000)

    # Transformez les données Elasticsearch en DataFrames
    movies_df = pd.DataFrame([hit['_source'] for hit in movies_data['hits']['hits']])

    return movies_df

dfMaster = get_jeu_de_donnée()
dfMovie = get_movies()

# Fonction pour recommender en fonction d'un utilisateur qui déjà des notes
def get_movie_recommendations(user_id, num_recommendations, combined_data):
    # Vérifiez si l'utilisateur existe dans les données
    if user_id not in dfMaster['UserID'].values:
        print("L'utilisateur {} n'existe pas dans les données.".format(user_id))
        return []

    # Sélectionnez les données d'entraînement et de test
    train_data, test_data = train_test_split(dfMaster, test_size=0.2)

    # Créez une matrice utilisateur-film
    user_movie_matrix = train_data.pivot_table(index='UserID', columns='MovieID', values='Rating')
    user_movie_matrix = user_movie_matrix.fillna(0)

    # Calculez la similarité entre utilisateurs
    user_similarity = cosine_similarity(user_movie_matrix)

    # Sélectionnez les films préférés de l'utilisateur
    user_ratings = user_movie_matrix.loc[user_id, :]

    # Créez une liste de films déjà notés par l'utilisateur
    rated_movies = user_ratings[user_ratings > 0].index

    # Créez une liste pour stocker les recommandations
    recommendations = []

    for movie_id in user_movie_matrix.columns:
        # Si l'utilisateur a déjà noté ce film, passez à l'itération suivante
        if movie_id in rated_movies:
            continue

        # Calculez la similarité entre l'utilisateur donné et les autres utilisateurs qui ont noté ce film
        similarity = user_movie_matrix[user_movie_matrix[movie_id] > 0].dot(user_ratings)

        # Triez les utilisateurs similaires par ordre décroissant de similarité
        sorted_users = similarity.sort_values(ascending=False)

        # Sélectionnez le premier utilisateur (le plus similaire) et ajoutez le film à la liste des recommandations
        if not sorted_users.empty:
            most_similar_user = sorted_users.index[0]
            # Ajoutez le "Title" au dictionnaire de recommandation
            recommendations.append({"MovieID": movie_id, "Similarity": sorted_users[most_similar_user], "Title": combined_data.loc[combined_data['MovieID'] == movie_id, 'Title'].iloc[0], "Genres": combined_data.loc[combined_data['MovieID'] == movie_id, 'Genres'].iloc[0]})

        # Si nous avons atteint le nombre souhaité de recommandations, sortez de la boucle
        if len(recommendations) >= num_recommendations:
            break

    # Triez les recommandations par score de similarité décroissant
    recommendations = sorted(recommendations, key=lambda x: x["Similarity"], reverse=True)

    return recommendations


# Fonction pour recommender en fonction d'un utilisateur qui a déjà noté des films
def get_movie_recommendations_users(user_age, user_occupation, user_genre, num_recommendations, combined_data):
    # Vérifiez si l'utilisateur existe dans les données
    user_exists = (dfMaster['Age'] == user_age) & (dfMaster['Occupation'] == user_occupation) & (dfMaster['Gender'] == user_genre)
    
    if not any(user_exists):
        print("L'utilisateur avec ces caractéristiques n'existe pas dans les données.")
        return []

    # Sélectionnez les données d'entraînement et de test
    train_data, test_data = train_test_split(dfMaster, test_size=0.2)

    # Créez une matrice utilisateur-film
    user_movie_matrix = train_data.pivot_table(index=['Age', 'Occupation', 'Gender'], columns='MovieID', values='Rating')
    user_movie_matrix = user_movie_matrix.fillna(0)

    # Calculez la similarité entre utilisateurs
    user_similarity = cosine_similarity(user_movie_matrix)

    # Sélectionnez les films préférés de l'utilisateur
    user_ratings = user_movie_matrix.loc[(user_age, user_occupation, user_genre), :]

    # Créez une liste de films déjà notés par l'utilisateur
    rated_movies = user_ratings[user_ratings > 0].index

    # Créez une liste pour stocker les recommandations
    recommendations = []

    for movie_id in user_movie_matrix.columns:
        # Si l'utilisateur a déjà noté ce film, passez à l'itération suivante
        if movie_id in rated_movies:
            continue

        # Calculez la similarité entre l'utilisateur donné et les autres utilisateurs qui ont noté ce film
        similarity = user_movie_matrix[user_movie_matrix[movie_id] > 0].dot(user_ratings)

        # Triez les utilisateurs similaires par ordre décroissant de similarité
        sorted_users = similarity.sort_values(ascending=False)

        # Sélectionnez le premier utilisateur (le plus similaire) et ajoutez le film à la liste des recommandations
        if not sorted_users.empty:
            most_similar_user = sorted_users.index[0]
            # Ajoutez le "Title" au dictionnaire de recommandation
            recommendations.append({
                "MovieID": movie_id,
                "Similarity": sorted_users[most_similar_user],
                "Title": combined_data.loc[combined_data['MovieID'] == movie_id, 'Title'].iloc[0],
                "Genres": combined_data.loc[combined_data['MovieID'] == movie_id, 'Genres'].iloc[0]
            })

        # Si nous avons atteint le nombre souhaité de recommandations, sortez de la boucle
        if len(recommendations) >= num_recommendations:
            break

    # Triez les recommandations par score de similarité décroissant
    recommendations = sorted(recommendations, key=lambda x: x["Similarity"], reverse=True)

    return recommendations



@app.route('/')
def home():
    if 'user_id' not in session or 'occupation' not in session:
        return redirect(url_for('login'))
    else:
        return redirect(url_for('profile'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None

    # Effacez la session précédente lors de l'accès à la page de connexion
    session.clear()

    if request.method == 'POST':
        user_id = request.form.get('user_id')
        occupation = request.form.get('occupation')

        try:
            # Rechercher l'utilisateur dans Elasticsearch
            user_info = es.search(index='users', body={
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"UserID": user_id}},
                            {"match": {"Occupation": occupation}}
                        ]
                    }
                }
            })

            if user_info['hits']['total']['value'] == 1:
                # Récupérez les informations de l'utilisateur à partir des résultats de la recherche
                user_data = user_info['hits']['hits'][0]['_source']
                gender = user_data.get('Gender', 'Inconnu')
                age = user_data.get('Age', 'Inconnu')
                zip_code = user_data.get('Zip-code', 'Inconnu')

                session['user_id'] = user_id
                session['occupation'] = occupation
                session['gender'] = gender  # Stockez le genre dans la session
                session['age'] = age  # Stockez l'âge dans la session
                session['zip_code'] = zip_code  # Stockez le code postal dans la session

                return redirect(url_for('profile'))
            else:
                error = "Invalid user credentials"
        except NotFoundError:
            return "Elasticsearch index 'users' not found"
        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template('login.html', error=error)

@app.route('/profile')
def profile():
    if 'user_id' in session and 'occupation' in session and 'gender' in session and 'age' in session and 'zip_code' in session:
        user_id = session['user_id']
        occupation = session['occupation']
        gender = session['gender']

        # Vérifiez si 'age' existe dans la session
        if 'age' in session:
            age = session['age']
            age_label = valid_ages.get(age, 'Inconnu')
        else:
            age_label = 'Âge non spécifié'

        zip_code = session['zip_code']

        # Initialisez la liste des films notés par l'utilisateur
        rated_movies = []
 
        # Maintenant, récupérez les films notés par l'utilisateur à partir de l'index "ratings" dans Elasticsearch.
        scroll = es.search(index='ratings', q=f"UserID:{user_id}", scroll='2m', size=10000)

        # Initialisez une liste pour stocker les résultats
        ratings_data = []

        while len(scroll['hits']['hits']) > 0:
            ratings_data.extend(scroll['hits']['hits'])
            scroll = es.scroll(scroll_id=scroll['_scroll_id'], scroll='2m')

        # Transformez les données de notation en un DataFrame
        user_ratings = pd.DataFrame([hit['_source'] for hit in ratings_data])

        for rating in user_ratings.itertuples():  # Utilisez itertuples pour parcourir les lignes du DataFrame
            movie_id = rating.MovieID
            rating_value = rating.Rating

            if rating_value == 1:
                rating_class = 'rating-1'
            elif rating_value == 2:
                rating_class = 'rating-2'
            elif rating_value == 3:
                rating_class = 'rating-3'
            elif rating_value == 4:
                rating_class = 'rating-4'
            elif rating_value == 5:
                rating_class = 'rating-5'
            else:
                rating_class = 'rating-unknown'

            # Recherchez les détails du film correspondant à l'ID du film dans l'index "movies"
            movie_info = es.search(index='movies', q=f"MovieID:{movie_id}", size=10000)

            # Vérifiez si des résultats ont été renvoyés
            if movie_info['hits']['total']['value'] > 0:
                # Récupérez le premier résultat (assumant qu'il n'y a qu'une seule correspondance par ID de film)
                movie_details = movie_info['hits']['hits'][0]['_source']
                movie_details['Rating'] = rating_value
                movie_details['RatingClass'] = rating_class

                leGenre = ""
                # Ajoutez la ligne suivante pour récupérer le genre du film
                for fromage in movie_details.get('Genres', 'Genres inconnus'):
                    leGenre+= ","+fromage
                movie_details['Genres'] = leGenre[1:]

                rated_movies.append(movie_details)

        return render_template('profile.html', user_id=user_id, occupation=occupation, occupations=occupations, gender=gender, age=age, age_label=age_label, zip_code=zip_code, rated_movies=rated_movies, movie_details=movie_details)

    return redirect(url_for('login'))


@app.route('/suggestions')
def suggestions():
    if 'user_id' in session and 'occupation' in session and 'gender' in session and 'age' in session and 'zip_code' in session:
        user_id = session['user_id']
        occupation = session['occupation']
        gender = session['gender']

        # Vérifiez si 'age' existe dans la session
        if 'age' in session:
            age = session['age']
            age_label = valid_ages.get(age, 'Inconnu')
        else:
            age_label = 'Âge non spécifié'

        zip_code = session['zip_code']

        return render_template('suggestions.html', user_id=user_id, occupation=occupation, occupations=occupations, gender=gender, age=age, age_label=age_label, zip_code=zip_code)

    return redirect(url_for('login'))

# Exemple de route pour les recommandations
@app.route('/get_recommendations')
def get_recommendations():
    try:
        # Où vous appelez la fonction get_movie_recommendations
        recommendations = get_movie_recommendations(int(session['user_id']), num_recommendations=10, combined_data=dfMovie)


        # Vérifiez si l'utilisateur n'existe pas dans les données
        if int(session['user_id']) not in dfMaster['UserID'].values:
            print("L'utilisateur {} n'existe pas dans les données.".format(session['user_id']))
            return []

        # Retournez les recommandations au format JSON
        return jsonify(recommendations)
    except Exception as e:
        print(f"Une erreur s'est produite : {str(e)}")
        return jsonify({"error": "Une erreur s'est produite lors de la récupération des recommandations."}), 500

@app.route('/get_recommendations_user')
def get_recommendations_user():
    try:
        # Vérifiez si 'age', 'occupation' et 'genre' existent dans la session
        if 'age' in session and 'occupation' in session and 'gender' in session:
            # Où vous appelez la fonction get_movie_recommendations_users
            g = 1 if session.get('gender') == 'F' else 0

            recommendations = get_movie_recommendations_users(
                int(session['age']),
                int(session['occupation']),
                g,
                num_recommendations=10,
                combined_data=dfMovie
            )

            # Vérifiez si l'utilisateur n'existe pas dans les données
            
            user_exists = (
                (dfMaster['Age'] == int(session['age'])) &
                (dfMaster['Occupation'] == int(session['occupation'])) &
                (dfMaster['Gender'] == g)
            )
            if not any(user_exists):
                print("L'utilisateur avec ces caractéristiques n'existe pas dans les données.")
                return jsonify({"error": "L'utilisateur avec ces caractéristiques n'existe pas dans les données."}), 404

            # Retournez les recommandations au format JSON
            return jsonify(recommendations)
        else:
            print("Les informations utilisateur nécessaires ne sont pas présentes dans la session.")
            return jsonify({"error": "Les informations utilisateur nécessaires ne sont pas présentes dans la session."}), 400
    except Exception as e:
        print(f"Une erreur s'est produite : {str(e)}")
        return jsonify({"error": "Une erreur s'est produite lors de la récupération des recommandations."}), 500


#recommendations = get_movie_recommendations(1, num_recommendations=10)
#(recommendations)
#print(dfMaster)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('occupation', None)
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user_id' in session and 'occupation' in session:
        return redirect(url_for('profile'))

    if request.method == 'POST':
        gender = request.form.get('gender')
        age_code = request.form.get('age')
        occupation_code = int(request.form.get('occupation'))
        zip_code = request.form.get('zip_code')

        age_label = valid_ages.get(age_code, 'Inconnu')

        # ... (votre code de validation)

        if not any(error[1] for error in get_flashed_messages()):
            new_user_id = es.search(index='users')['hits']['total']['value'] + 1
            new_user = {
                'UserID': new_user_id,
                'Gender': gender,
                'Age': age_label,
                'Occupation': str(occupation_code),
                'Zip-code': zip_code
            }
            es.index(index='users', id=new_user_id, body=new_user)

            session['user_id'] = new_user_id
            session['occupation'] = str(occupation_code)
            session['gender'] = gender
            session['age'] = age_label
            session['zip_code'] = zip_code

            return redirect(url_for('profile'))

    return render_template('inscription.html', occupations=occupations, valid_ages=valid_ages)  # Ajoutez valid_ages au contexte ici


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

