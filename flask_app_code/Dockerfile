# Utilisez une image de base Python
FROM python:3.8

# Répertoire de travail
WORKDIR /app

# Copiez le contenu actuel du répertoire courant dans le répertoire de travail de l'image
COPY . /app

# Installez les dépendances Python spécifiées dans le fichier requirements.txt
RUN pip install -r requirements.txt

# Exposez le port sur lequel l'application Flask fonctionnera
EXPOSE 5000

# Commande pour lancer l'application Flask
CMD ["python", "app.py"]
