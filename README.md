# Docker
Ce projet utilise Docker pour faciliter le déploiement et la gestion des dépendances.
Voici comment configurer et utiliser Docker pour ce projet.

### Prérequis
Assurez-vous d’avoir Docker et Docker Compose installés sur votre machine.
Vous pouvez les télécharger ici :
- [Docker](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Configuration
Le projet utilise le fichier Dockerfile pour construire l’image de l’application.
Toutes les dépendances Python du projet doivent être listées dans le fichier requirements.txt.
Ce fichier est copié dans l’image, et les dépendances sont installées
 lors de la construction de l’image.

### Construction de l'image
Pour construire l’image Docker,
 utilisez la commande suivante dans le répertoire racine du projet (là où se trouve le Dockerfile) :

```bash
docker build -t nom_de_votre_image .
```
Pour exécuter l’image :
```bash
docker run nom_de_votre_image
```
#### Privilèges administrateur
Selon la configuration de votre machine, il se peut que vous
 deviez exécuter les commandes Docker avec des privilèges administrateur.
Vous pouvez le faire en utilisant sudo, mais ce n’est pas recommandé dans la plupart des cas :
```bash
sudo docker build -t nom_de_votre_image .
```
```bash
sudo docker run nom_de_votre_image
```
