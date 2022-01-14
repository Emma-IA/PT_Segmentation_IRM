# Segmentation d'IRM cardiaque

Dans le domaine de l’imagerie médicale, le diagnostic d’une image cardiaque par résonance magnétique nécessite parfois l’intervention de radiologues qui doivent délimiter manuellement les différentes structures du cœur, pour en extraire des informations fonctionnelles telles que le volume cardiaque au cours du temps ou encore la fraction d’éjection. Ce projet a donc pour but de créer un algorithme permettant d'aider le cardiologue dans ce diagnositc et de segmenter les différentes parties du coeur sur un IRM. 

## Pour commencer

Ce projet a été réalisé avec Python et une interface utilisateur a été implémentée mais n'est pour l'instant disponible qu'en local car nous n'avons pas pris de serveur pour le stocker. Les IRM utilsés pour la base d'entraînement et validation ont été téléchargés à partir du site 

### Pré-requis

Pour faire tourner ce programme, il est nécéssaire d'avoir Python dans une version égale ou ultérieure à 3.7 ainsi que les modules suivants installés :
- pandas
- tensorflow  
- opencv2
- medpy
- nibabel

## Execution

Pour éxecuter le programme, il suffit d'éxecuter le fichier python main.py pour avoir un exemple avec un IRM provenant de notre base de donnée. Il est aussi possible de l'éxecuter avec un autre IRM en remplaçant le fichier contenu dans prediction dans main.py par le nouvel IRM. L'éxecution renverra alors la segmentation du nouvel IRM selon notre algorithme.

Afin de faciliter l'utilisation du programme, nous avons aussi développé une interface utilisateur en local avec flash et plotly. Elle n'est cependant pas disponible en ligne par manque de serveur d'hébergement mais le fichier peut être transféré sur demande.

## Auteurs

Les personnes ayant participé au projet sont listé c-dessous: 
- Pierre Paynat alias @PaynatPierre
- Alexandre Abela alias @alexandreabela
- Basile Hogenmuller alias @bashog
- Joey Skaf alias @jskaf34
- Emma Mendizabal alias @Emma-IA
- Vincent Lébé
- Hugo Fourel
- Jonathan Desnoyer
