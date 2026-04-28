#figures
from matplotlib.pyplot import subplot, scatter, plot, axis
from scipy.stats import linregress

#pour le calcul des statistiques
import statistics  #module natif du python
import numpy as np #package
import scipy.stats #package


"""
créer les jeux de données de Anscombe sous forme de vecteur numpy
"""





 



xmax = 20
ymax = 14

"""
- Afficher les statistiques de chaque jeux de données :moyenne, variance, std, correlation entre x et y
- utiliser numpy.corrcoef
- interpréter et conclure
"""








"""graphiques
- La regression linéaire permet de modéliser la relation entre x et y par un modèle simple linéaire
La bibliothèque Scipy contient la fonction "linregress" qui assure cette regression
on vous donne une partie du code, 
- Afficher les figures de tous les jeu de données
- Afficher l'écart type de l'erreur de regression de chaque jeu de donnée
- Interpréter et conclure
"""



ax1 = subplot(2, 2, 1)
scatter(x1, y1)
slope, intercept, r_value, p_value, std_err = linregress(x1, y1)
plot([0, xmax], [intercept, slope * xmax + intercept])
