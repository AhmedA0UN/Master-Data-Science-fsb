import DataRegression as dr
import CalculStat as cs
import numpy as np
import matplotlib.pyplot as plt
print("Appliquer la méthode des moindres carrés")
# Générer des données corrélées
np.random.seed(42)
x = np.linspace(0, 10, 50)
print("x=",x)
y = 2 * x + 1 + np.random.normal(0, 1, size=50)
print("y=",y)

# Calculer la moyenne des données
mean_x = cs.moyenne(x)
mean_y = cs.moyenne(y)
print("Moyenne de x:", mean_x)
print("Moyenne de y:", mean_y)

# Calculer la variance  des données
variance_x =cs.variance(x)
variance_y =cs.variance(y)
print("Variance  de x:", variance_x)
print("Variance de y:", variance_y)


# Calculer la covariance  
covariance_x_y=cs.covarience(x,y)
print("Covariance  entre x et y:", covariance_x_y)

# Calculer a
a=covariance_x_y/variance_x
print("la pente a =" ,a)


a2=covariance_x_y/variance_y
print("la pente a2 =" ,a2)

# Calculer b
b=(mean_y-(mean_x*a))
print("l'ordonne a l'orgine b =" ,b)

b2=(mean_x-(mean_y*a2))
print("l'ordonne a l'orgine b2 =" ,b2)
#a, b = np.polyfit(x, y, 1)



# Calculer y2 predite
y2=a*x+b
print("y2=", y2)

# Calculer la coeff de correlation
r=cs.correlation(x,y)
print("r=",r)

#calcul des écarts e
for i in range (50):
    e=y-y2
print("e=",e)


# Calculer la moyenne des ecarts
mean_ecart_e = cs.moyenne(e)
print("Moyenne d'écart e:", mean_ecart_e)

# Calculer la variance des ecarts
variance_ecart_e =cs.variance(e)
print("Variance  d'écart e:", variance_ecart_e)

# Calculer la moyenne de y2
mean_y2 = cs.moyenne(y2)
print("Moyenne de y2:", mean_y2)


# Calculer la variance de y2
variance_y2 =cs.variance(y2)
print("Variance  de y2:", variance_y2)
# **************on remarque que Variance  de y=Variance  d'écart e+Variance  de y2************************************


# Calculer la coeff de qualité d'ajustement
QA= variance_y2/variance_y
print("qualité d'ajustement=", QA)

y3=(((1/a2)*x)-(b2/a2))
if (cs.forteCorrelation(x,y)==True):
    print('forte correlation tu peux tracer droite')
    # Tracer la droite de régression
    plt.plot(x, y2, color='red', label='Droite de régression de Y en X')
    plt.plot(x, y3, color='green', label='Droite de régression de X en Y')
else :
    print('faible  correlation tu ne peux pas tracer droite')




# Tracer le nuage de points
plt.scatter(x, y, label='Données')


# Afficher le graphique
plt.xlabel('Axe X')
plt.ylabel('Axe Y')
plt.legend()
plt.show()



