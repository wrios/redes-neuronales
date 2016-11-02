import numpy as np
import MatPlotLib as mpl

#dataset
#archivo de entrada
entrada = open(entrada, 'r+')
#leer la entrada y separar X de Z
X = np.array(entradaLeida.append(-1))
#umbral hardcode
umbral = 1

#matriz con los pesos
pesos = []
#filas neuronas entrada, columnas neuronas salida
filas = int(raw_input("Cantidad de Filas: "))
columnas = int(raw_input("Cantidad de Columnas: "))
for i in range(filas):
	tmp = []
	for j in range(columnas):
		elemento =  random.random()
		tmp.append(elemento)
		tmp.append(umbral)
	pesos.append(tmp)
print pesos
W = np.array(pesos)

#umbral
# umbral = []
# for i in range(filas):
# 	elemento =  random.random()
# 	umbral.append(elemento)
# umbralnp = np.array(umbral)

#np.dot producto de matrices


#learning rate
lr = 0.01

#temporadas de entrenamiento
#parametro de entrada T

#resultado
def calcular(X)
	#funcion sumatoria
	sumatorias = np.dot(X, W)

	Y = []
	for i in range(columnas):
		salidaTmp = math.copysign(1, sumatorias[i])
		Y.append(salidaTmp)
	return Y

#entrenamiento
def entrenamiento(X, Z)
	e = 1
	eps = 0.1
	t = 0
	TEnd = 1000
	while e > eps && t < TEnd:
		e = 0
		for x in range(dataset):
			Y = calcular(X) #esVector
			E = Z - Y #vectorial
			dW = lr*(X*E)
			W += dW
			e += np.dot(E, E)
		t+=1