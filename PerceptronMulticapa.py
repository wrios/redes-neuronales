#variables con el tamaño de las capas intermedias.
int m
int l
#Matrices con los pesos desde una capa a la siguiente. (los pesos que inciden en una neurona J desde una neurona I).
#generados en matrices con números aleatorios entre 0 y 1.

#Vector de entrada (1x10) que contiene los datos de una instancia
E = random.rand(1, 10)

#Matriz de 10xm que representa el peso de las entradas en las neuronas de la primera capa intermedia (oculta)
wEO1 = random.rand(10, m)

#Matriz de mxl que reprensenta el peso de las neuronas de la primera capa intermedia en la segunda.
wO1O2 = random.rand(m, l)

#Matriz lx1 que representa el peso de las neuronas de la segunda capa en la neurona de salida que es única por ser un problema de decisión binario.
wO2S = random.rand(l, 1)

def propagar(E){
	rEO1 = dot(E, wEO1)
	rO1O2 = dot(rEO1, wO1O2)
	rO2S = dot(rO1O2, wO2S)

	return rO2S
}