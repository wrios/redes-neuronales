from random import choice
from numpy import array, dot, random

unit_step = lambda x: 0 if x < 0 else 1

#Entradas de letras, completar con el abecedario en forma de matriz de puntos.
training_data = [
	(array([0,0,1]), 0),
	(array([0,1,1]), 1),
	(array([1,0,1]), 1),
	(array([1,1,1]), 1),
]

#Matriz 5x5 para representar la grilla de la letra
w = random.rand(5, 5)
errors = []
eta = 0.2
n = 100

for i in xrange(n):
	x, expected = choice(training_data)
	result = dot(w, x)
	error = expected - unit_step(result)
	errors.append(error)
	w += eta * error * x

for x, _ in training_data:
	result = dot(x, w)
	print("{}: {} -> {}".format(x[:2], result, unit_step(result)))