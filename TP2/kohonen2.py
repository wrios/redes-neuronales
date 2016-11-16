from __future__ import division
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
import random
import platform
import itertools
from itertools import groupby as g
import csv
import sys
import getopt
import numpy as np
import ast
import scipy.io
from numpy import linalg as LA

TotalEpoca = 2000
m1 = 12
m2 = 12
W = []
X = np.ones((856,1))
Y = np.zeros((856,1))
M = []
WP = []
Map = np.zeros((m1, m2))
dmax = m1 #lo suficientemente grande para poder abarcar todas las neuronas y luego solo quede la ganadora
dmin = 0.55 # hace que vaya mas lento o rapido el aprendizaje

in_inicio_validacion = 0
in_fin_validacion = 0
in_cargarEntrenamiento = False
in_entrenamiento = True
in_guardarEntrenamiento = True
in_dataset = 'tp2_training_dataset.csv'

def learningRate(n):
	lr = 0.05/np.power(n+16,1.8)
	return lr


def armarMatrizActivaciones():
	global M
	for i in xrange(0,m1):
		fila = []
		for j in xrange(0,m2):
			fila.append([])
		M.append(fila)

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  

def armarMapaActivaciones():
	global M
	global Map
	for i in xrange(0,m1):
		for j in xrange(0,m2):
			Map[i][j] = most_common(M[i][j])

def gaussiana(n):
	r =  (dmax*((dmin/dmax)**(n/TotalEpoca)))
	if r < 1 and r > 0.3:
		r = 1
	return r 	

def armarMatriz():
	global W
	W = []
	for i in xrange(0,m1):
		fila = []
		for j in xrange(0,m2):
			fila.append(np.random.uniform( -0.1, 0.1, (856,1)))
		W.append(fila)

def activar(X):
	global Y
	norm = np.linalg.norm(W[0][0] - X)
	Y[:] = (W[0][0] - X)
	i = 0
	j = 0
	for i_t in xrange(0,m1):
		for j_t in xrange(0,m2):
			Y_t = (X - W[i_t][j_t] )
			norm_t = np.linalg.norm(Y_t)
			if norm_t < norm:
				norm = norm_t
				Y[:] = Y_t
				i = i_t
				j = j_t
	return i, j, Y_t

def rangoVecindad(n, i, j, i_t ,j_t):
	return np.exp((-((np.power(i-i_t,2)+(np.power(j-j_t,2)))/(2*(np.power(gaussiana(n),2))))))

def vecindad(n, i, j, Y_t):
	r = [] 
	for i_t in range(int(i-(gaussiana(n)/2)),int(i+(gaussiana(n)/2)+1)):
		for j_t in range(int(j-(gaussiana(n)/2)),int(1+i+(gaussiana(n)/2))):
			x = []
			t1 = i_t
			t2 = j_t
			if t1 < 0:
				t1 += m1
			if t1 > m1-1:
				t1 = t1%(m1-1)	
			if t2 < 0:
				t2 += m1
			if t2 > m1-1:
				t2 = t2%(m1-1)	 	
		#	print rangoVecindad(n, i, j, t1, t2)
			if not(i == t1 and j == t2):
				x = [t1, t2, (X - W[t1][t2])]#*rangoVecindad(n, i, j, t1, t2)]
			else:
				x = [t1, t2, Y_t] 
			r.append(x)		
	return r

def correccion(vecindades, n):
	global W
	e = 0 
	for vs in vecindades:
		for v in vs:
			i, j, delta = v	
			W[i][j] += (learningRate(n) * delta)
			e += np.linalg.norm(learningRate(n) * delta)
	print e, n, gaussiana(n)		

def epoca(n):
	global X
	global W
	global Y
	vecindades = []
	for vector in WP:
		i, j, Y_t = activar(vector)
		v = vecindad(n, i, j, Y_t)
		vecindades.append(v)
	correccion(vecindades, n)

def entrenamiento():
	global WP
	n = 1
	WP = []
	reader = csv.reader(open(in_dataset,'rb'))
	for vector in reader:
		X[:] = np.asarray(vector[1:]).reshape((856,1))
		WP.append(X)
	while n < TotalEpoca:
		progress(n, TotalEpoca)
		epoca(n)
		n += 1
#		print 'epoca',n

def cicloCompleto():
	global X
	global W
	global Y
	l = 1
	reader = csv.reader(open(in_dataset,'rb'))
	for vector in reader:
		#if not(int(in_inicio_validacion) <= e_actual <= int (in_fin_validacion)):
		X[:] = np.asarray(vector[1:]).reshape((856,1))
		i, j, Y_t = activar(X)
		M[i][j].append(vector[0])
		progress(l, 900, 'i: '+str(i)+'j: '+str(j)+'vector[0]: '+vector[0])
		l += 1

def graficar():
	print Map
	plt.matshow(Map)
	plt.show()

def most_common(L):
	if len(L) > 0:
		groups = itertools.groupby(sorted(L))
		def _auxfun((item, iterable)):
			return len(list(iterable)), -L.index(item)
		return max(groups, key=_auxfun)[0]
	else:
		return -1

def guardarEntrenamiento():
	outfile = open('entrenamiento_Kohonen'+str(TotalEpoca)+'_'+str(in_inicio_validacion)+'_'+str(in_fin_validacion), 'w')
	np.save(outfile, W)
	outfile.close()

def cargarEntrenamiento():
	global W
	infile = open('entrenamiento_Kohonen'+str(TotalEpoca)+'_'+str(in_inicio_validacion)+'_'+str(in_fin_validacion), 'r')
	W[:] = np.load(infile)
	infile.close()


def main():
	global in_dataset
	global in_cargarEntrenamiento
	global in_entrenamiento
	global in_guardarEntrenamiento
	global in_inicio_validacion
	global in_fin_validacion
	if len(sys.argv) > 1:
		if sys.argv[1] == '-h' or sys.argv[1] == '--help':
			print 'usage: python kohonen.py dataset cargarEntrenamiento? entrenar? guardarEntrenamiento? inicio_validacion fin_validacion'
			print 'los parametros bool se definen con 1 o 0, los de validacion con el inicio y final del segmento en int'
			print 'si no se especifica ningun parametro se usan los que estan por defecto en el codigo'
			print 'el entrenamiento se guarda y carga dependiendo de los parametros TotalEpoca en el codigo e inicio / validacion de parametros'
			return 0
		else:
			in_dataset = sys.argv[1]
			in_cargarEntrenamiento = sys.argv[2] == '1'
			in_entrenamiento = sys.argv[3] == '1'
			in_guardarEntrenamiento = sys.argv[4] == '1'
			in_inicio_validacion = sys.argv[5]
			in_fin_validacion = sys.argv[6]
	if in_cargarEntrenamiento:
		print 'Comienza Cargar Entrenamiento'
		cargarEntrenamiento()
		print 'Termina Cargar Entrenamiento'
	else:
		armarMatriz()
	if in_entrenamiento:
		print 'Comienza Entrenamiento'
		entrenamiento()
		print 'Termina Entrenamiento'
	if in_guardarEntrenamiento:
		print 'Comienza Guardar Entrenamiento'
		guardarEntrenamiento()
		print 'Termina Guardar Entrenamiento'
	print 'Comienza Ciclo Completo'
	armarMatrizActivaciones()
	cicloCompleto()
	print 'Termina Ciclo Completo'
	print 'Comienza Armar Mapa'
	armarMapaActivaciones()
	print 'Termina Armar Mapa'
	print 'Comienza Graficar'
	graficar()
	print 'Termina Graficar'

#El numero de pasos de entrenamiento se debe fijar antes apriori, para 
#calcular la tasa de convergencia de la funcion de vecindad y del learning right

if __name__ == '__main__':
	main()