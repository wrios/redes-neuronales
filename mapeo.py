from __future__ import division
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
import random
import platform
import itertools
import csv
import sys
import getopt
import numpy as np
import ast
import scipy.io
from numpy import linalg as LA
#mapa autoorganizado de kohonen
#matriz de pesos, donde se usa modulo para formar el mapa
#La matriz de pesos es de n (tam entrada) x m1xm2 (alto y ancho del mapa. cantidad de neuronas)


l = 856
dW = np.zeros((856,l))
W = np.random.uniform( -0.1, 0.1, (856,l))
Xm = np.zeros((856,l))
X = np.ones((856,1))
m1 = 4
m2 = 4
dmax = m1 #lo suficientemente grande para poder abarcar todas las neuronas y luego solo quede la ganadora
dmin = 0.5 # hace que vaya mas lento o rapido el aprendizaje 

#def p(j):
#	return <[j/(cardinal(?)), j mod m2] >
	

#VA A CORREGIR LA MATRIZ SOLO MODIFICANDO LA NEURONA GANADORA Y SUS VECINAS 


in_inicio_validacion = 0
in_fin_validacion = 0
in_cargarEntrenamiento = False
in_entrenamiento = True
in_guardarEntrenamiento = False
Wm = np.zeros((856,m1*m2))
dW = np.zeros((856,m1*m2))

#CREA LA MATRIZ QUE CONTIENE LAS NEURNAS y les pone dimension (856,1)
def fMascara(W):
	global mascara 
	mascara = []
 	for row in range(0,m1):
 		mascara.append([])
 		for column in range(0,m2):
 			mascara[row].append(W.T[(row*m1)+column].reshape((856,1)))
 	return mascara		

#compara si la matriz mascara tiene todos los elementos de la matriz W 
def comparacion(W): 	
	Z = fMascara(W)
	i = 0
	for row in range(0,m1):
		for column in range(0,m1):
			for j in range(0,m2):
				#if Z[row][column][j] == W.T[row][j]:
				if not(Z[row][column][j]==W.T[(m1*row)+column][j]):
					i = i + 1
				else:
					i = i	
	return i == 0		


#funcion gaussiana para la funcion de actucalizacion te da el radio de la epoca
def gaussiana(n):
	return (dmax*((dmin/dmax)**(n/TotalEpoca)))


#ganadora revice el vector dato y la matriz de neuronas
#devuelve una lista con el vector ganador y sus coordenadas correspondientes a la matriz 
def ganadora(X,M):	
#el reshape es porque al tomar las columnas de la matriz la dimension es (856,vacio)	
#la funcion devuelve el vector de la matrix mascara con sus coordenas para despues poder calcular el vecindario 
	Y = [M[0][0],0,0]
	r = np.linalg.norm(X-Y[0])
	for j in range(0,m1):
		for i in range(0,m2):
			if r > np.linalg.norm(X-M[j][i]):
				r = np.linalg.norm(X-M[j][i])
				Y = [M[j][i],j,i]
	return Y 	
#	Y = [M[0][0].reshape((856,1)),0,0]
#	r = np.linalg.norm(X-Y[0].reshape((856,1)))
#	for j in range(0,m1):
#		for i in range(0,m2):
#			if r > np.linalg.norm(X-M[j][i].reshape((856,1))):
#				r = np.linalg.norm(X-M[j][i].reshape((856,1)))
#				Y = [M[j][i].reshape((856,1)),j,i]
#	return Y

#vecindad toma como primer parametro una lista con el vector ganador y sus coordenadas,[vector,i,j]
#2do parametro es la epoca actual		
#3ro matriz con las neuronas  	
#devuelve una lista con los vecinos de la ganadora
def vecindad(Y,n,M):
	r = []
	#k = -1
	for i in range(0,m1):
		for j in range(0,m2):
			if abs(i- Y[1]) <= gaussiana(n) and abs(j-Y[2]) <= gaussiana(n):		
				#print 'AGREGANDO ELEMENTO**************'
				r.append([M[i][j],i,j])
				#k += 1
				#print 'k-esimo elemento agregado'
				#print k
				#print 'dimension de r[k]'
				#print r[k][1]
				#print r[k][2]
				#print 'dimension de la lista'
				#print np.shape(r)
	return r						



#1er parametro una vecindad
#2do parametro M (La matriz de neuronas)
#3ro parametro la ganadora
#4to parametro n
#tomas las todas las vecinas y la ganadora y hace la correcion
#luego reemplaza los valores en la matriz de neuronas
def correccion(V,M,Y,n):
#	i recorre los elementos que estan en la vecindad3
	for i in range(0,np.shape(V)[0]):
#		reemplaza la correccion en las vecinas
		V[i][0] = V[i][0] + (0.01/n) * (X-V[i][0])
		M[V[i][1]][V[i][2]] = V[i][0]
	return M 	


def epoca(n,M):
#	e_actual = 0
	reader = csv.reader(open('tp2_training_dataset.csv','rb'))
	for vector in reader:
		#if not(int(in_inicio_validacion) <= e_actual <= int (in_fin_validacion)):
		X[:] = np.asarray(vector[1:]).reshape((856,1))
		Y = ganadora(X,M)
		V = vecindad(ganadora(X,M),n,M)
		correccion(V,M,Y,n)
#		e_actual += 1	D 
	return M		


def entrenamiento(M):
	n = 1
	while n < TotalEpoca:
		epoca(n,M)
		n += 1
		print "epoca: ", n
	return M	



def main():
	global TotalEpoca
#numero total de epocas
	TotalEpoca = 500
#numero de espocas total
	n = 2500
#Crea la matriz con las neuronas 	
	M = fMascara(W)
#devuelve la ganadora 	
	Y = ganadora(X,M)
#devuelve el vecindario de la ganadora (segun la cantidad de epocas)	
	V = vecindad(Y,n,W)
#devuelve la correcion para una neurona	
	C = correccion(V,M,Y,n)
#devuelve una epoca(una pasada por el dataset)
	E = epoca(n,M)	
#entrenamiento
	print entrenamiento(M)	
			
	#print de la neurona [0][0]
	#print type(correccion(V,M,Y,n)[0][0])

#El numero de pasos de entrenamiento se debe fijar antes apriori, para 
#calcular la tasa de convergencia de la funcion de vecindad y del learning right






if __name__ == '__main__':
	main()