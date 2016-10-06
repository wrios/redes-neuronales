import csv
import sys
import getopt
import numpy as np

m = 7
l = 5
lr = 0.15
# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def activation():
#	global W
#	global X
#	global Y

	Y[0] = np.insert(X, 10,-1)

  	Y[1] = nonlin(np.dot(Y[0], W[0]))
  	Y[1] = np.insert(Y[1], m, -1)
  	
  	Y[2] = nonlin(np.dot(Y[1], W[1]))
  	
  	Y[2] = np.insert(Y[2], l, -1)

  	Y[3] = nonlin(np.dot(Y[2], W[2]))
  	return Y

def correction(esperado):
	global Y
	global deltaW
#Diferencia entre el resultado obtenido y el esperado	
	E = Y[2] - esperado
#Se usa el modulo y se lo eleva al cuadrado, porque el error cuadratico medio es buen indicador	
	e = np.power(np.absolute(E),2)

	for j in range(1, 3):
		
		D = E * nonlin( np.dot(Yprima[3-j-1], W[3-j]), True)

		deltaW[3-j] = deltaW[3-j] + lr * np.dot(Yprima[3-j-1].T, D)

	return e

#generados en matrices con numeros aleatorios entre 0 y 1.

#DimX() = 11
X = np.zeros(10)

#Dim(deltaW[0]) = 
W = [np.random.rand(11, m), np.random.rand(m+1, l), np.random.rand(l+1, 1)]
deltaW = [np.zeros((11, m)), np.zeros((m+1, l)), np.zeros((l+1, 1))]

Y = [np.zeros(m+1), np.zeros(l+1), np.zeros(1), np.zeros(1)]
Yprima = [np.zeros(m), np.zeros(l), np.zeros(1)]

def adaptation():
	global W
	global deltaW
	for j in range(1, 3):
		W[j] = W[j] + deltaW[j]
		deltaW[j] = 0
	return deltaW

def main():
  	

	Y[0] = np.insert(X, 10,-1)

  	Y[1] = nonlin(np.dot(Y[0], W[0]))
  	Y[1] = np.insert(Y[1], m, -1)
  	
  	Y[2] = nonlin(np.dot(Y[1], W[1]))
  	
  	Y[2] = np.insert(Y[2], l, -1)

  	Y[3] = nonlin(np.dot(Y[2], W[2]))

  	Y[0] = np.delete(Y[0], 10, -1)
  	Y[1] = np.delete(Y[1], m, -1)
  	Y[2] = np.delete(Y[2], l , -1)
#	for j in range(0, 3):
#		W[j] = W[j] + deltaW[j]
#	print W[j]

	W[0] = W[0] + deltaW[0]
	W[1] = W[1] + deltaW[1]
	W[2] = W[2] + deltaW[2]

	W[2] = np.delete(W[2], (l), axis = 0)
	W[1] = np.delete(W[1], (m), axis = 0)
	deltaW[2] = np.delete(deltaW[2], (l), axis = 0)
	deltaW[1] = np.delete(deltaW[1], (m), axis = 0)
	
	E = Y[2] - 1
	D = E * nonlin(np.dot(Y[2], W[2]), True)
	deltaW[2] = deltaW[2] + lr * np.dot(Y[2].T, D) 
# Se usa el modulo y se lo eleva al cuadrado, porque el error cuadratico medio es buen indicador	
	e = np.power(np.absolute(E),2)
	

	print "EMPEZAMOS CON EL FOR"
	print deltaW[2]
	print D.shape
	print e

	print "Y[0] Y W[0]"
	print Y[0].shape
	print W[0].shape

	print "Y[1] Y W[1]"
	print Y[1].shape
	print W[1].shape

	print "Y[2] Y W[2]"
	print Y[2].shape
	print W[2].shape

	print "Y[3]"
	print Y[3]

if __name__ == "__main__":
    main()