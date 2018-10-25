# -*- coding: utf-8 -*-
'''
    Nombre del archivo: arbol_clasificacion.py
    Autores: 
    Fecha de creación: 
    Última fecha de modificación:
    Versión de Python: 3.6
    Descripción: Código que implementa el algoritmo de árboles de clasificasión de Machine learning
'''

#Librerías necesarias para la ejecución del código
#Pandas nos permite obtener los datos del archivo de excel
import pandas as pd
#Numpy nos permite convertir los datos para que puedan ser manipulados
import numpy as np

#Importamos los datos de un archivo CSV 
data=pd.read_csv('2012.csv')
#Utilizando pandas nos retorna un DataFrame así que lo convertimos a una lista de listas que va a ser nuestro trainning data
training_data=np.array(data).tolist()

etiquetas = ["sexo", "prom_bacho","prom_s","bachillerato","ingreso_mensual","trabaja","tiempo_iv","n_habitantes","padre_esco","madre_esco","n_hermanos","dependencia_economica","razon_ingenieria","A_TIEMPO"]

def cantidadClases(filas):
    '''Contamos qué tantas respuestas distintas hay en cada columna. Va a ser útil para calcular la impureza de Gini'''
    counts = {}  
    for fila in filas:
        etiqueta = fila[-1]
        if etiqueta not in counts:
            counts[etiqueta] = 0
        counts[etiqueta] += 1
    return counts

def esNumerico(valor):
    '''Revisar si un valor dado es o no numérico'''
    return isinstance(valor, int) or isinstance(valor, float)

class Pregunta:
    '''Clase que define la pregunta o el nodo a partir del cuál se particiona la información'''
    def __init__(self, columna, valor):
        self.columna = columna
        self.valor = valor

    def match(self, example):
        '''Compara la información dependiendo de si es un valor numérico o de texto '''
        val = example[self.columna]
        if esNumerico(val):
            return val >= self.valor
        else:
            return val == self.valor

    def __repr__(self):
        condicion = "=="
        if esNumerico(self.valor):
            condicion = ">="
        return "Is %s %s %s?" % (
            etiquetas[self.columna], condicion, str(self.valor))

def particion(filas, pregunta):
    '''Función que parte el conjunto en dos dependiendo a si se cumple o no la condición'''
    conjuntoVerdadero, conjuntoFalso = [], []
    for fila in filas:
        if pregunta.match(fila):
            conjuntoVerdadero.append(fila)
        else:
            conjuntoFalso.append(fila)
    return conjuntoVerdadero, conjuntoFalso

def gini(filas):
    '''Función que calcula la impureza de Gini'''
    counts = cantidadClases(filas)
    impureza = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(filas))
        impureza -= prob_of_lbl**2
    return impureza

def gananciaInformacion(izquierda, derecha, incertidumbreActual):
    '''Función que calcula la ganancia de información basándose en la impureza de gini'''
    p = float(len(izquierda)) / (len(izquierda) + len(derecha))
    return incertidumbreActual - p * gini(izquierda) - (1 - p) * gini(derecha)

def encontrarMejorParticion(filas):
    '''Función que encuentra el mejor particioamiento al hallar la mejor pregunta'''
    mejorGanancia = 0  
    mejorPregunta = None  
    incertidumbreActual = gini(filas)
    n_features = len(filas[0]) - 1  

    for col in range(n_features): 

        valores = set([fila[col] for fila in filas]) 

        for val in valores:

            pregunta = Pregunta(col, val)

            #Intentar particionar el conjunto
            verdadero_filas, falso_filas = particion(filas, pregunta)

            #Si alguno de los conjuntos está vacío ya no va a calcular la impote ni la ganancia de información
            if len(verdadero_filas) == 0 or len(falso_filas) == 0:
                continue

            gain = gananciaInformacion(verdadero_filas, falso_filas, incertidumbreActual)

            if gain >= mejorGanancia:
                mejorGanancia, mejorPregunta = gain, pregunta

    return mejorGanancia, mejorPregunta

class Hoja:
    ''' Clase que almacena los conjuntos finales, hoja o las predicciones '''
    def __init__(self, filas):
        self.predictions = cantidadClases(filas)

class DecisionNodo:
    ''' Clase que contiene las preguntas/nodo y sus conjuntos verdaderos y falsos '''
    def __init__(self,pregunta,conjuntoVerdadero,conjuntoFalso):
        self.pregunta = pregunta
        self.conjuntoVerdadero = conjuntoVerdadero
        self.conjuntoFalso = conjuntoFalso

def crearArbol(filas):
    ''' Función que va creando el árbol recursivamente de acuerdo a la información '''
    ganancia, pregunta = encontrarMejorParticion(filas)

    if ganancia == 0:
        return Hoja(filas)

    conjuntosVerdaderos, conjuntosFalsos = particion(filas, pregunta)

    conjuntoVerdadero = crearArbol(conjuntosVerdaderos)

    conjuntoFalso = crearArbol(conjuntosFalsos)

    return DecisionNodo(pregunta, conjuntoVerdadero, conjuntoFalso)

def imprimirArbol(nodo, spacing=""):
    '''Función que imprime un árbol a partir de un nodo'''
    if isinstance(nodo, Hoja):
        print (spacing + "Predict", nodo.predictions)
        return

    print (spacing + str(nodo.pregunta))

    print (spacing + '--> verdadero:')
    imprimirArbol(nodo.conjuntoVerdadero, spacing + "  ")

    print (spacing + '--> falso:')
    imprimirArbol(nodo.conjuntoFalso, spacing + "  ")

miArbol = crearArbol(training_data)
#imprimirArbol(miArbol)

def clasificar(fila, nodo):
    ''' Función que clasicica los nodos recursivamente y al final retorna las predicciones '''
    if isinstance(nodo, Hoja):
        return nodo.predictions

    if nodo.pregunta.match(fila):
        return clasificar(fila, nodo.conjuntoVerdadero)
    else:
        return clasificar(fila, nodo.conjuntoFalso)

def imprimirHoja(counts):
    ''' Función que imprime el nodo hoja al final'''
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = int(counts[lbl] / total * 100)
    return probs

