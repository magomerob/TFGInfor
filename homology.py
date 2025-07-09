import numpy as np
import os
import networkx as nx
import pickle
from itertools import chain, combinations
import matplotlib.pyplot as plt

def process_matrix(filename, inputdir = "matrices", nThresholds = 2000, thresholdMin=0.0, thresholdMax=0.5, outputdir = "matricesProcesadas"):
    """
    Procesa una matriz de atención umbralizándola en múltiples niveles para construir 
    grafos bipartitos y una matriz de disimilitud tipo Rips. Guarda el resultado en un 
    archivo .pkl para su análisis topológico posterior.

    Args:
        filename (str): Ruta al archivo .npy con la matriz de atención.
        inputdir (str): Directorio de entrada donde se encuentran las matrices (no usado directamente).
        nThresholds (int): Número de umbrales a evaluar entre thresholdMin y thresholdMax.
        thresholdMin (float): Valor mínimo del umbral.
        thresholdMax (float): Valor máximo del umbral.
        outputdir (str): Carpeta donde se guardará el archivo de salida .pkl.

    Returns:
        str: Ruta al archivo .pkl generado con la matriz de Rips y los grafos bipartitos.
    """
    
    # Generar umbrales uniformemente distribuidos
    thresholds = np.linspace(thresholdMin, thresholdMax, nThresholds)[1:]

    # Cargar matriz de atención
    attn_mx = np.load(filename)
    attn_mx = attn_mx
    
    # Crear carpeta de salida si no existe
    os.makedirs(outputdir, exist_ok=True)

    # Diccionario para almacenar resultados por umbral
    results = {}

    W = np.maximum(attn_mx, attn_mx.T)
    D = 1.0 - W
    np.fill_diagonal(D, 0.0)
    
    
    bpgraph = {}
    for t in thresholds:
        # Crear copia umbralizada de la matriz
        tMatrix = attn_mx.copy()
        np.fill_diagonal(tMatrix, 0)
        binarized = (tMatrix >= t).astype(int)

        # Crear grafo no dirigido a partir de la matriz binarizada
        
        BpG = nx.Graph()
        n = binarized.shape[0]
        
        BpG.add_nodes_from(range(n*2))
        for i in range(n):
            for j in range(n):
                if binarized[i, j] == 1 or binarized[j, i] == 1:
                    BpG.add_edge(i, j)
        
        # Guardar resultados por umbral
        bpgraph[t] =  BpG
        
    results = {
        'ripsMatrix':D,
        'bpgraph': bpgraph
    }
    # Guardar el diccionario completo
    filename2 = filename.split("/")[-1]
    output_path = outputdir+"/"+filename2.replace(".npy", "_complex.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    return output_path
