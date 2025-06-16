import pickle
import numpy as np
import networkx as nx
from scipy.linalg import eigvalsh  # para espectro del Laplaciano
from gudhi import SimplexTree, RipsComplex
import os

def load_complex(filepath):
    """Carga un complejo generado por homology.py"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def euler_characteristic(complex_data):
    """Calcula la característica de Euler para cada umbral"""
    chi_by_threshold = {}
    for t, data in complex_data['bpgraph'].items():
        G = data
        beta0 = nx.number_connected_components(G)

        n, m = G.number_of_nodes(), G.number_of_edges()
        beta1 = m - n + beta0

        chi = beta0 - beta1
        chi_by_threshold[t] = chi
        
    return chi_by_threshold

def persistence_diagram(complex_data, dimensions):
    """Calcula el diagrama de persistencia a partir de la filtración"""
    
    ripsM = complex_data['ripsMatrix']
    rips = RipsComplex(distance_matrix=ripsM, max_edge_length=1.0)
    st   = rips.create_simplex_tree(max_dimension=dimensions)
    diag = st.persistence()
    
    return diag

def laplacian_spectra(complex_data, normalized=False):
    """Calcula el espectro del Laplaciano para cada grafo umbralizado"""
    spectra = {}
    for t, data in complex_data['bpgraph'] .items():
        G = data       
        if len(G) == 0:
            spectra[t] = []
            continue
        if normalized:
            L = nx.normalized_laplacian_matrix(G).todense()
        else:
            L = nx.laplacian_matrix(G).todense()
        eigenvalues = eigvalsh(L)
        spectra[t] = eigenvalues
    return spectra

def analyze_complex(filepath, normalized_laplacian=False, outputdir="topology", dimensions=3):
    """
    Realiza el análisis topológico completo de un complejo generado por homology.py.

    Parámetros:
    - filepath: ruta al archivo .pkl generado por homology.py
    - max_homology_dim: dimensión máxima para los diagramas de persistencia
    - normalized_laplacian: si True, se usa el Laplaciano normalizado

    Retorna:
    - diccionario con los resultados: característica de Euler, espectros, y diagrama de persistencia
    """

    os.makedirs(outputdir, exist_ok=True)
    
    complex_data = load_complex(filepath)
    chi = euler_characteristic(complex_data)
    spectra = laplacian_spectra(complex_data, normalized=normalized_laplacian)
    diagram = persistence_diagram(complex_data, dimensions)
    
    results = {
        'euler_characteristic': chi,
        'laplacian_spectra': spectra,
        'persistence_diagram': diagram
    }
    
    filepath2 = filepath.split("/")[-1]
    output_path = outputdir+"/"+filepath2.replace("_complex.pkl", "_topology.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    return output_path
