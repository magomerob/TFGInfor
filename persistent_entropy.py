import numpy as np, gudhi
from gudhi.representations import Entropy
import os
import pickle
from gudhi import SimplexTree, RipsComplex
import matplotlib.pyplot as plt

def load_complex(filepath):
    """Carga un complejo generado por homology.py"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

data = []

for filename in os.listdir('homotemp'):
    
    temp = int(filename.split("_")[1][1:])/100
    
    complex_data = load_complex(os.path.join('homotemp', filename))
    
    # -------- 1. point cloud -> simplex tree -> diagram ------------
    ripsM = complex_data['ripsMatrix']
    rips = RipsComplex(distance_matrix=ripsM, max_edge_length=1.0)
    st   = rips.create_simplex_tree(max_dimension=3)
    st.compute_persistence()
    diag = st.persistence_intervals_in_dimension(3)

    # -------- 2. persistent entropy -------------------------------
    entropy = Entropy()(diag)     # default: scalar, normalised lifetimes
    
    data.append((temp, float(entropy)))
    
# sort and unpack
data.sort(key=lambda t: t[0])
temp, entropies = zip(*data)
x = np.array(temp)
y = np.array(entropies)

# fit linear trend
m, b = np.polyfit(x, y, 1)
y_trend = m * x + b

# -- PLOT --
plt.figure(figsize=(9, 5))

# scatter with small, semi-transparent markers
plt.scatter(x, y, s=20, alpha=0.4, label="Entropía H_3")

# thick dashed trend line
plt.plot(x, y_trend, linewidth=2.5, linestyle="--", color="orange", label=f"Tendencia (m={m:.2e})")

# nicer grid and axes
plt.grid(which="major", linestyle=":", linewidth=0.8, alpha=0.7)
plt.xlabel("Temperatura", fontsize=12)
plt.ylabel("Entropía persistente (H₁)", fontsize=12)
plt.title("Evolución de la entropía persistente", fontsize=14, pad=12)

# tighten y-limits a bit around the data
ymin, ymax = y.min(), y.max()
yrange = ymax - ymin
plt.ylim(ymin - 0.05*yrange, ymax + 0.05*yrange)

plt.legend(frameon=False, fontsize=11)
plt.tight_layout()
plt.show()