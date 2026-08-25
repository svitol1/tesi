"""
Here we can watch the full matrix of cosine similarity between the 12 ECG
leads, which is used to define the signed graph structure in the ECGGraphDataset.
The matrix values range from -1 to 1, indicating the degree of similarity
between each pair of leads based on their spatial orientation.
"""

import numpy as np
import plotly.graph_objects as go
from pyvis.network import Network


# Ordine fisso dei nodi.
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_IDX = {name: i for i, name in enumerate(LEAD_ORDER)}

# Coordinate vettoriali unitarie (x, y, z) approssimate per le 12 derivazioni
# x = Asse Trasversale (Destra -> Sinistra)
# y = Asse Longitudinale/Verticale (Testa -> Piedi)
# z = Asse Anteroposteriore (Schiena -> Petto)
LEAD_VECTORS_3D = {
    # Derivazioni degli arti (Piano Frontale z=0, basate sul sistema esassiale)
    "I":   np.array([ 1.000,  0.000,  0.000]),  # 0°
    "II":  np.array([ 0.500,  0.866,  0.000]),  # +60°
    "III": np.array([-0.500,  0.866,  0.000]),  # +120°
    "aVR": np.array([-0.866, -0.500,  0.000]),  # -150°
    "aVL": np.array([ 0.866, -0.500,  0.000]),  # -30°
    "aVF": np.array([ 0.000,  1.000,  0.000]),  # +90°

    # Derivazioni Precordiali (Piano Orizzontale/Spaziale 3D)
    "V1":  np.array([-0.342,  0.174,  0.923]),  # Parasternale destra
    "V2":  np.array([-0.174,  0.174,  0.969]),  # Parasternale sinistra
    "V3":  np.array([ 0.174,  0.087,  0.981]),  # Tra V2 e V4
    "V4":  np.array([ 0.500,  0.000,  0.866]),  # Emiclaveare sinistra
    "V5":  np.array([ 0.866, -0.087,  0.492]),  # Ascellare anteriore
    "V6":  np.array([ 0.985, -0.174,  0.000])   # Ascellare media
}


def _compute_cosine_similarity_matrix() -> np.ndarray:
    """
    Calcola la matrice 12x12 di similarità del coseno tra i vettori di vista
    delle derivazioni. Valore in [-1, 1]:
        +1  -> stessa direzione (derivazioni "concordi")
         0  -> ortogonali (nessuna relazione lineare attesa)
        -1  -> direzioni opposte (derivazioni "discordi")
    """
    vectors = np.array([LEAD_VECTORS_3D[lead] for lead in LEAD_ORDER])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_normalized = vectors / norms
    return np.dot(vectors_normalized, vectors_normalized.T)

def create_heatmap(matrix):

    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=LEAD_ORDER,
            y=LEAD_ORDER,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[
                [0, "#2166ac"],
                [0.5, "#f7f7f7"],
                [1, "#b2182b"],
            ],
            text=[[f"{value:.2f}" for value in row] for row in matrix],
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}<br>cosine similarity: %{z:.4f}<extra></extra>",
        )
    )

    figure.update_layout(
        title="12-Lead ECG - Cosine Similarity Matrix",
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
    )

    figure.show()

def main():
    matrix = _compute_cosine_similarity_matrix()
    create_heatmap(matrix)

# define main function to be called when the script is executed
if __name__ == "__main__":
    main()