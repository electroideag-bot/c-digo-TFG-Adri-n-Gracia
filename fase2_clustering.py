"""
fase2_clustering.py
-------------------
Fase 2. Tipología de las tiendas mediante agrupamiento sobre dos variables,
la superficie y la captación. Se ajusta k-means para k = 3, se valida con el
coeficiente de silueta y se comparan las particiones de k-means y de
agrupamiento jerárquico (enlace de Ward) mediante el índice de Rand ajustado.

Entrada : data/tiendas.pkl
Salida  : salidas/arquetipos.csv
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score


def main():
    tiendas = pd.read_pickle('data/tiendas.pkl')
    X = tiendas[['sup', 'capt']].values
    Xs = StandardScaler().fit_transform(X)

    # k-means con k = 3
    km = KMeans(n_clusters=3, n_init=20, random_state=0).fit(Xs)
    sil = silhouette_score(Xs, km.labels_)

    # Agrupamiento jerárquico (Ward) y concordancia con k-means
    ward = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(Xs)
    ari = adjusted_rand_score(km.labels_, ward.labels_)

    tiendas = tiendas.copy()
    tiendas['cluster'] = km.labels_

    # Perfil medio de cada arquetipo (ordenado por facturación media)
    perfil = (tiendas.groupby('cluster')
              .agg(n=('fact', 'size'),
                   superficie_media=('sup', 'mean'),
                   captacion_media=('capt', 'mean'),
                   facturacion_media=('fact', 'mean'))
              .sort_values('facturacion_media'))
    perfil.to_csv('salidas/arquetipos.csv')

    print(f'k-means (k=3): silueta = {sil:.3f}')
    print(f'Concordancia k-means vs. Ward (ARI) = {ari:.3f}')
    print('\nPerfil medio de los arquetipos:')
    print(perfil.round(1).to_string())
    print('\n(Interpretación: Periférico = mayoría; Gran formato = superficie '
          'alta; Núcleo = captación muy alta del área metropolitana.)')


if __name__ == '__main__':
    main()
