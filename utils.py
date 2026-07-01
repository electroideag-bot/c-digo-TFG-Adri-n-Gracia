"""
utils.py
--------
Funciones auxiliares comunes a las tres fases del análisis: distancias
geográficas, aproximación de la distancia por carretera y el reparto por
gravedad que define la captación efectiva.

TFG - Recomendación de ubicación y previsión de facturación de una nueva
tienda Obramat en la Comunidad de Madrid.
"""

import numpy as np

# --- Constantes del modelo ---------------------------------------------------
R_TIERRA = 6371.0        # radio de la Tierra en kilómetros
FACTOR_CARRETERA = 1.3   # corrector de distancia recta a distancia por carretera
RADIO_SERVICIO = 15.0    # radio de captación de una tienda, en kilómetros
DIST_MINIMA = 0.5        # distancia mínima para evitar divisiones por cero (km)


def haversine(lat1, lon1, lat2, lon2):
    """Distancia del círculo máximo (km). Admite escalares o arrays de NumPy."""
    p = np.pi / 180.0
    a = (0.5 - np.cos((lat2 - lat1) * p) / 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2)
    return 2 * R_TIERRA * np.arcsin(np.sqrt(a))


def distancia_carretera(lat1, lon1, lat2, lon2):
    """Aproxima la distancia por carretera aplicando el factor corrector."""
    return haversine(lat1, lon1, lat2, lon2) * FACTOR_CARRETERA


def pesos_gravedad(distancias, radio=RADIO_SERVICIO, exponente=2):
    """
    Pesos del modelo de gravitación comercial de Huff.

    Para un código postal, cada tienda dentro del radio de servicio recibe un
    peso inversamente proporcional a la distancia elevada al `exponente`
    (por defecto 2). Las tiendas fuera del radio reciben peso cero.
    """
    dentro = distancias <= radio
    pesos = np.where(dentro,
                     1.0 / np.maximum(distancias, DIST_MINIMA) ** exponente,
                     0.0)
    return pesos


def captacion_efectiva(coord_tiendas, demanda, exponente=2):
    """
    Calcula la captación efectiva de cada tienda.

    La demanda de cada código postal que queda al alcance de varias tiendas se
    reparte entre ellas por gravedad, en lugar de asignarse entera a cada una.
    Devuelve un array con la captación efectiva (en millones de euros) por
    tienda, en el mismo orden que `coord_tiendas`.

    Parámetros
    ----------
    coord_tiendas : array (n_tiendas, 2) con columnas [lat, lon].
    demanda       : DataFrame con columnas ['lat', 'lon', 'fact'] por código
                    postal (facturación potencial en euros).
    """
    lat_t, lon_t = coord_tiendas[:, 0], coord_tiendas[:, 1]
    lat_d = demanda['lat'].values
    lon_d = demanda['lon'].values
    fact_d = demanda['fact'].values

    # Matriz de distancias por carretera (códigos postales x tiendas)
    D = np.zeros((len(fact_d), len(lat_t)))
    for j in range(len(lat_t)):
        D[:, j] = distancia_carretera(lat_d, lon_d, lat_t[j], lon_t[j])

    # Pesos de gravedad y normalización por código postal (por filas)
    W = pesos_gravedad(D, exponente=exponente)
    suma = W.sum(axis=1, keepdims=True)
    suma[suma == 0] = 1.0            # evita 0/0 en códigos postales sin cobertura
    reparto = W / suma

    # Captación efectiva de cada tienda (suma de la demanda que le corresponde)
    efectiva = (fact_d[:, None] * reparto).sum(axis=0) / 1e6
    return efectiva
