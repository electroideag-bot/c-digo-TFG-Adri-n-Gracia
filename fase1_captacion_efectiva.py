"""
fase1_captacion_efectiva.py
---------------------------
Fase 1 (núcleo de la medida de mercado). A partir de la capa de demanda por
código postal y de la localización de las tiendas, calcula para cada almacén:

  - la captación BRUTA: la demanda situada a menos de 15 km por carretera,
    asignada en exclusiva a la tienda;
  - la captación EFECTIVA: la misma demanda, pero repartida por gravedad
    (modelo de Huff, peso 1/d^2) entre las tiendas que la cubren, de modo que
    descuenta la canibalización entre almacenes de la propia red.

La captación efectiva es la variable de mercado que alimenta la regresión de
la Fase 3. La captación bruta se conserva para comparar.

Entrada : data/tiendas.pkl, data/demanda_es.pkl
Salida  : salidas/captacion.csv  (y verificación contra el dato precalculado)
"""

import numpy as np
import pandas as pd

from utils import (distancia_carretera, captacion_efectiva,
                   RADIO_SERVICIO)


def captacion_bruta(coord_tiendas, demanda):
    """Demanda total (M€) a menos de RADIO_SERVICIO km por carretera de cada tienda."""
    lat_d, lon_d = demanda['lat'].values, demanda['lon'].values
    fact_d = demanda['fact'].values
    bruta = np.zeros(len(coord_tiendas))
    for j, (la, lo) in enumerate(coord_tiendas):
        d = distancia_carretera(lat_d, lon_d, la, lo)
        bruta[j] = fact_d[(d <= RADIO_SERVICIO)].sum() / 1e6
    return bruta


def main():
    tiendas = pd.read_pickle('data/tiendas.pkl')
    demanda = pd.read_pickle('data/demanda_es.pkl')
    coord = tiendas[['lat', 'lon']].values

    bruta = captacion_bruta(coord, demanda)
    efectiva = captacion_efectiva(coord, demanda, exponente=2)

    out = pd.DataFrame({
        'ciudad': tiendas['ciudad'].values,
        'captacion_bruta_M': np.round(bruta, 2),
        'captacion_efectiva_M': np.round(efectiva, 2),
        'retiene_%': np.round(100 * efectiva / bruta, 0),
        'facturacion_M': np.round(tiendas['fact'].values, 1),
    })
    out.to_csv('salidas/captacion.csv', index=False)

    # Verificación: debe coincidir con el valor precalculado en el pickle
    err = np.abs(efectiva - tiendas['net_final'].values).max()
    print('Captación efectiva recalculada. Diferencia máxima frente al dato '
          f'precalculado: {err:.4f} M€ (debe ser ~0).')
    print('\nEjemplos (tiendas más y menos solapadas):')
    print(out.sort_values('retiene_%').head(4).to_string(index=False))
    print('...')
    print(out.sort_values('retiene_%').tail(3).to_string(index=False))


if __name__ == '__main__':
    main()
