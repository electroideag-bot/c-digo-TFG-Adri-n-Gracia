"""
fase3_prediccion.py
-------------------
Fase 3 (II). Predicción de la facturación para los candidatos del corredor
noroeste, proyección de maduración a cinco años y comparación con la red para
juzgar la viabilidad.

  1. Captación efectiva de cada candidato (reparto por gravedad frente a la
     red existente) y predicción con el modelo logarítmico, con intervalos de
     confianza del valor esperado y de predicción.
  2. Maduración: modelo de crecimiento acotado con dos parámetros
     interpretables, f1 (fracción del régimen el primer año) y lambda
     (fracción del hueco que se cierra cada año), en tres escenarios.
  3. Viabilidad: posición de la nueva tienda en la distribución de la red.

Entrada : data/tiendas.pkl, data/demanda_es.pkl
Salida  : salidas/prediccion.json
"""

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import distancia_carretera, captacion_efectiva, RADIO_SERVICIO

# Candidatos del corredor noroeste (código postal -> coordenadas)
CANDIDATOS = {
    'Villanueva del Pardillo': (40.4833, -3.9667),
    'Las Rozas':               (40.5578, -3.8917),
    'Moralzarzal':             (40.6782, -3.9707),
}

# Escenarios de maduración: (f1, lambda)
ESCENARIOS = {
    'Conservador': (0.55, 0.33),
    'Central':     (0.64, 0.44),
    'Optimista':   (0.73, 0.50),
}


def captacion_efectiva_candidato(vlat, vlon, coord_red, demanda):
    """Captación efectiva de un candidato, compartiendo mercado con la red existente."""
    coord = np.vstack([coord_red, [vlat, vlon]])
    return captacion_efectiva(coord, demanda, exponente=2)[-1]


def curva_maduracion(f1, lam, n=5):
    """Crecimiento acotado: f(t) = f(t-1) + lambda * (1 - f(t-1)), con f(1) = f1."""
    f = [f1]
    for _ in range(n - 1):
        f.append(f[-1] + lam * (1 - f[-1]))
    return f


def main():
    tiendas = pd.read_pickle('data/tiendas.pkl')
    demanda = pd.read_pickle('data/demanda_es.pkl')
    coord_red = tiendas[['lat', 'lon']].values

    y = tiendas['fact'].values
    lnet = np.log(tiendas['net_final'].values)
    costa = tiendas['costa'].values
    madrid = tiendas['madrid'].values

    # Modelo final (para predecir con intervalos)
    ols = sm.OLS(y, sm.add_constant(np.column_stack([lnet, costa, madrid]))).fit()

    resultados = {'predicciones': {}}
    for nombre, (la, lo) in CANDIDATOS.items():
        ec = captacion_efectiva_candidato(la, lo, coord_red, demanda)
        x = np.array([1, np.log(ec), 0, 1])          # costa = 0, madrid = 1
        pr = ols.get_prediction(x.reshape(1, -1))
        s68 = pr.summary_frame(alpha=0.32)
        s95 = pr.summary_frame(alpha=0.05)
        resultados['predicciones'][nombre] = dict(
            captacion_efectiva=round(float(ec), 1),
            central=round(float(s68['mean'][0]), 1),
            IC68=[round(float(s68['mean_ci_lower'][0]), 1),
                  round(float(s68['mean_ci_upper'][0]), 1)],
            IC95=[round(float(s95['mean_ci_lower'][0]), 1),
                  round(float(s95['mean_ci_upper'][0]), 1)],
            IP68=[round(float(s68['obs_ci_lower'][0]), 1),
                  round(float(s68['obs_ci_upper'][0]), 1)])

    central = resultados['predicciones']['Villanueva del Pardillo']['central']

    # Maduración a cinco años
    resultados['maduracion'] = {}
    for e, (f1, lam) in ESCENARIOS.items():
        fr = curva_maduracion(f1, lam)
        vals = [round(central * x, 1) for x in fr]
        resultados['maduracion'][e] = dict(f1=f1, lam=lam, anios=vals,
                                            acumulado=round(sum(vals), 1))

    # Viabilidad: posición de la nueva tienda en la red
    f = tiendas['fact'].values
    resultados['viabilidad'] = dict(
        prediccion_regimen=central,
        percentil_regimen=round(float(100 * (f < central).mean())),
        mediana_red=round(float(np.median(f)), 1),
        media_red=round(float(f.mean()), 1),
        media_arquetipo_periferico=round(
            float(tiendas.loc[tiendas['cl'] == tiendas['cl'].mode()[0], 'fact'].mean()), 1),
        tiendas_por_debajo_del_anio1=int(
            (f < resultados['maduracion']['Central']['anios'][0]).sum()),
        n_tiendas=int(len(f)))

    json.dump(resultados, open('salidas/prediccion.json', 'w'),
              ensure_ascii=False, indent=2)

    # --- Resumen por pantalla ------------------------------------------------
    print('Predicción por candidato (costa = 0, madrid = 1):')
    for k, v in resultados['predicciones'].items():
        print(f"  {k:24} captación efectiva {v['captacion_efectiva']:>4} -> "
              f"{v['central']:>4} M€  IC68 {v['IC68']}  IC95 {v['IC95']}")
    print('\nMaduración a cinco años (M€):')
    for e, v in resultados['maduracion'].items():
        print(f"  {e:12} (f1={v['f1']}, lambda={v['lam']}): {v['anios']}  "
              f"acumulado={v['acumulado']}")
    viab = resultados['viabilidad']
    print(f"\nViabilidad: {viab['prediccion_regimen']} M€ = percentil "
          f"{viab['percentil_regimen']} de la red (mediana {viab['mediana_red']}, "
          f"media arquetipo periférico {viab['media_arquetipo_periferico']}).")
    print(f"El primer año supera la facturación de régimen de "
          f"{viab['tiendas_por_debajo_del_anio1']} de {viab['n_tiendas']} tiendas.")


if __name__ == '__main__':
    main()
