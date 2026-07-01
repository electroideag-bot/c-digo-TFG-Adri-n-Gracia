"""
fase3_modelos.py
----------------
Fase 3 (I). Ajuste y contraste de los modelos de facturación sobre la red
completa de 38 tiendas.

  1. Selección de variables con LASSO (variables de tamaño frente a los dos
     controles estructurales, costa y Madrid). LASSO conserva la captación,
     la costa y Madrid, y anula las variables de tamaño.
  2. Modelo final: regresión logarítmica de la captación EFECTIVA con los dos
     controles estructurales.
  3. Contraste frente a un abanico de modelos de complejidad creciente
     (lineal, logarítmico, árbol, Random Forest, XGBoost) por validación
     cruzada dejando uno fuera (LOOCV).

Todas las variables explicativas son ex ante (conocidas antes de abrir).

Entrada : data/tiendas.pkl, data/variables_tamano.pkl
Salida  : salidas/resultados_modelos.json, salidas/lasso_path.pkl
"""

import json
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LassoCV, lars_path
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor


def loocv(modelo, X, y):
    """R2, RMSE y MAE por validación cruzada dejando uno fuera, y R2 en entrenamiento."""
    pred = cross_val_predict(modelo, X, y, cv=LeaveOneOut())
    r2cv = r2_score(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae = mean_absolute_error(y, pred)
    r2tr = modelo.fit(X, y).score(X, y)
    return dict(R2cv=r2cv, RMSEcv=rmse, MAEcv=mae, R2train=r2tr)


def main():
    tiendas = pd.read_pickle('data/tiendas.pkl')
    rich = pd.read_pickle('data/variables_tamano.pkl')

    y = tiendas['fact'].values          # facturación anual (M€)
    costa = tiendas['costa'].values
    madrid = tiendas['madrid'].values
    sup = tiendas['sup'].values
    net = tiendas['net_final'].values   # captación efectiva (M€)
    lnet = np.log(net)                  # log de la captación efectiva

    resultados = {}

    # --- 1. Selección de variables con LASSO ---------------------------------
    size = rich[['nCP15', 'HHI15', 'pob15', 'firms15']].reset_index(drop=True).copy()
    size['superficie'] = sup
    size['log_captacion'] = np.log(tiendas['capt'].values)   # captación bruta
    size['costa'] = costa
    size['madrid'] = madrid
    feats = ['nCP15', 'HHI15', 'pob15', 'firms15', 'superficie',
             'log_captacion', 'costa', 'madrid']
    Xs = StandardScaler().fit_transform(size[feats].values)

    lasso = LassoCV(cv=5, random_state=0, max_iter=300000).fit(Xs, y)
    conserva = [f for f, c in zip(feats, lasso.coef_) if abs(c) > 1e-6]
    resultados['lasso'] = dict(alpha=float(lasso.alpha_), conserva=conserva)

    # Senda de coeficientes (para la figura de la memoria)
    alphas, _, coefs = lars_path(Xs, y - y.mean(), method='lasso')
    pickle.dump({'coefs': coefs.tolist(), 'feats': feats},
                open('salidas/lasso_path.pkl', 'wb'))

    # --- 2. Modelo final: log de la captación efectiva + costa + Madrid ------
    X = np.column_stack([lnet, costa, madrid])
    ols = sm.OLS(y, sm.add_constant(X)).fit()
    resultados['modelo_final'] = dict(
        coeficientes=dict(zip(['const', 'ln_captacion_efectiva', 'costa', 'madrid'],
                              np.round(ols.params, 3).tolist())),
        p_valores=dict(zip(['const', 'ln_captacion_efectiva', 'costa', 'madrid'],
                           np.round(ols.pvalues, 4).tolist())),
        R2=round(ols.rsquared, 3), R2adj=round(ols.rsquared_adj, 3),
        s=round(np.sqrt(ols.scale), 2))

    # --- 3. Contraste de modelos por LOOCV -----------------------------------
    Xtree = np.column_stack([net, costa, madrid, sup])   # árboles usan la captación sin log
    contraste = {
        'Lineal': loocv(LinearRegression(), np.column_stack([net, costa, madrid]), y),
        'Logaritmica': loocv(LinearRegression(), np.column_stack([lnet, costa, madrid]), y),
        'Arbol': loocv(DecisionTreeRegressor(max_depth=3, random_state=0), Xtree, y),
        'RandomForest': loocv(RandomForestRegressor(n_estimators=400, random_state=0), Xtree, y),
        'XGBoost': loocv(XGBRegressor(n_estimators=300, max_depth=3,
                                      learning_rate=0.05, random_state=0), Xtree, y),
    }
    resultados['contraste'] = {k: {kk: round(vv, 3) for kk, vv in v.items()}
                               for k, v in contraste.items()}

    # Importancias de los conjuntos de árboles
    rf = RandomForestRegressor(n_estimators=400, random_state=0).fit(Xtree, y)
    xgb = XGBRegressor(n_estimators=300, max_depth=3,
                       learning_rate=0.05, random_state=0).fit(Xtree, y)
    nombres = ['captacion_efectiva', 'costa', 'madrid', 'superficie']
    resultados['importancias'] = dict(
        RandomForest=dict(zip(nombres, np.round(rf.feature_importances_, 3).tolist())),
        XGBoost=dict(zip(nombres, np.round(xgb.feature_importances_, 3).tolist())))

    json.dump(resultados, open('salidas/resultados_modelos.json', 'w'),
              ensure_ascii=False, indent=2)

    # --- Resumen por pantalla ------------------------------------------------
    print('LASSO conserva:', conserva)
    print('\nModelo final (facturación = const + b1*ln(captación efectiva) '
          '+ b2*costa + b3*madrid):')
    for k, v in resultados['modelo_final']['coeficientes'].items():
        print(f'  {k:24}: {v:8}   (p = {resultados["modelo_final"]["p_valores"][k]})')
    print(f"  R2 = {resultados['modelo_final']['R2']}, "
          f"s = {resultados['modelo_final']['s']} M€")
    print('\nContraste por validación cruzada (LOOCV):')
    print(f"  {'Modelo':14}{'R2cv':>8}{'RMSEcv':>9}{'MAEcv':>8}{'R2train':>9}")
    for k, v in resultados['contraste'].items():
        print(f"  {k:14}{v['R2cv']:>8}{v['RMSEcv']:>9}{v['MAEcv']:>8}{v['R2train']:>9}")


if __name__ == '__main__':
    main()
