"""
fase3_figuras.py
----------------
Genera las figuras de la Fase 3 que aparecen en la memoria. Debe ejecutarse
después de fase3_modelos.py y fase3_prediccion.py, ya que lee sus salidas.

Salida : salidas/figuras/*.png
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from xgboost import XGBRegressor

# Paleta de la memoria
TEAL, AMBER, GREEN, RED, GREY = '#1F4E5F', '#C98A2B', '#3E7C59', '#B83232', '#8A8A8A'
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False})
DIR = 'salidas/figuras'
os.makedirs(DIR, exist_ok=True)


def cargar():
    tiendas = pd.read_pickle('data/tiendas.pkl')
    res = json.load(open('salidas/resultados_modelos.json'))
    pred = json.load(open('salidas/prediccion.json'))
    return tiendas, res, pred


def figura_lasso(res):
    lp = pickle.load(open('salidas/lasso_path.pkl', 'rb'))
    coefs = np.array(lp['coefs']); feats = lp['feats']
    xx = np.abs(coefs).sum(axis=0)
    etiq = {'log_captacion': 'log(captación)', 'costa': 'costa', 'madrid': 'madrid',
            'nCP15': 'nº CP', 'HHI15': 'concentración', 'pob15': 'población',
            'firms15': 'empresas', 'superficie': 'superficie'}
    keep = {'log_captacion', 'costa', 'madrid'}
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for i, f in enumerate(feats):
        col = TEAL if f == 'log_captacion' else AMBER if f == 'costa' else GREEN if f == 'madrid' else GREY
        ax.plot(xx, coefs[i], color=col, lw=2.2 if f in keep else 1.0,
                alpha=1.0 if f in keep else 0.55)
        ax.text(xx[-1] + 0.4, coefs[i, -1], etiq[f], fontsize=8.5, va='center',
                color=col, fontweight='bold' if f in keep else 'normal')
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_xlabel('Complejidad del modelo (norma L1 de los coeficientes)')
    ax.set_ylabel('Coeficiente estandarizado')
    ax.set_title('Senda de coeficientes de LASSO', fontsize=12, pad=8)
    ax.set_xlim(0, xx[-1] * 1.35); ax.grid(alpha=0.2)
    _guardar(fig, 'lasso')


def figura_estructural(tiendas):
    y = tiendas['fact'].values; net = tiendas['net_final'].values
    lnet = np.log(net); costa = tiendas['costa'].values; madrid = tiendas['madrid'].values
    m = sm.OLS(y, sm.add_constant(np.column_stack([lnet, costa, madrid]))).fit()
    a, b, c, d = m.params
    grid = np.linspace(net.min(), net.max(), 200); lg = np.log(grid)
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(grid, a + b * lg, color=TEAL, lw=2, label='Interior no metropolitano')
    ax.plot(grid, a + b * lg + d, color=GREEN, lw=2, label='Área de Madrid (+%.1f M€)' % d)
    ax.plot(grid, a + b * lg + c, color=AMBER, lw=2, label='Costa mediterránea (+%.1f M€)' % c)
    gr = (costa == 0) & (madrid == 0); gm = madrid == 1; gc = costa == 1
    ax.scatter(net[gr], y[gr], color=TEAL, s=40, zorder=4, edgecolor='white', lw=0.5)
    ax.scatter(net[gm], y[gm], color=GREEN, s=52, zorder=4, edgecolor='white', lw=0.6, marker='s')
    ax.scatter(net[gc], y[gc], color=AMBER, s=52, zorder=4, edgecolor='white', lw=0.6, marker='D')
    ax.set_xlabel('Captación efectiva (M€)'); ax.set_ylabel('Facturación anual (M€)')
    ax.set_title('Modelo final: efecto de la costa y de Madrid sobre la captación efectiva',
                 fontsize=11.5, pad=8)
    ax.legend(fontsize=9.5, loc='lower right'); ax.grid(alpha=0.2)
    _guardar(fig, 'estructural')


def figura_importancia_rf(res):
    imp = res['importancias']['RandomForest']
    feat = list(imp.keys()); vals = [imp[k] * 100 for k in feat]
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    ax.barh([feat[i] for i in order], [vals[i] for i in order], color=TEAL)
    for i, v in enumerate([vals[i] for i in order]):
        ax.text(v + 1, i, f'{v:.0f}%', va='center', fontsize=10)
    ax.set_xlabel('Importancia (%)'); ax.set_xlim(0, 100)
    ax.set_title('Random Forest: importancia de las variables', fontsize=12, pad=8)
    ax.grid(axis='x', alpha=0.25)
    _guardar(fig, 'rf_importancia')


def figura_xgb(tiendas, res):
    y = tiendas['fact'].values
    Xtree = np.column_stack([tiendas['net_final'].values, tiendas['costa'].values,
                             tiendas['madrid'].values, tiendas['sup'].values])
    m = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0)
    pt = m.fit(Xtree, y).predict(Xtree)
    pc = cross_val_predict(XGBRegressor(n_estimators=300, max_depth=3,
                                        learning_rate=0.05, random_state=0),
                           Xtree, y, cv=LeaveOneOut())
    r2t = res['contraste']['XGBoost']['R2train']; r2c = res['contraste']['XGBoost']['R2cv']
    fig, ax = plt.subplots(figsize=(8.0, 5.0)); lim = [0, 85]
    ax.plot(lim, lim, color=GREY, ls='--', lw=1)
    ax.scatter(y, pt, color=TEAL, s=42, edgecolor='white', lw=0.5,
               label=f'En entrenamiento (R²={r2t:.2f})'.replace('.', ','))
    ax.scatter(y, pc, color=RED, s=42, marker='^', edgecolor='white', lw=0.5,
               label=f'En validación cruzada (R²={r2c:.2f})'.replace('.', ','))
    for i in range(len(y)):
        ax.plot([y[i], y[i]], [pt[i], pc[i]], color=GREY, alpha=0.25, lw=0.7)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel('Facturación real (M€)'); ax.set_ylabel('Facturación predicha (M€)')
    ax.set_title('XGBoost: ajuste dentro frente a fuera de muestra', fontsize=12, pad=8)
    ax.legend(fontsize=9.5, loc='upper left'); ax.grid(alpha=0.2)
    _guardar(fig, 'xgb')


def figura_contraste(res):
    c = res['contraste']
    modelos = ['Regresión\nlineal', 'Regresión\nlogarítmica', 'Árbol',
               'Random\nForest', 'XGBoost']
    claves = ['Lineal', 'Logaritmica', 'Arbol', 'RandomForest', 'XGBoost']
    r2cv = [c[k]['R2cv'] for k in claves]; r2tr = [c[k]['R2train'] for k in claves]
    x = np.arange(5); w = 0.38
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.bar(x - w / 2, r2tr, w, label='Entrenamiento', color=GREY)
    ax.bar(x + w / 2, r2cv, w, label='Validación cruzada', color=TEAL)
    for i in range(5):
        ax.text(x[i] - w / 2, r2tr[i] + 0.01, f'{r2tr[i]:.2f}'.replace('.', ','),
                ha='center', fontsize=8.5)
        ax.text(x[i] + w / 2, r2cv[i] + 0.01, f'{r2cv[i]:.2f}'.replace('.', ','),
                ha='center', fontsize=8.5, color=TEAL)
    ax.set_xticks(x); ax.set_xticklabels(modelos, fontsize=9.5)
    ax.set_ylabel('R²'); ax.set_ylim(0, 1.08)
    ax.set_title('Ajuste en entrenamiento frente a generalización', fontsize=12, pad=8)
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.25)
    _guardar(fig, 'contraste')


def figura_prediccion(tiendas, pred):
    y = tiendas['fact'].values; net = tiendas['net_final'].values
    lnet = np.log(net); costa = tiendas['costa'].values; madrid = tiendas['madrid'].values
    m = sm.OLS(y, sm.add_constant(np.column_stack([lnet, costa, madrid]))).fit()
    a, b, c, d = m.params
    grid = np.linspace(net.min(), 40, 200); lg = np.log(grid)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.plot(grid, a + b * lg + d, color=GREEN, lw=2, label='Curva del área de Madrid')
    ax.plot(grid, a + b * lg, color=TEAL, lw=1.5, ls='--', alpha=0.6,
            label='Interior no metropolitano')
    ax.scatter(net[madrid == 1], y[madrid == 1], color=GREEN, s=48, marker='s',
               edgecolor='white', lw=0.6, zorder=4)
    pv = pred['predicciones']['Villanueva del Pardillo']
    ec, cen = pv['captacion_efectiva'], pv['central']
    ax.errorbar(ec, cen, yerr=[[cen - pv['IC68'][0]], [pv['IC68'][1] - cen]],
                fmt='o', color=RED, ms=11, capsize=6, lw=2, zorder=6,
                label='Villanueva del Pardillo')
    ax.annotate(f'{cen:.0f} M€\nIC 68% [{pv["IC68"][0]:.0f}; {pv["IC68"][1]:.0f}]',
                (ec, cen), xytext=(ec + 6, cen - 8), fontsize=9.5, color=RED,
                arrowprops=dict(arrowstyle='->', color=RED))
    for nom, mk in [('Las Rozas', 'v'), ('Moralzarzal', '^')]:
        p2 = pred['predicciones'][nom]
        ax.scatter(p2['captacion_efectiva'], p2['central'], color='#6A4C93', s=60,
                   marker=mk, zorder=5, edgecolor='white', lw=0.6)
        ax.annotate(nom, (p2['captacion_efectiva'], p2['central']),
                    xytext=(p2['captacion_efectiva'] + 1, p2['central'] + 2),
                    fontsize=8, color='#6A4C93')
    ax.set_xlim(0, 40); ax.set_xlabel('Captación efectiva (M€)')
    ax.set_ylabel('Facturación anual (M€)')
    ax.set_title('Predicción de la nueva tienda sobre la curva del área de Madrid',
                 fontsize=11.5, pad=8)
    ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=0.2)
    _guardar(fig, 'prediccion')


def figura_maduracion(pred):
    mad = pred['maduracion']; central = pred['viabilidad']['prediccion_regimen']
    anios = [1, 2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for e, col in [('Conservador', GREY), ('Central', TEAL), ('Optimista', GREEN)]:
        v = mad[e]['anios']
        ax.plot(anios, v, marker='o', color=col, lw=2, ms=6,
                label=f"{e} (f₁={mad[e]['f1']}, λ={mad[e]['lam']})")
    ax.axhline(central, color=RED, ls='--', lw=1.3)
    ax.text(5.05, central, f'régimen {central:.0f} M€', color=RED, fontsize=9, va='center')
    ax.set_xticks(anios); ax.set_xlabel('Año desde la apertura')
    ax.set_ylabel('Facturación anual (M€)'); ax.set_ylim(12, 32)
    ax.set_title('Trayectoria de maduración de la nueva tienda', fontsize=12, pad=8)
    ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=0.25)
    _guardar(fig, 'maduracion')


def _guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(f'{DIR}/{nombre}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {nombre}.png')


def main():
    tiendas, res, pred = cargar()
    print('Generando figuras en', DIR)
    figura_lasso(res)
    figura_estructural(tiendas)
    figura_importancia_rf(res)
    figura_xgb(tiendas, res)
    figura_contraste(res)
    figura_prediccion(tiendas, pred)
    figura_maduracion(pred)


if __name__ == '__main__':
    main()
