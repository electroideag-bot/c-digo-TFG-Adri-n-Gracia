# Código del análisis — TFG Obramat

Recomendación de ubicación y previsión de facturación de una nueva tienda
Obramat en la Comunidad de Madrid. Este repositorio contiene el código que
reproduce el análisis de las tres fases del trabajo.

## Estructura

```
codigo_TFG_Obramat/
├── utils.py                        Funciones comunes (distancias, captación efectiva)
├── fase1_captacion_efectiva.py     Captación bruta y efectiva (reparto por gravedad de Huff)
├── fase2_clustering.py             Tipología de tiendas (k-means, arquetipos)
├── fase3_modelos.py                Modelos, LASSO y contraste por validación cruzada
├── fase3_prediccion.py             Predicción, maduración a 5 años y viabilidad
├── fase3_figuras.py                Figuras de la Fase 3
├── data/                           Datos de entrada
│   ├── tiendas.pkl                 38 tiendas: coordenadas, captación bruta y efectiva,
│   │                               facturación, superficie, controles y cluster
│   ├── demanda_es.pkl              Demanda potencial por código postal (lat, lon, fact)
│   └── variables_tamano.pkl        Variables de tamaño candidatas (para LASSO)
└── salidas/                        Resultados y figuras (se generan al ejecutar)
```

Los datos de origen sin procesar son el fichero maestro de Obramat
(`datos_extrapolados_tfg.xlsx`, confidencial) y la base de códigos postales de
GeoNames (`ES.txt`). Los ficheros `.pkl` de `data/` son el resultado ya
depurado y geocodificado de esa preparación, e incluyen todo lo necesario para
reproducir el análisis sin volver a descargar la red viaria.

## Requisitos

```
pip install -r requirements.txt
```

## Reproducción

Ejecutar desde la carpeta `codigo_TFG_Obramat/`, en este orden:

```bash
python fase1_captacion_efectiva.py     # verifica la captación efectiva
python fase2_clustering.py             # arquetipos de tienda
python fase3_modelos.py                # modelos + LASSO + contraste  -> salidas/*.json
python fase3_prediccion.py             # predicción + maduración + viabilidad
python fase3_figuras.py                # figuras (requiere las dos salidas anteriores)
```

## Notas metodológicas

- **Captación efectiva.** La demanda a menos de 15 km por carretera de cada
  tienda se reparte entre las tiendas que la cubren mediante el modelo de
  gravitación de Huff (peso 1/d²), descontando la canibalización interna. Es la
  variable de mercado de la regresión.
- **Modelo final.** Regresión logarítmica de la captación efectiva con dos
  controles estructurales (costa y pertenencia a Madrid), elegida por
  validación cruzada dejando uno fuera (LOOCV) frente a árbol, Random Forest y
  XGBoost.
- **Maduración.** Modelo de crecimiento acotado con dos parámetros
  interpretables, f₁ (fracción del régimen el primer año) y λ (fracción del
  hueco que se cierra cada año), en tres escenarios de velocidad.
- **Reproducibilidad.** Todas las semillas están fijadas (`random_state=0`),
  por lo que los resultados son exactos.
