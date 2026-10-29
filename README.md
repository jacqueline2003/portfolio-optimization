# Optimización de Portafolios: Clásica, Machine Learning y Deep Learning

Proyecto personal de optimización de portafolios de inversión usando tres enfoques: teoría clásica de Markowitz, modelos de Machine Learning y redes neuronales LSTM. El objetivo es comparar las estrategias en un backtest con rebalanceo mensual sobre datos reales del mercado estadounidense.

## Estructura

| Notebook | Descripción |
|----------|-------------|
| `Part1_Markowitz_Portfolio_Optimization.ipynb` | Markowitz clásico: frontera eficiente, portafolio de mínima varianza y máximo Sharpe Ratio con simulación Monte Carlo |
| `Part2_ML_Portfolio_Optimization.ipynb` | Random Forest y XGBoost para predecir retornos a 21 días y optimizar portafolios |
| `Part3_DL_LSTM_Portfolio_Optimization.ipynb` | Redes LSTM (Long Short-Term Memory) para capturar dependencias temporales en retornos |
| `Part4_Backtesting_Strategy_Comparison.ipynb` | Backtest comparativo con rebalanceo mensual durante 2024 |

## Datos

- **Activos:** AAPL, MSFT, JPM, JNJ, XOM, KO, AMZN, NVDA, CAT, PG
- **Benchmark:** SPY (S&P 500)
- **Periodo de entrenamiento:** 2020–2023
- **Periodo de backtest:** 2024
- **Fuente:** Yahoo Finance (vía `yfinance`)

## Metodología

### Part 1 — Markowitz
- Retornos diarios y matriz de varianza-covarianza
- Simulación Monte Carlo (50,000 portafolios con distribución de Dirichlet)
- Optimización numérica con `scipy.optimize` (SLSQP)
- Frontera eficiente y Capital Market Line (Rf = 5%)

### Part 2 — Machine Learning
- 12 features técnicas: momentum (1d, 5d, 10d, 21d), medias móviles, RSI, MACD, Bollinger %B, volatilidad
- Target: retorno a 21 días
- Random Forest (200 árboles, max_depth=6)
- XGBoost (200 estimadores, learning_rate=0.05)
- Retornos predichos alimentan optimización de Markowitz

### Part 3 — Deep Learning (LSTM)
- Secuencias de 60 días × 12 features como entrada
- Arquitectura: LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)
- Early stopping (patience=10)
- StandardScaler ajustado solo en entrenamiento

### Part 4 — Backtesting
- Rebalanceo mensual (primer día hábil de cada mes)
- 6 estrategias comparadas: Equal Weight, Histórico, RF, XGBoost, LSTM, SPY
- Métricas: retorno anualizado, volatilidad, Sharpe, Sortino, Max Drawdown, Calmar, turnover

## Requisitos

```
python >= 3.9
numpy
pandas
matplotlib
scipy
scikit-learn
xgboost
tensorflow
yfinance
```

Instalación rápida con Anaconda:
```bash
pip install yfinance xgboost tensorflow
```

## Autor

Jacqueline Alessandra Espinosa Celestino  
Licenciatura en Actuaría — FCFM, UANL  
Octubre 2026
