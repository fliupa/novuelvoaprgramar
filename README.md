[README.md](https://github.com/user-attachments/files/22375629/README.md)
# Sistema de Trading Algorítmico con Machine Learning

## Descripción

Este proyecto implementa un sistema de trading algorítmico avanzado que utiliza técnicas de machine learning para predecir movimientos del mercado de valores de la Bolsa Mexicana de Valores (BMV). El sistema analiza múltiples empresas y genera señales de trading basadas en modelos predictivos.

## Características

- **Análisis de múltiples empresas**: FEMSAUBD, GAPB, GMEXICOB, ALPEKA, CEMEXCPO
- **Múltiples modelos de ML**: Random Forest, Gradient Boosting, Redes Neuronales, LSTM
- **Análisis individual y combinado**: Modos de análisis por ticker individual y conjunto
- **Backtesting realista**: Simulación de trading con comisiones y slippage
- **Métricas de performance**: Sharpe Ratio, retornos, accuracy de predicciones
- **Visualización avanzada**: Gráficos de performance y correlaciones

## Tecnologías Utilizadas

- **Python 3.12+**
- **Pandas & NumPy**: Manipulación de datos
- **Scikit-learn**: Modelos de machine learning tradicionales
- **TensorFlow/Keras**: Redes neuronales y LSTM
- **Matplotlib & Seaborn**: Visualización de datos
- **yFinance**: Descarga de datos de mercado

## Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias principales

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
matplotlib>=3.7.0
seaborn>=0.12.0
yfinance>=0.2.0
```

## Uso

### Ejecución básica

```python
# Importar la clase principal
from acciones3 import TradingBotML

# Lista de empresas a analizar
empresas_bmv = [
    "FEMSAUBD.MX",
    "GAPB.MX", 
    "GMEXICOB.MX",
    "ALPEKA.MX",
    "CEMEXCPO.MX"
]

# Análisis individual por ticker
bot_individual = TradingBotML(tickers=empresas_bmv, mode="individual")
bot_individual.create_features(lookback=20, forecast_horizon=1)
bot_individual.train_supervised_models()
bot_individual.make_predictions()

# Análisis combinado
bot_combined = TradingBotML(tickers=empresas_bmv, mode="combined")
bot_combined.create_features(lookback=20, forecast_horizon=1)
bot_combined.train_supervised_models()
bot_combined.make_predictions()
```

### Backtesting

```python
# Backtesting con capital inicial de $1,000,000
initial_capital = 1000000

# Backtesting individual
for ticker in empresas_bmv:
    backtest = bot_individual.backtest_strategy_realistic(
        best_model, 
        initial_capital=initial_capital, 
        ticker=ticker
    )

# Backtesting combinado
backtest_combined = bot_combined.backtest_strategy_realistic(
    best_model_combined, 
    initial_capital=initial_capital
)
```

## Estructura del Proyecto

```
invs/
├── acciones3.ipynb          # Notebook principal con implementación completa
├── acciones3.py            # Versión Python del sistema
├── README.md               # Este archivo
├── requirements.txt        # Dependencias del proyecto
├── *.csv                   # Datos históricos de empresas BMV
└── correlation_matrix.png  # Matriz de correlaciones generada
```

## Métodos Principales

### TradingBotML

- `__init__()`: Inicializa el bot con datos históricos
- `create_features()`: Crea características técnicas para ML
- `train_supervised_models()`: Entrena múltiples modelos de ML
- `make_predictions()`: Genera predicciones para cada modelo
- `evaluate_models()`: Evalúa el performance de los modelos
- `backtest_strategy_realistic()`: Backtesting con condiciones realistas
- `plot_results()`: Visualiza resultados de trading

### Características Técnicas Implementadas

- **Indicadores técnicos**: RSI, MACD, Bollinger Bands, Medias Móviles
- **Features de precio**: Returns, volatilidad, volumen
- **Features temporales**: Día de la semana, mes, trimestre
- **Normalización**: MinMax scaling para entrenamiento de modelos

## Modelos de Machine Learning

1. **Random Forest Regressor**
2. **Gradient Boosting Regressor** 
3. **Multi-layer Perceptron (MLP)**
4. **Long Short-Term Memory (LSTM)**
5. **K-means Clustering** (para análisis no supervisado)
6. **PCA** (para reducción de dimensionalidad)

## Métricas de Evaluación

- **Accuracy de dirección**: Precisión en predecir dirección del precio
- **Mean Squared Error (MSE)**: Error cuadrático medio
- **Mean Absolute Error (MAE)**: Error absoluto medio
- **Sharpe Ratio**: Risk-adjusted returns
- **Total Return**: Retorno total de la estrategia
- **Number of Trades**: Cantidad de operaciones realizadas

## Resultados

El sistema genera:

1. **Reportes de performance** por ticker y modelo
2. **Matrices de correlación** entre empresas
3. **Gráficos de equity curves**
4. **Métricas comparativas** vs Buy & Hold
5. **Señales de trading** con niveles de confianza

## Advertencias

⚠️ **Este es un proyecto educativo** - No utilizar para trading real sin validación adicional.

- Los resultados pasados no garantizan performance futura
- Incluye supuestos simplificados sobre ejecución de órdenes
- No considera todos los costos de transacción reales
- Requiere validación out-of-sample para uso real


