#%%
# Sophy4 Trading Framework

## Overzicht

**Sophy4** is een modulair Python-framework voor algoritmische handel, gericht op backtesting, live trading, risicobeheer, strategie-optimalisatie en FTMO-compliance. Het ondersteunt zowel traditionele technische strategieën als machine learning modellen, en is ontworpen voor schaalbaarheid, onderhoudbaarheid en professionele evaluatie van trading performance.

---

## 🔑 Kerneigenschappen

- **Modulair ontwerp** – elk domein (strategie, risk, FTMO, backtest, etc.) is gescheiden
- **Backtesting engine** – snel en robuust (op basis van VectorBT)
- **Live trading integratie** – via MetaTrader5
- **Risicobeheer** – zoals drawdown-controle, daily loss limits, risk-per-trade
- **Monitoring & logging** – inclusief performance tracking
- **FTMO-regelcontrole** – automatische evaluatie van de compliance-eisen
- **Machine Learning integratie** – LSTM en DAX spike voorspellers
- **CLI & Config files** – voor snelle workflows

---

## 📁 Projectstructuur

```plaintext
Sophy4/
│
├── backtest/                   # Backtesting functionaliteit
│   ├── backtest.py
│   └── data_loader.py
│
├── live/                       # Live trading componenten
│
├── risk/                       # Risicobeheerlogica
│   └── risk_management.py
│
├── monitor/                    # Realtime performance monitoring
│
├── ftmo_compliance/            # FTMO compliance-checks
│
├── strategies/                 # Strategieën en modeltraining
│   ├── base_strategy.py
│   ├── bollong.py
│   ├── bollong_vectorized.py
│   ├── simple_order_block.py
│   ├── order_block_lstm_strategy.py
│   ├── train_lstm_model.py
│   └── train_dax_spike_model.py
│
├── utils/                      # Hulpmethoden, logica
│
├── optimization/               # Parameter search, grid search, etc.
│
├── models/                     # Getrainde modellen
│
├── results/                    # Result output, rapportages
│
├── docs/                       # Projectdocumentatie
│
├── main.py                     # CLI entrypoint
├── cli.py                      # Typer CLI commands
├── config.py                   # Algemene configuraties
├── timeframe_config.json       # Per-timeframe instellingen
└── pyproject.toml              # Dependency declaratie
