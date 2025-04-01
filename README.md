# 🧠 Tradambot – AI-Powered Solana Crypto Scalping Bot

**Tradambot** is an automated trading bot that executes high-frequency trades on the Solana blockchain using advanced technical indicators, machine learning (LightGBM), and Optuna-tuned predictive modeling. It dynamically selects the best coins, monitors real-time price action, and uses a hybrid rule-based + model-driven strategy to buy and sell with high precision.

---

## 📌 Features

- ✅ Executes buy/sell trades using Jupiter Aggregator
- ✅ Uses real-time candlestick data from CryptoCompare
- ✅ LightGBM model trained with Optuna hyperparameter tuning
- ✅ Class imbalance handled via SMOTETomek
- ✅ Rule-based signal generation (RSI, OBV, Stochastic, MACD, EMA, Bollinger Bands, ADX)
- ✅ Auto-retraining when model is missing or inaccurate
- ✅ Trailing stop-loss to protect profits
- ✅ Persistent position tracking via `position.json`
- ✅ Trade history logging (`trade_log.json`)
- ✅ Multicoin support via `.env` symbol settings
- ✅ Runs on interval via `apscheduler` with async data fetch

---

## 📊 Indicators Used

- **RSI** – Overbought/Oversold
- **Stochastic %K/D** – Crossovers
- **MACD** – Trend momentum
- **OBV** – Volume confirmation
- **EMA 5 / EMA 20** – Short & Medium trend
- **ADX** – Trend strength
- **Bollinger Bands** – Reversals & volatility

---

## 🛠 Technologies

- Python 3.10+
- LightGBM
- Optuna (Hyperparameter tuning)
- SMOTETomek (imbalanced-learn)
- Solana SDK (`solana` & `solders`)
- TA-lib via `ta`
- APScheduler (Job Scheduling)
- Jupiter Aggregator API (for swaps)
- CryptoCompare API (Market data)

---

## ⚙️ Configuration

Edit your `.env` file:

```env
PRIMARY_MINT=SOL
SECONDARY_MINT=YOUR_TOKEN_ADDRESS
SECONDARY_MINT_SYMBOL=ABC
API_KEY2=your_cryptocompare_api_key


🙏 Acknowledgements
Special thanks to Soltrade, which served as an inspiration and foundation during the research and development of Tradambot. Portions of the strategy logic, structure, and design were adapted and enhanced based on insights gained from the Soltrade project.
