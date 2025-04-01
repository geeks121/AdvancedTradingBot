from imblearn.combine import SMOTETomek
import lightgbm as lgb
import ta
import asyncio
from tradambot.wallet import find_balance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
import numpy as np
import requests
import pandas as pd
import joblib
import os
from tradambot.transactions import perform_swap, market
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
from tradambot.indicators import calculate_ema, calculate_rsi, calculate_macd, calculate_bbands, calculate_adx, calculate_stoch, calculate_obv
from tradambot.log import log_general, log_transaction
from tradambot.config import config
#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json
import random  
from imblearn.over_sampling import SMOTE
from collections import Counter

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

from dotenv import load_dotenv
import os

load_dotenv()
coin_symbol = os.getenv("SECONDARY_MINT_SYMBOL")

API_KEY2 = os.getenv("API_KEY2")
TRADE_LOG_FILE = "trade_log.json"
retrained = False

def save_trade_log(trade_entry):
    """Append trade details to a log file for reporting."""
    if os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE, "r") as f:
            trade_log = json.load(f)
    else:
        trade_log = []

    trade_log.append(trade_entry)

    with open(TRADE_LOG_FILE, "w") as f:
        json.dump(trade_log, f, indent=4)

def run_buy_script(token_address):
    import subprocess
    from datetime import datetime

    script_path = "buy_tokens.py"
    LOG_FILE = "tradambot.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    command = ["python3", script_path, token_address]

    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"\n--- {timestamp} | BUY attempt for {token_address} ---\n")
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            log_file.write(result.stdout)
            log_file.write(result.stderr)
            print(result.stdout, result.stderr)

            # ❌ Detect known buy failures
            if any(err in result.stdout.lower() for err in [
                "no balance available",
                "failed to fetch quote",
                "failed to prepare swap",
                "swap failed",
                "error",
            ]):
                print(f"⚠️ Buy skipped for {token_address} due to failure or zero balance.")
                log_file.write(f"⚠️ Buy skipped for {token_address} due to failure or zero balance.\n")
                return False

            # ✅ Detect actual successful buy
            if any(success in result.stdout.lower() for success in [
                "swap completed",
                "transaction finished",
                "transaction id:",
                "✅"
            ]):
                print(f"✅ Buy successful for {token_address}")
                return True

        except Exception as e:
            error_msg = f"❌ Exception during BUY: {e}"
            print(error_msg)
            log_file.write(f"{error_msg}\n")

    print(f"❌ Buy failed for {token_address}")
    return False


def run_sell_script(token_address):
    import subprocess, time
    from datetime import datetime

    script_path = "/sell_tokens.py"
    LOG_FILE = "tradambot.log"
    command = ["python3", script_path, token_address]

    with open(LOG_FILE, "a") as log_file:
        for attempt in range(1, 10):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"\n--- {timestamp} | SELL attempt {attempt} for {token_address} ---\n")

            try:
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                log_file.write(result.stdout)
                log_file.write(result.stderr)
                print(result.stdout, result.stderr)

                # ❌ Detect if there's no token balance to sell
                if "no balance available" in result.stdout.lower():
                    print(f"⚠️ Skipping sell for {token_address} due to zero balance.")
                    log_file.write(f"⚠️ Skipping sell for {token_address} due to zero balance.\n")
                    return False

                # ✅ Detect successful sell execution
                if (
                    "success" in result.stdout.lower()
                    or "✅" in result.stdout
                    or "swap completed" in result.stdout.lower()
                    or "transaction finished" in result.stdout.lower()
                    or "transaction id:" in result.stdout.lower()
                ):
                    print(f"✅ Sell successful for {token_address}")
                    return True

            except Exception as e:
                error_msg = f"❌ Exception during SELL: {e}"
                print(error_msg)
                log_file.write(f"{error_msg}\n")

            time.sleep(4)

    print(f"❌ All SELL attempts failed for {token_address}")
    return False


# ✅ Define expected feature set (to ensure training & prediction match)
EXPECTED_FEATURES = ['high', 'low', 'open', 'close', 'volumefrom', 'volumeto',
                     'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'ema5', 'ema20',
                     'upper_bband', 'lower_bband', 'rsi', 'adx', 'obv', 'minute']


POSITION_FILE = "position.json"

def get_position():
    """Retrieve position data from file and ensure all required fields exist."""
    if os.path.exists(POSITION_FILE):
        try:
            with open(POSITION_FILE, "r") as f:
                position = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            log_general.warning("⚠️ Position file is corrupted or missing. Resetting to default.")
            position = {}  # If corrupted, reset to empty dict
    else:
        position = {}  # Empty position

    # ✅ Ensure all required keys exist with default values
    default_position = {
        "is_open": False,
        "entry_price": None,
        "trade_amount": 0,  # Ensures trade_amount is always available
        "sl": None,  # Stop-loss
        "tp": None,  # Take-profit
        "trailing_stop": None  # Ensure trailing stop is included
    }
    
    for key, value in default_position.items():
        if key not in position:
            position[key] = value  # Fill missing fields

    return position

def save_position(position):
    """Safely save position data to file, ensuring no missing fields."""
    try:
        with open(POSITION_FILE, "w") as f:
            json.dump(position, f, indent=4)
        log_general.debug(f"✅ Position saved: {position}")
    except Exception as e:
        log_general.error(f"❌ Error saving position file: {e}")


# Simulated balance and P&L tracking
#virtual_balance = find_balance(config().secondary_mint)  # Starting with $1000
virtual_balance = float(find_balance(config().secondary_mint) or 0)  # Default to 0 if None
trades = []  # List to store completed trades for tracking P&L

def simulate_price_change(close_price):
    return close_price * (1 + (random.uniform(-0.005, 0.005)))  # Simulate small price change


API_KEY = ""
API_URL = "https://min-api.cryptocompare.com/data/v2/histominute"
REALTIME_PRICE_URL = "https://min-api.cryptocompare.com/data/price"

market('position.json')


# ✅ Fetch candlestick data and update latest market price
latest_market_price = None  # Global variable to store last fetched price

latest_market_price = None

def fetch_realtime_price():
    global latest_market_price
    try:
        params = {
            'fsym': config().secondary_mint_symbol,
            'tsyms': 'USD'
        }
        headers = {'authorization': f'Apikey {API_KEY2}'}
        response = requests.get(REALTIME_PRICE_URL, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        price_data = response.json()
        if 'USD' in price_data:
            latest_market_price = price_data['USD']
            log_general.debug(f"🔄 Updated real-time market price: {latest_market_price}")
            return latest_market_price
        else:
            log_general.warning("⚠️ USD price not found in response")
            return None
    except Exception as e:
        log_general.error(f"❌ Error fetching real-time price: {e}")
        return None

def fetch_candlestick():
    global latest_market_price
    log_general.debug("Fetching candlestick data from API...")

    url = API_URL
    headers = {'authorization': f'Apikey {API_KEY2}'}
    params = {
        'tsym': config().primary_mint_symbol,
        'fsym': config().secondary_mint_symbol,
        'limit': 2000,
        'aggregate': 1 
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()

        if data.get('Response') == 'Error':
            log_general.error(f"Error fetching candlestick data: {data.get('Message')}")
            return None

        candles = data.get('Data', {}).get('Data', [])
        if candles:
            latest_market_price = fetch_realtime_price()
        return candles

    except Exception as e:
        log_general.error(f"Exception in fetching candlestick data: {e}")
        latest_market_price = None
        return None

# ✅ Get new data for analysis
# ✅ Get new data for analysis
def get_new_data():
    candle_dict = fetch_candlestick()

    if not candle_dict:
        log_general.warning("No candlestick data retrieved. Skipping analysis.")
        return None  # ✅ Ensure function exits if no data is available

    new_df = pd.DataFrame(candle_dict)
    if new_df.empty:  # ✅ Check if the DataFrame is empty
        log_general.warning("Fetched candlestick data is empty. Skipping analysis.")
        return None

    new_df = new_df[['time', 'high', 'low', 'open', 'close', 'volumefrom', 'volumeto']]
    new_df['time'] = pd.to_datetime(new_df['time'], unit='s')

    new_df['hour'] = new_df['time'].dt.hour
    new_df['minute'] = new_df['time'].dt.minute
    new_df['dayofweek'] = new_df['time'].dt.dayofweek

    # ✅ Compute indicators safely
    try:
        new_df['stoch_k'], new_df['stoch_d'] = calculate_stoch(new_df)
        new_df['macd'], new_df['macd_signal'] = calculate_macd(new_df)
        new_df['ema5'] = calculate_ema(new_df, 5)
        new_df['ema20'] = calculate_ema(new_df, 20)
        new_df['upper_bband'], new_df['lower_bband'] = calculate_bbands(new_df)
        new_df['rsi'] = calculate_rsi(new_df)
        new_df['adx'] = calculate_adx(new_df)
        new_df['obv'] = calculate_obv(new_df)
    except Exception as e:
        log_general.error(f"Error calculating indicators: {e}")
        return None  # ✅ Exit function if indicators fail

    # ✅ Ensure required indicators exist before computing `target`
    required_indicators = {'rsi', 'macd', 'macd_signal', 'adx', 'obv', 'stoch_k', 'stoch_d', 'ema5', 'ema20', 'close', 'upper_bband', 'lower_bband'}
    if not required_indicators.issubset(new_df.columns):
        log_general.warning("Missing required indicators. Skipping analysis.")
        return None

    # ✅ Define buy and sell conditions using improved logic
   

    new_df['target'] = np.where(
    (
        # ✅ Must-Have 1: Bollinger Band Reversal
        ((new_df['close'] <= new_df['lower_bband']) & 
         (new_df['close'].shift(-1) > new_df['close']) &  # ✅ Next candle must be green
         (new_df['rsi'] > 50) &  # ✅ RSI Above 50
         (new_df['rsi'].shift(1) < new_df['rsi']) &  # ✅ RSI is Increasing
         (new_df['rsi'].shift(2) < new_df['rsi'].shift(1)) &  # ✅ RSI Increasing for 2 Candles
         (new_df['obv'] > new_df['obv'].shift(1))) |

        # ✅ Must-Have 2: Stochastic Bullish Crossover
        ((new_df['stoch_k'] < 40) & (new_df['stoch_d'] < 40) &  
         (new_df['stoch_k'] > new_df['stoch_d']) & 
         (new_df['obv'] > new_df['obv'].shift(1)))
    ) 
    &
    (  # ✅ Confirmation Factors (Not Required, But Strengthen the Signal)
        ((new_df['adx'] > 15) &   # ✅ Strong Trend Confirmation
         (new_df['ema5'] > new_df['ema20']) |  # ✅ Short-term Uptrend
         (new_df['macd'] > new_df['macd_signal']) |  # ✅ MACD Bullish Crossover
         ((new_df['rsi'] > 40) & (new_df['rsi'].shift(1) < new_df['rsi']))  # ✅ RSI Recovery
        )
    ) 
    &
    (  # ✅ Confirmation Factors (Not Required, But Strengthen the Signal)
        ((new_df['adx'] > 20) |   # ✅ Strong Trend Confirmation
         (new_df['ema5'] > new_df['ema20']) |  # ✅ Short-term Uptrend
         (new_df['macd'] > new_df['macd_signal']) |  # ✅ MACD Bullish Crossover
         ((new_df['rsi'] > 50) & (new_df['rsi'].shift(1) < new_df['rsi']))  # ✅ RSI Recovery
        )
    ),
    1,  # ✅ Buy Signal

    np.where(
        (
            # ✅ Weak Trend / Losing Momentum
            (((new_df['adx'] <= 25) | (new_df['adx'].rolling(3).mean().shift(1) > new_df['adx'])) &
             (new_df['obv'] < new_df['obv'].shift(1))) |

            # ✅ Confirmed Downtrend
            (((new_df['ema5'] < new_df['ema20']) & 
              (new_df['ema5'].shift(1) < new_df['ema20'].shift(1))) &
             (new_df['obv'] < new_df['obv'].shift(1))) |

            # ✅ RSI Overbought & Dropping
            (((new_df['rsi'] > 70) & (new_df['rsi'].rolling(2).mean().shift(1) > new_df['rsi'])) &
             (new_df['obv'] < new_df['obv'].shift(1))) |

            # ✅ MACD Bearish Crossover
            (((new_df['macd'] < new_df['macd_signal']) | 
             ((new_df['macd'].shift(1) - new_df['macd_signal'].shift(1)) > (new_df['macd'] - new_df['macd_signal']))) &
             (new_df['obv'] < new_df['obv'].shift(1))) |

            # ✅ Stochastic Bearish Crossover
            (((new_df['stoch_k'] >= 80) & (new_df['stoch_k'] < new_df['stoch_d'])) &
             (new_df['obv'] < new_df['obv'].shift(1))) |

            # ✅ Price Rejected at Upper BBand
            (((new_df['close'] >= new_df['upper_bband']) &
              (new_df['close'].shift(1) > new_df['upper_bband']) &
              (new_df['close'] < new_df['upper_bband'].shift(1))) &
             (new_df['obv'] < new_df['obv'].shift(1)))
        ),
        0,  # ✅ Sell Signal
        np.nan  # ✅ Ignore uncertain cases
    )
)
 
    # ✅ Remove uncertain rows
    new_df.dropna(inplace=True)

    from dotenv import load_dotenv


    # Load coin symbol from .env
    load_dotenv()
    coin_symbol = os.getenv("SECONDARY_MINT_SYMBOL")

    # ✅ Define file path per coin
    os.makedirs("historical_data", exist_ok=True)
    data_file = f"historical_data/{coin_symbol}_data.csv"

    # ✅ Load existing historical data
    if os.path.exists(data_file):
       old_df = pd.read_csv(data_file)
       old_df['time'] = pd.to_datetime(old_df['time'])
    else:
       old_df = pd.DataFrame()

    # ✅ Merge Old and New Data (Keep Last 15000 Rows)
    df = pd.concat([old_df, new_df]).drop_duplicates(subset=['time']).sort_values('time').tail(20000)

    # ✅ Drop derived columns before saving to CSV
    #columns_to_exclude = [
    #    'bullish_divergence',
    #    'bearish_divergence',
    #    'dynamic_stop_loss',
    #    'atr'
    #]
    #df  = df.drop(columns=[col for col in columns_to_exclude if col in df.columns])
    
    # ✅ Save Updated Data to File 
    # ✅ Save only the clean market data
    df.to_csv(data_file, index=False)
    #log_general.info(f"Historical data updated for {coin_symbol} at {data_file}")
    #print(f"✅ Historical data updated for {coin_symbol} at {data_file}")
   
    log_general.info(f"Historical data updated for {coin_symbol} at {data_file}")

    return df.dropna()  # ✅ Ensure valid data is returned


# ✅ Monitor and retrain model if necessary
import os
import json
import joblib
from dotenv import load_dotenv
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

load_dotenv()
coin_symbol = os.getenv("SECONDARY_MINT_SYMBOL")  # Loaded once globally or inside the function

def find_best_thresholds(model, X_val, y_val):
    import numpy as np
    from sklearn.metrics import f1_score

    probs = model.predict_proba(X_val)  # [:, 0] for sell, [:, 1] for buy
    thresholds = np.arange(0.50, 0.95, 0.01)

    best_buy_f1 = 0
    best_buy_threshold = 0.80

    best_sell_f1 = 0
    best_sell_threshold = 0.80

    for threshold in thresholds:
        # Buy: class 1
        preds_buy = (probs[:, 1] >= threshold).astype(int)
        f1_buy = f1_score(y_val, preds_buy, pos_label=1, zero_division=0)
        if f1_buy > best_buy_f1:
            best_buy_f1 = f1_buy
            best_buy_threshold = threshold

        # Sell: class 0
        preds_sell = (probs[:, 0] >= threshold).astype(int)
        f1_sell = f1_score(y_val, preds_sell, pos_label=0, zero_division=0)
        if f1_sell > best_sell_f1:
            best_sell_f1 = f1_sell
            best_sell_threshold = threshold

    return round(best_buy_threshold, 2), round(best_sell_threshold, 2), best_buy_f1, best_sell_f1



def tune_lightgbm(X, y, symbol, n_trials=50):
    import optuna
    import json
    from lightgbm import LGBMClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    param_save_path = f"models/{symbol.upper()}_params.json"

    # ✅ If parameters already exist, skip tuning
    if os.path.exists(param_save_path):
        with open(param_save_path, "r") as f:
            best_params = json.load(f)
        print(f"📦 Loaded existing best parameters for {symbol.upper()}")
        model = LGBMClassifier(**best_params)
        model.fit(X, y)
        return model, best_params

    # 🚀 Otherwise, perform tuning
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "class_weight": "balanced",
            "random_state": 42,
            "verbosity": -1
        }

        model = LGBMClassifier(**params)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return f1_score(y_val, preds, zero_division=0)

    print(f"🎯 Starting LightGBM tuning for {symbol.upper()}...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print(f"✅ Best parameters for {symbol.upper()}: {best_params}")

    # ✅ Add verbosity to suppress LightGBM warnings
    best_params["verbosity"] = -1
    
    # ✅ Save best parameters for future use
    with open(param_save_path, "w") as f:
        json.dump(best_params, f)

    final_model = LGBMClassifier(**best_params)
    final_model.fit(X, y)

    return final_model, best_params

retrained = False

def monitor_and_retrain(target_accuracy):
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    retrained = False

    model_filename = f"{model_dir}/trained_model_{coin_symbol}.pkl"
    accuracy_filename = f"{model_dir}/best_accuracy_{coin_symbol}.json"

    # Check if model file exists
    if os.path.exists(model_filename):
        best_accuracy = 0
        if os.path.exists(accuracy_filename):
            with open(accuracy_filename, 'r') as f:
                best_accuracy = json.load(f).get('accuracy', 0)

        log_general.info(f"📌 Using pre-trained model for {coin_symbol} (accuracy: {best_accuracy:.4f}) — skipping retraining.")
        model = joblib.load(model_filename)
        model.buy_threshold = getattr(model, "buy_threshold", 0.5)
        model.sell_threshold = getattr(model, "sell_threshold", 0.7)
        return model, best_accuracy, retrained

    # ✅ Model file not found — perform 3 training attempts
    for attempt in range(1, 4):
        log_general.info(f"🚧 Training attempt {attempt} for {coin_symbol}...")
        retrained = True

        df = get_new_data()
        if df is None or df.empty:
            log_general.warning("⚠️ No data for model training.")
            return None, None, retrained

        features = df[EXPECTED_FEATURES]
        target = df['target']

        class_counts = Counter(target)
        log_general.info(f"Class distribution before balancing: {class_counts}")

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target
        )

        counter = Counter(y_train)
        majority = max(counter.values())
        minority = min(counter.values())
        majority_class = max(counter, key=counter.get)
        minority_class = min(counter, key=counter.get)
        desired_ratio = 2
        desired_minority = majority // desired_ratio

        smt = SMOTETomek(sampling_strategy={
            majority_class: majority,
            minority_class: desired_minority
        }, random_state=42)

        X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
        log_general.info(f"✅ Resampled class distribution: {Counter(y_resampled)}")

        model, best_params = tune_lightgbm(X_resampled, y_resampled, coin_symbol, n_trials=50)

        y_proba = model.predict_proba(X_test)[:, 1]
        threshold = 0.3
        y_pred = (y_proba >= threshold).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        report = classification_report(y_test, y_pred, digits=4)

        from datetime import datetime
        log_general.info(f"Model Accuracy for {coin_symbol}: {accuracy:.4f}")
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S}       Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S}       Classification Report:\n{report}")

        buy_thres, sell_thres, buy_f1, sell_f1 = find_best_thresholds(model, X_test, y_test)
        model.buy_threshold = max(0.0, min(1.0, buy_thres))
        model.sell_threshold = max(0.0, min(1.0, sell_thres - 0.02))

        log_general.info(f"🎯 Best buy threshold: {model.buy_threshold:.4f} (F1: {buy_f1:.4f})")
        log_general.info(f"🎯 Best sell threshold: {model.sell_threshold:.4f} (F1: {sell_f1:.4f})")

        # ✅ Save the model only if it's better
        if attempt == 1 or accuracy > best_accuracy:
            best_accuracy = accuracy
            joblib.dump(model, model_filename)
            with open(accuracy_filename, 'w') as f:
                json.dump({'accuracy': accuracy}, f)
            log_general.info(f"✅ Model saved for {coin_symbol} on attempt {attempt}.")

    return model, best_accuracy, retrained



# ✅ Perform trading analysis
def perform_analysis():
    log_general.debug("Performing market analysis and deciding on trade...")
    df = get_new_data()
    
    if df is None:
        log_general.warning("No features retrieved. Skipping analysis.")
        return

    # ✅ Step 1: Check Model Accuracy and Skip Retraining if Above Threshold
    target_accuracy = 0.98
    model, accuracy, retrained = monitor_and_retrain(target_accuracy)

    if accuracy >= target_accuracy:
        log_general.info(f"Model accuracy ({accuracy:.2f}) has met/exceeded the threshold ({target_accuracy}). Skipping retraining.")
    else:
        log_general.info(f"Model accuracy ({accuracy:.2f}) is below the threshold. Retraining the model.")

    # ✅ Step 2: Get Position Data
    position = get_position()
    if position is None:
        log_general.warning("Could not fetch position data. Skipping trade execution.")
        return

    # ✅ Step 3: Predict Market Direction
    X_new = df[EXPECTED_FEATURES].iloc[-1:].values
    prediction_proba = model.predict_proba(X_new)[0]  # Get prediction probabilities
    buy_prob = prediction_proba[1]  # Probability of a buy
    sell_prob = prediction_proba[0]  # Probability of a sell
    prediction = model.predict(X_new)[0]  # Final predicted class

    # ✅ Fetch actual target from indicator-based logic
    actual_target = df['target'].iloc[-1]  # From handcrafted rules

    log_general.info("🔍 Model vs Indicator Logic Comparison:")
    log_general.info(f"Model Prediction: {prediction} | Probabilities -> Buy: {buy_prob:.4f}, Sell: {sell_prob:.4f}")
    log_general.info(f"Indicator-Based Target: {actual_target}")
    log_general.info(f"{'✅ MATCH' if prediction == actual_target else '❌ MISMATCH'} between model and rule-based logic")


    log_general.info(f"Recent targets from indicator logic: {df['target'].tail(5).tolist()}")

    log_general.info(f"Target distribution in data: {df['target'].value_counts().to_dict()}")
 
    global virtual_balance
    trade_price = df['close'].iloc[-1]

    model = joblib.load(f"models/trained_model_{coin_symbol}.pkl")
    #ddlatest_data = fetch_latest_data_as_dataframe()

    # 🔽 Place the dynamic probability threshold logic here:
    buy_threshold = getattr(model, 'buy_threshold', 0.80)
    sell_threshold = getattr(model, 'sell_threshold', 0.80)

    log_general.info(f"🔹 Current Buy/Sell Probability Thresholds: Buy: {buy_threshold:.1f}, Sell: {sell_threshold:.2f}")
    
    #probs = model.predict_proba(latest_data[EXPECTED_FEATURES])[0]
    #prob_sell = probs[0]
    #prob_buy = probs[1]

    
    # ✅ Log the probabilities
    log_general.info(f"🔹 Current Probabilities -> Buy: {buy_prob:.4f}, Sell: {sell_prob:.4f}")
    #print(f"🔹 Current Probabilities -> Buy: {buy_prob:.4f}, Sell: {sell_prob:.4f}")

    if retrained:
        log_general.info("⚠️ Skipping trade this cycle — model just retrained.")
        return
     
    # ✅ Step 4: Buy if Not in a Position and Buy Probability is High
        # ✅ Step 4: Buy if Not in a Position and Buy Probability is High
        # ✅ Step 4: Buy if Not in a Position and Buy Probability is High
    input_amount = find_balance(config().primary_mint)
    if not position['is_open'] and buy_prob >= buy_threshold:

        if prediction == 1 and actual_target == 1:
            log_general.info("✅ Model and rule agree on BUY. Executing trade...")

            was_bought = run_buy_script(config().secondary_mint)
            if not was_bought:
                log_general.warning("❌ Buy failed. Skipping position update.")
                return

            trade_amount = round(input_amount / trade_price, 6) if trade_price > 0 else 0.0
            trailing_stop_price = trade_price * 0.985  # Initial trailing stop 1.5% below entry

            position.update({
                'is_open': True,
                'entry_price': trade_price,
                'trade_amount': trade_amount,
                'trailing_stop': trailing_stop_price,
                'current_pnl': 0.0
            })
            save_position(position)

            trades.append(f"Bought {trade_amount} at {trade_price}")

            trade_entry = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "action": "BUY",
                "price": trade_price,
                "amount_spent": input_amount,
                "amount_bought": trade_amount,
                "profit_loss": 0.00
            }
            save_trade_log(trade_entry)

            log_transaction.debug(f"Trade executed: Bought {trade_amount} at {trade_price}")
            log_general.info(f"✅ Buy Executed: {trade_amount} at {trade_price}")
            print(f"✅ Buy Executed: {trade_amount} at {trade_price}")

        else:
            log_general.warning("❌ Mismatch between model and rule logic — skipping BUY and retraining model.")
            monitor_and_retrain(target_accuracy=2.0)
            return

    # ✅ Step 5: Track PnL and Trailing Stop While Holding
   
        # ✅ Step 5: Track PnL and Trailing Stop While Holding
    elif position['is_open']:
        if latest_market_price is not None:
            entry_price = position.get("entry_price", 0)
            trade_amount = position.get("trade_amount", 0)
            current_pnl = (latest_market_price - entry_price) * trade_amount
            position['current_pnl'] = current_pnl

            # ✅ Track highest price since entry
            position['highest_price'] = max(
                position.get('highest_price', entry_price),
                latest_market_price
            )

            # ✅ Trailing stop based on highest price (not current price)
            new_trailing_stop = position['highest_price'] * 0.94  # 1.5% below peak

            # ✅ Only update if trailing stop moves up
            if new_trailing_stop > position['trailing_stop']:
                log_general.info(f"🔁 Trailing stop updated: {position['trailing_stop']:.4f} → {new_trailing_stop:.4f}")
                position['trailing_stop'] = new_trailing_stop

            # ✅ Check if price falls below trailing stop
            if latest_market_price <= position['trailing_stop']:
                log_general.info(f"🚨 Price dropped below trailing stop ({latest_market_price:.4f} ≤ {position['trailing_stop']:.4f}) — selling...")
                prediction = 0
                sell_prob = 1.0  # Force sell

            save_position(position)

            log_general.info(f"📈 Holding Position - Unrealized PnL: {current_pnl:.2f}")
            print(f"📈 Holding Position - Unrealized PnL: {current_pnl:.2f}")
        else:
            print("⚠️ Market price not available yet. Cannot calculate PnL.")

    # ✅ Step 6: Sell If Already in a Position and Sell Signal or Trailing Stop Triggered
    if position['is_open'] and (prediction == 0 and sell_prob >= sell_threshold):

        was_sold = run_sell_script(config().secondary_mint)
        if not was_sold:
            log_general.warning("❌ Sell failed. Holding position.")
            return

        input_amount = find_balance(config().secondary_mint)
        trade_amount = position.get("trade_amount", 0)
        sell_price = latest_market_price
        entry_price = position.get("entry_price", 0)
        profit_loss = (sell_price - entry_price) * trade_amount
        virtual_balance += profit_loss

        position.update({
            'is_open': False,
            'entry_price': None,
            'trade_amount': None,
            'trailing_stop': None,
            'current_pnl': None
        })
        save_position(position)

        trades.append(f"Sold at {sell_price}, Profit/Loss: {profit_loss:.2f}")

        trade_entry = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "action": "SELL",
            "price": sell_price,
            "amount_sold": input_amount,
            "amount_traded": trade_amount,
            "profit_loss": profit_loss,
            "final_balance_after_trade": virtual_balance
        }
        save_trade_log(trade_entry)

        log_transaction.debug(f"Trade executed: Sold at {sell_price}, Profit/Loss: {profit_loss:.2f}")
        log_general.info(f"✅ Sell Executed: Sold at {sell_price}, Profit/Loss: {profit_loss:.2f}")
        print(f"✅ Sell Executed: Sold at {sell_price}, Profit/Loss: {profit_loss:.2f}")

    else:
        log_general.info("⚠️  No trade executed this cycle. Waiting for new signals.")
        #print("No trade executed this cycle. Waiting for new signals.")
  

# **Start trading bot**
def start_trading():
    log_general.debug("tradambot has now initialized the trading algorithm.")
    
    executors = {'default': ThreadPoolExecutor(max_workers=3)}
    trading_sched = BackgroundScheduler(executors=executors)

    trading_sched.add_job(perform_analysis, 'interval', seconds=config().price_update_seconds, max_instances=1)
    trading_sched.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        log_general.debug("Stopping tradambot...")
        trading_sched.shutdown()
