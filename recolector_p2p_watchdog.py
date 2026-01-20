import requests
import csv
import time
from datetime import datetime
from statistics import mean
import os
import threading

# ===== CONFIGURACIÓN =====
INTERVALO = 300  # 5 minutos
MONEDAS = ["USDT", "USDC", "DAI"]
FIAT = "ARS"
FILAS = 10

# Rutas adaptadas para Linux / EC2
CSV_FILE = "./p2p_historial.csv"
LOG_FILE = "./bot.log"

URL = "https://p2p.binance.com/bapi/c2c/v2/friendly/c2c/adv/search"
HEADERS = {"Content-Type": "application/json"}

# ===== FUNCIONES =====
def log(text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {text}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def obtener_precios(asset, trade_type):
    try:
        payload = {
            "page": 1,
            "rows": FILAS,
            "payTypes": [],
            "asset": asset,
            "fiat": FIAT,
            "tradeType": trade_type
        }
        r = requests.post(URL, headers=HEADERS, json=payload, timeout=10).json()
        precios = [float(ad["adv"]["price"]) for ad in r["data"]]
        return precios
    except Exception as e:
        log(f"⚠️ Error al obtener precios {asset} {trade_type}: {e}")
        return []

def recolectar():
    now = datetime.now()
    fecha = now.strftime("%Y-%m-%d")
    hora = now.strftime("%H:%M")

    for moneda in MONEDAS:
        buy_prices = obtener_precios(moneda, "BUY")
        sell_prices = obtener_precios(moneda, "SELL")

        if len(buy_prices) < 3 or len(sell_prices) < 3:
            log(f"⚠️ Pocas ofertas para {moneda}, saltando")
            continue

        buy_avg = mean(buy_prices)
        sell_avg = mean(sell_prices)
        spread_pct = (sell_avg - buy_avg) / buy_avg * 100

        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([fecha, hora, moneda, round(buy_avg, 2), round(sell_avg, 2),
                             round(spread_pct, 3), min(len(buy_prices), len(sell_prices))])

        log(f"{moneda} | Buy {buy_avg:.2f} | Sell {sell_avg:.2f} | Spread {spread_pct:.2f}%")

# ===== CREAR CSV Y LOG SI NO EXISTEN =====
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fecha", "hora", "moneda", "buy_avg", "sell_avg", "spread_pct", "muestras"])

if not os.path.exists(LOG_FILE):
    open(LOG_FILE, "w").close()

log("📊 Recolector P2P con watchdog iniciado")

# ===== WATCHDOG =====
def main_loop():
    while True:
        try:
            recolectar()
        except Exception as e:
            log(f"⚠️ Error en ciclo principal: {e}")
        time.sleep(INTERVALO)

def watchdog():
    while True:
        time.sleep(INTERVALO)
        if not threading.main_thread().is_alive():
            log("⚠️ Thread principal caído, reiniciando...")
            threading.Thread(target=main_loop, daemon=True).start()

# ===== EJECUCIÓN =====
if __name__ == "__main__":
    threading.Thread(target=main_loop, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()
    while True:
        time.sleep(60)

