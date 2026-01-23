import requests
import csv
import time
from datetime import datetime
import os

# ========= CONFIG =========
INTERVALO = 300  # 5 minutos
MONEDAS = ["USDT", "USDC", "DAI"]
FIAT = "ARS"
ROWS = 50
CSV_FILE = "p2p_raw.csv"

URL = "https://p2p.binance.com/bapi/c2c/v2/friendly/c2c/adv/search"

HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0"
}

# ========= INIT =========
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "moneda",
            "lado",
            "precio",
            "metodos_pago",
            "min_ars",
            "max_ars",
            "disponible"
        ])

print("🚀 Recolector RAW iniciado")

# ========= FUNC =========
def recolectar():
    filas = 0
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for moneda in MONEDAS:
            for lado in ["BUY", "SELL"]:
                
                payload = {
                    "page": 1,
                    "rows": 20,  # 👈 importante
                    "asset": moneda,
                    "fiat": FIAT,
                    "tradeType": lado,

                    "countries": [],
                    "payTypes": [],

                    "publisherType": None,
                    "merchantCheck": False,
                    "proMerchantAds": False,
                    "shieldMerchantAds": False
                }


                try:
                    r = requests.post(URL, headers=HEADERS, json=payload, timeout=10)
                    resp = r.json()

                    # 🔒 BLINDAJE TOTAL
                    data = resp.get("data") or []

                    print(f"{moneda} {lado} items: {len(data)}")

                    if not data:
                        print(f"⚠️ Sin datos {moneda} {lado}")
                        continue

                    for ad in data:
                        adv = ad.get("adv")
                        if not adv:
                            continue

                        precio = float(adv.get("price", 0))
                        min_ars = float(adv.get("minSingleTransAmount", 0))
                        max_ars = float(adv.get("maxSingleTransAmount", 0))
                        disponible = float(adv.get("surplusAmount", 0))

                        metodos = adv.get("tradeMethods") or []
                        metodos_pago = "|".join(
                            m.get("identifier", "UNK") for m in metodos
                        )

                        writer.writerow([
                            ts,
                            moneda,
                            lado,
                            precio,
                            metodos_pago,
                            min_ars,
                            max_ars,
                            disponible
                        ])

                        filas += 1

                except Exception as e:
                    print(f"⚠️ Error {moneda} {lado}: {e}")

    print(f"[{ts}] 📥 Guardadas {filas} filas")


# ========= LOOP =========
if __name__ == "__main__":
    while True:
        recolectar()
        time.sleep(INTERVALO)

