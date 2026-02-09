import requests
import json
import time
import csv
import os
from datetime import datetime, timedelta

class ScraperDataCrudaML:
    def __init__(self):
        self.config = {
            "monedas": ["USDT", "USDC"],
            "max_anuncios": 20,
            "intervalo": 300, 
            "archivo_base": "p2p_raw_dataset_"
        }
        self.cache_mep = (0.0, 0.0)
        self.cache_blue = (0.0, 0.0)
        
        self.session = requests.Session()
        self.headers = {
            "Accept": "*/*",
            "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Origin": "https://p2p.binance.com",
            "Referer": "https://p2p.binance.com/es/trade/all-payments/USDT?fiat=ARS",
            "Cache-Control": "no-cache"
        }
        
    def obtener_hora_argentina(self):
        return datetime.utcnow() - timedelta(hours=3)

    def obtener_btc_global(self):
        try:
            res = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=10)
            if res.status_code == 200:
                return float(res.json()['price'])
        except: pass
        return 0.0

    def obtener_datos_fiat_reales(self):
        url = "https://criptoya.com/api/dolar"
        try:
            res = self.session.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
            if res.status_code == 200:
                data = res.json()
                blue_v = float(data.get('blue', {}).get('ask', 0))
                blue_c = float(data.get('blue', {}).get('bid', 0))
                mep_v = float(data.get('mep', {}).get('al30', {}).get('24hs', {}).get('price', 0))
                mep_c = round(mep_v * 0.992, 2) if mep_v > 0 else 0
                
                if mep_v > 0: self.cache_mep = (mep_c, mep_v)
                if blue_v > 0: self.cache_blue = (blue_c, blue_v)
                
                return self.cache_mep, self.cache_blue
        except: pass
        return self.cache_mep, self.cache_blue

    def extraer_datos_anuncio(self, anuncio, mep_c, mep_v, blue_c, blue_v, btc_p):
        adv = anuncio.get("adv", {})
        advertiser = anuncio.get("advertiser", {})
        metodos = adv.get("tradeMethods", [])
        ahora = self.obtener_hora_argentina()
        
        return {
            "timestamp": ahora.strftime("%Y-%m-%d %H:%M:%S"),
            "mes_archivo": ahora.strftime("%Y-%m"),
            "moneda": adv.get("asset", ""),
            "lado": adv.get("tradeType", ""),
            "precio": float(adv.get("price", 0)),
            "mep_compra": mep_c,
            "mep_venta": mep_v,
            "blue_compra": blue_c,
            "blue_venta": blue_v,
            "btc_usdt_global": btc_p,
            "user_no": advertiser.get("userNo", ""),
            "nick_name": advertiser.get("nickName", ""),
            "month_order_count": advertiser.get("monthOrderCount", 0),
            "month_finish_rate": float(advertiser.get("monthFinishRate", 0)),
            "positive_rate": float(advertiser.get("positiveRate", 0)),
            "user_grade": advertiser.get("userGrade", 0),
            "user_type": advertiser.get("userType", ""),
            "pro_merchant": 1 if advertiser.get("proMerchant") else 0,
            "vip_level": advertiser.get("vipLevel", 0),
            "active_time_seconds": advertiser.get("activeTimeInSecond", 0),
            "user_rating_score": advertiser.get("userRatingScore", 0),
            "user_rating_count": advertiser.get("userRatingCount", 0),
            "disponible": float(adv.get("surplusAmount", 0)),
            "min_single_amount": float(adv.get("minSingleTransAmount", 0)),
            "max_single_amount": float(adv.get("maxSingleTransAmount", 0)),
            "dynamic_max_amount": float(adv.get("dynamicMaxSingleTransAmount", 0)),
            "tradable_quantity": float(adv.get("tradableQuantity", 0)),
            "commission_rate": float(adv.get("commissionRate", 0)),
            "req_trade_count_min": adv.get("userAllTradeCountMin", 0),
            "req_finish_rate_min": adv.get("userTradeCompleteRateMin", 0),
            "req_reg_days_limit": adv.get("buyerRegDaysLimit", 0),
            "adv_no": f"ID_{adv.get('advNo', '')}",
            "classify": adv.get("classify", ""),
            "price_type": adv.get("priceType", ""),
            "pay_time_limit": adv.get("payTimeLimit", 0),
            "metodos_pago": "|".join([str(m.get("tradeMethodName") or "") for m in metodos]),
            "remarks": str(adv.get("remarks", "")).replace("\n", " ").replace(",", " "),
            "raw_json": json.dumps(anuncio)
        }

    def obtener_anuncios(self, moneda, trade_type):
        url = "https://p2p.binance.com/bapi/c2c/v2/friendly/c2c/adv/search"
        payload = {
            "proMerchantAds": False, "page": 1, "rows": self.config["max_anuncios"],
            "payTypes": [], "countries": [], "publisherType": None,
            "asset": moneda, "fiat": "ARS", "tradeType": trade_type, "shieldCheck": False
        }
        try:
            res = self.session.post(url, json=payload, headers=self.headers, timeout=15)
            if res.status_code == 200:
                return res.json().get("data", [])
        except: return []

    def ejecutar_ciclo(self):
        print(f"🔄 Iniciando captura: {datetime.now().strftime('%H:%M:%S')}")
        m_data, b_data = self.obtener_datos_fiat_reales()
        self.cache_mep, self.cache_blue = m_data, b_data
        
        todos_datos = []
        btc_p = self.obtener_btc_global()
        m_c, m_v = self.cache_mep
        b_c, b_v = self.cache_blue
        
        for moneda in self.config["monedas"]:
            for lado in ["BUY", "SELL"]:
                anuncios = self.obtener_anuncios(moneda, lado)
                if anuncios:
                    for a in anuncios:
                        todos_datos.append(self.extraer_datos_anuncio(a, m_c, m_v, b_c, b_v, btc_p))
                time.sleep(1) 
                
        if todos_datos:
            mes_actual = todos_datos[0]['mes_archivo']
            nombre_archivo = f"{self.config['archivo_base']}{mes_actual}.csv"
            
            # Guardamos localmente (sobrescribimos el temporal de esta sesión)
            with open(nombre_archivo, "w", encoding="utf-8", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=todos_datos[0].keys())
                # SÍ escribimos el encabezado para que el YAML sepa qué columnas hay
                writer.writeheader()
                writer.writerows(todos_datos)
            
            print(f"✅ Éxito local: {len(todos_datos)} filas preparadas en {nombre_archivo}")
        else:
            print("⚠️ Ciclo sin datos.")

if __name__ == "__main__":
    ScraperDataCrudaML().ejecutar_ciclo()
