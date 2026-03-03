import requests
import pandas as pd
import os
import re
import time
from datetime import datetime
import pytz
from config import CONFIG

class ContextoExterior:
    def __init__(self):
        self.data_root = CONFIG['DATA_ROOT']
        self.contexto_dir = os.path.join(self.data_root, 'contexto')
        self.contexto_file = os.path.join(self.contexto_dir, 'contexto.csv')
        os.makedirs(self.contexto_dir, exist_ok=True)
        self.tz = pytz.timezone('America/Argentina/Buenos_Aires')
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})

    def obtener_dxy_vix_sp500(self):
        try:
            import yfinance as yf
            tickers = {
                'dxy': 'DX-Y.NYB',
                'vix': '^VIX',
                'sp500': '^GSPC'
            }
            result = {}
            for key, symbol in tickers.items():
                data = yf.download(symbol, period='1d', interval='1h', progress=False, auto_adjust=False)
                if not data.empty:
                    result[key] = data['Close'].iloc[-1].item()
                else:
                    result[key] = None
                time.sleep(0.5)
            return result
        except Exception as e:
            print(f"Error en Yahoo Finance: {e}")
            return {'dxy': None, 'vix': None, 'sp500': None}

    def obtener_btc_eth_data(self):
        try:
            url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true"
            res = self.session.get(url, timeout=15)
            if res.status_code == 200:
                data = res.json()
                btc = data.get('bitcoin', {})
                eth = data.get('ethereum', {})
                return {
                    'btc_price': btc.get('usd'),
                    'btc_change_24h': btc.get('usd_24h_change'),
                    'btc_volume': btc.get('usd_24h_vol'),
                    'eth_price': eth.get('usd'),
                    'eth_change_24h': eth.get('usd_24h_change'),
                    'eth_volume': eth.get('usd_24h_vol')
                }
            else:
                print(f"CoinGecko respondió {res.status_code}")
        except Exception as e:
            print(f"Error en CoinGecko: {e}")
        return {
            'btc_price': None,
            'btc_change_24h': None,
            'btc_volume': None,
            'eth_price': None,
            'eth_change_24h': None,
            'eth_volume': None
        }

    def obtener_fear_greed(self):
        try:
            url = "https://api.alternative.me/fng/?limit=1"
            res = self.session.get(url, timeout=10)
            if res.status_code == 200:
                return int(res.json()['data'][0]['value'])
        except Exception as e:
            print(f"Error obteniendo Fear & Greed: {e}")
        return None

    def obtener_riesgo_pais(self):
        """
        Intenta obtener el riesgo país usando 3 métodos distintos.
        Diseñado para funcionar en GitHub Actions.
        """
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'}
        
        # MÉTODO 1: API Directa de Ámbito (Suele funcionar mejor en servidores)
        try:
            res = requests.get("https://mercados.ambito.com/riesgopais/ultimo", headers=headers, timeout=10)
            if res.status_code == 200:
                val = res.json().get('valor', '0').replace('.', '').split(',')[0]
                return int(val)
        except:
            pass

        # MÉTODO 2: Scraping de Texto en DolarHoy
        try:
            res = requests.get("https://dolarhoy.com/cotizacion-riesgo-pais", headers=headers, timeout=10)
            import re
            match = re.search(r'class="value">([\d\.]+)<', res.text)
            if match:
                return int(match.group(1).replace('.', ''))
        except:
            pass

        # MÉTODO 3: Fallback a un valor por defecto o None
        print("⚠️ No se pudo obtener Riesgo País en este ciclo.")
        return None
    def ejecutar(self):
        timestamp_arg = datetime.now(self.tz).strftime("%Y-%m-%d %H:%M:%S")
        datos = {'timestamp': timestamp_arg}

        print("Obteniendo datos de Yahoo Finance...")
        yahoo = self.obtener_dxy_vix_sp500()
        datos.update(yahoo)

        print("Obteniendo datos de BTC y ETH desde CoinGecko...")
        cripto = self.obtener_btc_eth_data()
        datos.update(cripto)

        print("Obteniendo Fear & Greed...")
        datos['fear_greed'] = self.obtener_fear_greed()

        print("Obteniendo riesgo país...")
        datos['riesgo_pais'] = self.obtener_riesgo_pais()

        # Crear DataFrame y guardar
        df_new = pd.DataFrame([datos])
        if os.path.exists(self.contexto_file):
            df_existing = pd.read_csv(self.contexto_file, parse_dates=['timestamp'])
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_final = df_new

        df_final.to_csv(self.contexto_file, index=False)
        print(f"✅ Contexto guardado: {timestamp_arg}")
        print(f"   DXY: {datos['dxy']}, VIX: {datos['vix']}, SP500: {datos['sp500']}")
        print(f"   BTC: {datos.get('btc_price')} ({datos.get('btc_change_24h')}%), ETH: {datos.get('eth_price')} ({datos.get('eth_change_24h')}%)")
        print(f"   Fear & Greed: {datos['fear_greed']}, Riesgo país: {datos['riesgo_pais']}")

if __name__ == "__main__":
    if 'DATA_ROOT' not in os.environ:
        os.environ['DATA_ROOT'] = './ml_engine'
    ContextoExterior().ejecutar()
