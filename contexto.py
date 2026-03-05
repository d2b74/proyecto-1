
import requests
import pandas as pd
import os
import time
from datetime import datetime
import pytz
from config import CONFIG


class ContextoExterior:

    def __init__(self):
        self.data_root    = CONFIG['DATA_ROOT']
        self.contexto_dir = os.path.join(self.data_root, 'contexto')
        self.contexto_file = os.path.join(self.contexto_dir, 'contexto.csv')
        os.makedirs(self.contexto_dir, exist_ok=True)
        self.tz = pytz.timezone('America/Argentina/Buenos_Aires')
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})

    def _ultimo_valido(self, columna):
        try:
            if os.path.exists(self.contexto_file):
                df = pd.read_csv(self.contexto_file)
                if columna in df.columns:
                    serie = df[columna].dropna()
                    if not serie.empty:
                        return serie.iloc[-1]
        except Exception:
            pass
        return None

    def _safe_float(self, val, columna=None):
        """
        Convierte val a float.
        Si falla (string, None, NaN), intenta recuperar el último valor válido del CSV.
        """
        try:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                raise ValueError
            return float(val)
        except (ValueError, TypeError):
            if columna:
                cached = self._ultimo_valido(columna)
                if cached is not None:
                    try:
                        return float(cached)
                    except (ValueError, TypeError):
                        pass
            return None

    def _safe_int(self, val, columna=None):
        """Igual que _safe_float pero para enteros."""
        result = self._safe_float(val, columna)
        return int(result) if result is not None else None

    def obtener_dxy_vix_sp500(self):
        try:
            import yfinance as yf
            tickers = {'dxy': 'DX-Y.NYB', 'vix': '^VIX', 'sp500': '^GSPC'}
            result = {}
            for key, symbol in tickers.items():
                data = yf.download(symbol, period='1d', interval='1h',
                                   progress=False, auto_adjust=False)
                if not data.empty:
                    result[key] = self._safe_float(data['Close'].iloc[-1].item(), key)
                else:
                    result[key] = self._safe_float(None, key)
                time.sleep(0.5)
            return result
        except Exception as e:
            print(f"Error en Yahoo Finance: {e}")
            return {
                'dxy':   self._safe_float(None, 'dxy'),
                'vix':   self._safe_float(None, 'vix'),
                'sp500': self._safe_float(None, 'sp500'),
            }

    def obtener_btc_eth_data(self):
        campos = {
            'btc_price':      ('bitcoin', 'usd'),
            'btc_change_24h': ('bitcoin', 'usd_24h_change'),
            'btc_volume':     ('bitcoin', 'usd_24h_vol'),
            'eth_price':      ('ethereum', 'usd'),
            'eth_change_24h': ('ethereum', 'usd_24h_change'),
            'eth_volume':     ('ethereum', 'usd_24h_vol'),
        }
        try:
            url = ("https://api.coingecko.com/api/v3/simple/price"
                   "?ids=bitcoin,ethereum&vs_currencies=usd"
                   "&include_24hr_change=true&include_24hr_vol=true")
            res = self.session.get(url, timeout=15)
            if res.status_code == 200:
                data = res.json()
                return {
                    col: self._safe_float(data.get(asset, {}).get(field), col)
                    for col, (asset, field) in campos.items()
                }
            print(f"CoinGecko respondió {res.status_code}")
        except Exception as e:
            print(f"Error en CoinGecko: {e}")
        # Todos fallaron: intentar cache
        return {col: self._safe_float(None, col) for col in campos}

    def obtener_fear_greed(self):
        try:
            url = "https://api.alternative.me/fng/?limit=1"
            res = self.session.get(url, timeout=10)
            if res.status_code == 200:
                val = res.json()['data'][0]['value']
                return self._safe_int(val, 'fear_greed')
        except Exception as e:
            print(f"Error Fear & Greed: {e}")
        return self._safe_int(None, 'fear_greed')

    def obtener_riesgo_pais(self):

        headers = {'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                   'AppleWebKit/537.36')}

        # MÉTODO 1: ArgentinaDatos
        try:
            res = requests.get(
                "https://api.argentinadatos.com/v1/finanzas/indices/riesgo-pais/ultimo",
                headers=headers, timeout=10
            )
            if res.status_code == 200:
                valor = self._safe_int(res.json().get('valor'), 'riesgo_pais')
                if valor and valor > 0:
                    print(f"✅ Riesgo País (ArgentinaDatos): {valor}")
                    return valor
        except Exception as e:
            print(f"ArgentinaDatos falló: {e}")

        # MÉTODO 2: Ámbito
        try:
            res = requests.get("https://mercados.ambito.com/riesgopais/ultimo",
                               headers=headers, timeout=10)
            if res.status_code == 200:
                raw  = res.json().get('valor', '0').replace('.', '').split(',')[0]
                valor = self._safe_int(raw, 'riesgo_pais')
                if valor and valor > 0:
                    print(f"✅ Riesgo País (Ámbito): {valor}")
                    return valor
        except Exception as e:
            print(f"Ámbito falló: {e}")

        # MÉTODO 3: Cache
        cached = self._safe_int(None, 'riesgo_pais')
        if cached:
            print(f"⚠️ Riesgo País desde cache: {cached}")
            return cached

        print("⚠️ No se pudo obtener Riesgo País.")
        return None

    def ejecutar(self):
        timestamp_arg = datetime.now(self.tz).strftime("%Y-%m-%d %H:%M:%S")
        datos = {'timestamp': timestamp_arg}

        print("Obteniendo datos de Yahoo Finance...")
        datos.update(self.obtener_dxy_vix_sp500())

        print("Obteniendo BTC / ETH desde CoinGecko...")
        datos.update(self.obtener_btc_eth_data())

        print("Obteniendo Fear & Greed...")
        datos['fear_greed'] = self.obtener_fear_greed()

        print("Obteniendo Riesgo País...")
        datos['riesgo_pais'] = self.obtener_riesgo_pais()

        campos_float = ['dxy', 'vix', 'sp500', 'btc_price', 'btc_change_24h', 'btc_volume',
                        'eth_price', 'eth_change_24h', 'eth_volume']
        for campo in campos_float:
            if campo in datos and datos[campo] is not None:
                try:
                    datos[campo] = float(datos[campo])
                except (ValueError, TypeError):
                    datos[campo] = self._safe_float(None, campo)

        campos_int = ['fear_greed', 'riesgo_pais']
        for campo in campos_int:
            if campo in datos and datos[campo] is not None:
                try:
                    datos[campo] = int(datos[campo])
                except (ValueError, TypeError):
                    datos[campo] = self._safe_int(None, campo)

        df_new = pd.DataFrame([datos])

        if os.path.exists(self.contexto_file):
            df_existing = pd.read_csv(self.contexto_file, parse_dates=['timestamp'])
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_final = df_new

        # Escritura atómica
        tmp_path = self.contexto_file + '.tmp'
        df_final.to_csv(tmp_path, index=False)
        os.replace(tmp_path, self.contexto_file)

        print(f"✅ Contexto guardado: {timestamp_arg}")
        btc_change = datos.get('btc_change_24h')
        eth_change = datos.get('eth_change_24h')
        print(f"   DXY: {datos.get('dxy')} | VIX: {datos.get('vix')} | SP500: {datos.get('sp500')}")
        print(f"   BTC: {datos.get('btc_price')} "
              f"({'N/A' if btc_change is None else f'{btc_change:.2f}%'}) | "
              f"ETH: {datos.get('eth_price')} "
              f"({'N/A' if eth_change is None else f'{eth_change:.2f}%'})")
        print(f"   Fear & Greed: {datos.get('fear_greed')} | Riesgo País: {datos.get('riesgo_pais')}")


if __name__ == "__main__":
    if 'DATA_ROOT' not in os.environ:
        os.environ['DATA_ROOT'] = './ml_engine'
    ContextoExterior().ejecutar()
