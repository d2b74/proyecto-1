import os

DATA_ROOT = os.getenv('DATA_ROOT', '/content/drive/MyDrive/p2p_datasets/')
ASSET = 'USDT'
CAPITAL_USD = 20000
MIN_RATE_CONFIANZA = 0.95
RAW_PATH = os.getenv('RAW_PATH', os.path.join(DATA_ROOT, 'raw'))
UMBRALES = {
    'itc_peligro': 65,
    'itc_oportunidad': 35,
    'confianza_minima': 50,
    'fuerza_baja': 0.4,
    'var_umbral': 0.5,
    'spread_min_scalper': 0.15,
    'trend_min_swing': 0.1,
    'posicion_max_estrategica': 20,
    'itc_threshold_scalper': 65,
    'itc_threshold_swing': 40,
    'itc_threshold_estrategica': 30,
    'factor_scalper': 1.0,
    'factor_swing': 0.5,
    'factor_estrategica': 0.8,
    'target_scalper_pct': 0.2,
    'target_swing_pct': 0.4,
    'target_estrategica_pct': 0.8,
    'riesgo_minimo_factor': 0.2,
    'btc_vol_threshold': 0.005,
}

CONFIG = {
    'DATA_ROOT': DATA_ROOT,
    'asset': ASSET,
    'capital_usd': CAPITAL_USD,
    'min_rate_confianza': MIN_RATE_CONFIANZA,
    'umbrales': UMBRALES,
    'path_datasets': RAW_PATH,
}
