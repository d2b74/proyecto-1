import pandas as pd
import numpy as np
import os
import holidays
from utils import obtener_ventana_fresca
from config import CONFIG

class MacroAnalizadorSystem:


    def __init__(self, cfg):
        self.cfg = cfg
        self.asset = cfg.get('asset', 'USDT').upper()
        self.data_root = cfg['DATA_ROOT']  # o cfg.get('DATA_ROOT', '/content/drive/MyDrive/data_pool')
        self.base_path = os.path.join(self.data_root, self.asset)
        self.path_macro_hist = os.path.join(self.base_path, 'macro_historico.csv')
        self.feriados_arg = holidays.Argentina()

        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def _get_contexto_temporal(self, ts):
        return {
            'hora': ts.hour,
            'dia_semana': ts.weekday(),
            'es_finde': ts.weekday() >= 5,
            'es_feriado': ts in self.feriados_arg,
            'es_horario_bancario': (ts.weekday() < 5) and (10 <= ts.hour <= 17)
        }

    def _get_btc_volatility(self, df, window_min=15):
        """Calcula la volatilidad de BTC en una ventana aproximada de window_min minutos."""
        df_btc = df[df['btc_usdt_global'] > 0].copy()
        if len(df_btc) < 2:
            return 0.0

        df_btc['ret_btc'] = np.log(df_btc['btc_usdt_global'] / df_btc['btc_usdt_global'].shift(1))
        df_btc['ret_btc'] = df_btc['ret_btc'].replace([np.inf, -np.inf], np.nan)

        # Estimar número de filas en la ventana
        time_span_min = (df_btc['timestamp'].iloc[-1] - df_btc['timestamp'].iloc[0]).total_seconds() / 60.0
        if time_span_min < 1:
            return 0.0
        approx_rows = max(1, int(window_min * len(df_btc) / time_span_min))
        rets = df_btc['ret_btc'].dropna().iloc[-approx_rows:]
        return rets.std() if len(rets) > 1 else 0.0

    def _get_brecha_velocity(self, df):
        """Calcula la velocidad de cambio de la brecha MEP en % por hora."""
        mask = (df['mep_venta'] > 0) & (df['precio'] > 0)
        df_temp = df[mask].copy()
        if len(df_temp) < 5:
            return 0.0

        df_temp['brecha'] = (df_temp['precio'] / df_temp['mep_venta'] - 1) * 100
        x = np.arange(len(df_temp))
        y = df_temp['brecha'].values
        if np.std(x) == 0:
            return 0.0
        b = np.cov(x, y)[0, 1] / np.var(x)  # pendiente por índice

        # Convertir a cambio por hora
        time_span_hours = (df_temp['timestamp'].iloc[-1] - df_temp['timestamp'].iloc[0]).total_seconds() / 3600.0
        if time_span_hours < 0.1:
            return 0.0
        indices_por_hora = len(df_temp) / time_span_hours
        velocity = b * indices_por_hora
        return velocity

    def ejecutar(self, timestamp_referencia=None):
        # Obtener ventana de 24h
        df = obtener_ventana_fresca(self.data_root, self.asset, horas_atras=24)
        if df is None or df.empty:
            print("⚠️ No hay datos para análisis macro.")
            return None

        if timestamp_referencia is None:
            ts_ref = pd.to_datetime(df['timestamp'].max())
        else:
            ts_ref = pd.to_datetime(timestamp_referencia)

        df = df[df['timestamp'] <= ts_ref].copy()

        # Tomar solo lado SELL (precio de venta como referencia)
        df_v = df[df['lado'] == 'SELL'].copy().sort_values('timestamp')
        if len(df_v) < 10:
            print("⚠️ Muestras insuficientes para análisis macro (<10).")
            return None

        # Precios y rango
        p_act = df_v['precio'].iloc[-1]
        p_ini = df_v['precio'].iloc[0]
        var_24h = ((p_act / p_ini) - 1) * 100
        min_24h = df_v['precio'].min()
        max_24h = df_v['precio'].max()
        if max_24h != min_24h:
            posicion_rel = ((p_act - min_24h) / (max_24h - min_24h)) * 100
        else:
            posicion_rel = 50.0
        posicion_rel = max(0, min(100, posicion_rel))

        # Volatilidad
        df_v = df_v[df_v['precio'] > 0].copy()
        df_v['ret'] = np.log(df_v['precio'] / df_v['precio'].shift(1))
        df_v['ret'] = df_v['ret'].replace([np.inf, -np.inf], np.nan)
        volatilidad = df_v['ret'].dropna().std()
        if np.isnan(volatilidad):
            volatilidad = 0.0

        # Volumen proxy
        volumen_proxy = df_v['max_single_amount'].sum()

        # Brechas MEP y Blue
        df_reciente = df_v.tail(20)
        mep_avg = df_reciente['mep_venta'].mean()
        blue_avg = df_reciente['blue_venta'].mean()

        mep_avg_safe = mep_avg if (mep_avg > 0 and not np.isnan(mep_avg)) else p_act
        blue_avg_safe = blue_avg if (blue_avg > 0 and not np.isnan(blue_avg)) else p_act

        brecha_mep = ((p_act / mep_avg_safe) - 1) * 100
        brecha_blue = ((p_act / blue_avg_safe) - 1) * 100

        # Media de brecha en 24h
        mask_mep = (df_v['mep_venta'] > 0) & (df_v['precio'] > 0)
        brechas_hist = ((df_v.loc[mask_mep, 'precio'] / df_v.loc[mask_mep, 'mep_venta']) - 1) * 100
        mep_gap_avg_24h = brechas_hist.mean() if not brechas_hist.empty else brecha_mep

        # Correlación con BTC
        corr_btc = 0.0
        if 'btc_usdt_global' in df_v.columns:
            df_btc = df_v[df_v['btc_usdt_global'] > 0].copy()
            if len(df_btc) > 20:
                df_btc['ret_btc'] = np.log(df_btc['btc_usdt_global'] / df_btc['btc_usdt_global'].shift(1))
                df_btc['ret_btc'] = df_btc['ret_btc'].replace([np.inf, -np.inf], np.nan)
                common = df_btc[['ret', 'ret_btc']].dropna()
                if len(common) > 5:
                    corr_btc = common['ret'].corr(common['ret_btc'])

        # Nuevas métricas
        brecha_velocity = self._get_brecha_velocity(df_v)
        distancia_media_brecha = brecha_mep - mep_gap_avg_24h
        btc_vol_15m = self._get_btc_volatility(df_v, 15)
        btc_vol_1h = self._get_btc_volatility(df_v, 60)

        ctx = self._get_contexto_temporal(ts_ref)

        res = {
            'timestamp': ts_ref,
            'asset': self.asset,
            'p_actual': p_act,
            'min_24h': min_24h,
            'max_24h': max_24h,
            'var_24h': var_24h,
            'volatilidad': volatilidad,
            'posicion_rel': posicion_rel,
            'mep_avg': mep_avg,
            'mep_gap_avg_24h': mep_gap_avg_24h,
            'blue_avg': blue_avg,
            'brecha_mep': brecha_mep,
            'brecha_blue': brecha_blue,
            'corr_btc': corr_btc,
            'vol_proxy': volumen_proxy,
            'n_muestras': len(df_v),
            'brecha_velocity': round(brecha_velocity, 4),
            'distancia_media_brecha': round(distancia_media_brecha, 2),
            'btc_vol_15m': round(btc_vol_15m, 6),
            'btc_vol_1h': round(btc_vol_1h, 6),
            **ctx
        }

        self.guardar_historico(res)

        # Reporte en consola
        self._mostrar_reporte(res, ctx)

        return res

    def _mostrar_reporte(self, res, ctx):
        print(f"\n=== 🌎 MACRO THEORACLE COMPLETO | {res['timestamp'].strftime('%d/%m %H:%M')} ===")
        print(f"🕒 CONTEXTO: {'🏦 BANCARIO' if ctx['es_horario_bancario'] else '🌙 EXTRA-B'} | {'⚠️ FERIADO' if ctx['es_feriado'] else 'DÍA HÁBIL'}")
        print(f"📈 PRECIO: ${res['p_actual']:.2f} | VAR 24H: {res['var_24h']:+.2f}% | VOLAT: {res['volatilidad']:.6f}")
        print(f"📍 RANGO: ${res.get('min_24h',0):.0f} - ${res.get('max_24h',0):.0f} (Posición: {res['posicion_rel']:.1f}%)")
        print(f"💵 BRECHA MEP: {res['brecha_mep']:+.2f}% (Prom 24h: {res['mep_gap_avg_24h']:+.2f}%)")
        print(f"   Velocidad brecha: {res['brecha_velocity']:.4f} %/h | Distancia a media: {res['distancia_media_brecha']:+.2f}%")
        print(f"₿ CORRELACIÓN BTC: {res['corr_btc']:.2f} | Vol BTC 15m: {res['btc_vol_15m']:.6f} | 1h: {res['btc_vol_1h']:.6f}")
        print(f"📦 VOLUMEN PROXY: ${res['vol_proxy']/1e6:.2f}M ARS | MUESTRAS: {res['n_muestras']}")
        print("="*65)

    def guardar_historico(self, reg):
        df_new = pd.DataFrame([reg])
        if not os.path.exists(self.path_macro_hist):
            df_new.to_csv(self.path_macro_hist, index=False)
        else:
            df_new.to_csv(self.path_macro_hist, mode='a', header=False, index=False)
