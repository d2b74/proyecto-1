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
        self.data_root = cfg['DATA_ROOT']
        self.base_path = os.path.join(self.data_root, self.asset)
        self.path_macro_hist = os.path.join(self.base_path, 'macro_historico.csv')
        self.feriados_arg = holidays.Argentina()

        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def _get_contexto_temporal(self, ts):
        dia = ts.day
        return {
            'hora':              ts.hour,
            'dia_semana':        ts.weekday(),
            'es_finde':          ts.weekday() >= 5,
            'es_feriado':        ts in self.feriados_arg,
            'es_horario_bancario': (ts.weekday() < 5) and (10 <= ts.hour <= 17),
            # ── Patrones temporales argentinos ──────────────────────────────
            'es_fin_de_mes':     dia >= 25,        # bool — presión compradora esperada
            'es_lunes_manana':   (ts.weekday() == 0) and (9 <= ts.hour <= 12),  # pent-up demand
            'semana_del_mes':    (dia - 1) // 7 + 1,  # int 1-5
        }

    def _get_btc_volatility(self, df, window_min=15):

        df_btc = df[df['btc_usdt_global'] > 0].copy()
        if len(df_btc) < 2:
            return 0.0
        df_btc['ret_btc'] = np.log(df_btc['btc_usdt_global'] / df_btc['btc_usdt_global'].shift(1))
        df_btc['ret_btc'] = df_btc['ret_btc'].replace([np.inf, -np.inf], np.nan)
        time_span_min = (df_btc['timestamp'].iloc[-1] - df_btc['timestamp'].iloc[0]).total_seconds() / 60.0
        if time_span_min < 1:
            return 0.0
        approx_rows = max(1, int(window_min * len(df_btc) / time_span_min))
        rets = df_btc['ret_btc'].dropna().iloc[-approx_rows:]
        return float(rets.std()) if len(rets) > 1 else 0.0

    def _get_brecha_velocity(self, df):

        mask = (df['mep_venta'] > 0) & (df['precio'] > 0)
        df_temp = df[mask].copy()
        if len(df_temp) < 5:
            return 0.0

        df_temp['brecha'] = (df_temp['precio'] / df_temp['mep_venta'] - 1) * 100

        # Tiempo en horas desde el primer registro — basado en datos, no en reloj
        df_temp['elapsed_hours'] = (
            df_temp['timestamp'] - df_temp['timestamp'].iloc[0]
        ).dt.total_seconds() / 3600.0

        x = df_temp['elapsed_hours'].values
        y = df_temp['brecha'].values

        var_x = np.var(x)
        if var_x == 0:
            return 0.0

        # Pendiente directa en %/hora
        b = np.cov(x, y)[0, 1] / var_x
        return round(float(b), 4)

    def ejecutar(self, timestamp_referencia=None):
        df = obtener_ventana_fresca(self.data_root, self.asset, horas_atras=24)
        if df is None or df.empty:
            print("⚠️ No hay datos para análisis macro.")
            return None

        if timestamp_referencia is None:
            ts_ref = pd.to_datetime(df['timestamp'].max())
        else:
            ts_ref = pd.to_datetime(timestamp_referencia)

        # Sin look-ahead bias: solo datos hasta ts_ref (inclusive)
        df = df[df['timestamp'] <= ts_ref].copy()

        df_v = df[df['lado'] == 'SELL'].copy().sort_values('timestamp')
        if len(df_v) < 10:
            print("⚠️ Muestras insuficientes para análisis macro (<10).")
            return None

        # Precios y rango
        p_act    = float(df_v['precio'].iloc[-1])
        p_ini    = float(df_v['precio'].iloc[0])
        var_24h  = ((p_act / p_ini) - 1) * 100
        min_24h  = float(df_v['precio'].min())
        max_24h  = float(df_v['precio'].max())
        posicion_rel = (((p_act - min_24h) / (max_24h - min_24h)) * 100
                        if max_24h != min_24h else 50.0)
        posicion_rel = max(0.0, min(100.0, posicion_rel))

        # Volatilidad
        df_v = df_v[df_v['precio'] > 0].copy()
        df_v['ret'] = np.log(df_v['precio'] / df_v['precio'].shift(1))
        df_v['ret'] = df_v['ret'].replace([np.inf, -np.inf], np.nan)
        volatilidad = float(df_v['ret'].dropna().std())
        if np.isnan(volatilidad):
            volatilidad = 0.0

        volumen_proxy = float(df_v['max_single_amount'].sum())

        # Brechas MEP y Blue (usando los últimos 20 registros como referencia reciente)
        df_reciente = df_v.tail(20)
        mep_avg  = float(df_reciente['mep_venta'].mean())
        blue_avg = float(df_reciente['blue_venta'].mean())

        mep_avg_safe  = mep_avg  if (mep_avg  > 0 and not np.isnan(mep_avg))  else p_act
        blue_avg_safe = blue_avg if (blue_avg > 0 and not np.isnan(blue_avg)) else p_act

        brecha_mep  = ((p_act / mep_avg_safe)  - 1) * 100
        brecha_blue = ((p_act / blue_avg_safe) - 1) * 100

        # Media histórica de la brecha MEP en las últimas 24h
        mask_mep = (df_v['mep_venta'] > 0) & (df_v['precio'] > 0)
        brechas_hist = ((df_v.loc[mask_mep, 'precio'] / df_v.loc[mask_mep, 'mep_venta']) - 1) * 100
        mep_gap_avg_24h = float(brechas_hist.mean()) if not brechas_hist.empty else brecha_mep

        # Correlación con BTC
        corr_btc = 0.0
        if 'btc_usdt_global' in df_v.columns:
            df_btc = df_v[df_v['btc_usdt_global'] > 0].copy()
            if len(df_btc) > 20:
                df_btc['ret_btc'] = np.log(df_btc['btc_usdt_global'] / df_btc['btc_usdt_global'].shift(1))
                df_btc['ret_btc'] = df_btc['ret_btc'].replace([np.inf, -np.inf], np.nan)
                common = df_btc[['ret', 'ret_btc']].dropna()
                if len(common) > 5:
                    corr_btc = float(common['ret'].corr(common['ret_btc']))

        brecha_velocity    = self._get_brecha_velocity(df_v)
        distancia_media_brecha = round(brecha_mep - mep_gap_avg_24h, 2)
        btc_vol_15m        = round(self._get_btc_volatility(df_v, 15),  6)
        btc_vol_1h         = round(self._get_btc_volatility(df_v, 60),  6)

        ctx = self._get_contexto_temporal(ts_ref)

        res = {
            'timestamp':             ts_ref,
            'asset':                 self.asset,
            'p_actual':              round(p_act,    2),
            'min_24h':               round(min_24h,  2),
            'max_24h':               round(max_24h,  2),
            'var_24h':               round(var_24h,  4),
            'volatilidad':           round(volatilidad, 6),
            'posicion_rel':          round(posicion_rel, 2),
            'mep_avg':               round(mep_avg,  2),
            'mep_gap_avg_24h':       round(mep_gap_avg_24h, 4),
            'blue_avg':              round(blue_avg, 2),
            'brecha_mep':            round(brecha_mep,  4),
            'brecha_blue':           round(brecha_blue, 4),
            'corr_btc':              round(corr_btc, 4),
            'vol_proxy':             round(volumen_proxy, 0),
            'n_muestras':            len(df_v),
            'brecha_velocity':       brecha_velocity,
            'distancia_media_brecha':distancia_media_brecha,
            'btc_vol_15m':           btc_vol_15m,
            'btc_vol_1h':            btc_vol_1h,
            **ctx
        }

        self.guardar_historico(res)
        self._mostrar_reporte(res, ctx)
        return res

    def _mostrar_reporte(self, res, ctx):
        print(f"\n=== 🌎 MACRO | {res['timestamp'].strftime('%d/%m %H:%M')} ===")
        bancario_str = '🏦 BANCARIO' if ctx['es_horario_bancario'] else '🌙 EXTRA-B'
        feriado_str  = '⚠️ FERIADO'  if ctx['es_feriado']          else 'DÍA HÁBIL'
        extras = []
        if ctx['es_fin_de_mes']:   extras.append("📅 FIN DE MES")
        if ctx['es_lunes_manana']: extras.append("🔔 LUNES MAÑANA")
        extra_str = (' | ' + ' | '.join(extras)) if extras else ''
        print(f"🕒 {bancario_str} | {feriado_str}{extra_str}")
        print(f"📈 PRECIO: ${res['p_actual']:.2f} | VAR 24H: {res['var_24h']:+.2f}% | "
              f"VOLAT: {res['volatilidad']:.6f}")
        print(f"📍 RANGO: ${res['min_24h']:.0f} – ${res['max_24h']:.0f} "
              f"(Posición: {res['posicion_rel']:.1f}%)")
        print(f"💵 BRECHA MEP: {res['brecha_mep']:+.2f}% (Prom 24h: {res['mep_gap_avg_24h']:+.2f}%)")
        print(f"   Vel. brecha: {res['brecha_velocity']:.4f} %/h | "
              f"Dist. a media: {res['distancia_media_brecha']:+.2f}%")
        print(f"₿  CORR BTC: {res['corr_btc']:.2f} | "
              f"Vol BTC 15m: {res['btc_vol_15m']:.6f} | 1h: {res['btc_vol_1h']:.6f}")
        print(f"📦 Vol proxy: ${res['vol_proxy']/1e6:.2f}M ARS | Muestras: {res['n_muestras']}")
        print("=" * 65)

    def guardar_historico(self, reg):

        df_new = pd.DataFrame([reg])
        tmp_path = self.path_macro_hist + '.tmp'

        if os.path.exists(self.path_macro_hist):
            df_existing = pd.read_csv(self.path_macro_hist)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_final = df_new

        df_final.to_csv(tmp_path, index=False)
        os.replace(tmp_path, self.path_macro_hist)
