
import pandas as pd
import numpy as np
import os
import calendar
import json
from datetime import datetime
import pytz
from micro import MicroEngine
from macro import MacroAnalizadorSystem
from config import CONFIG


COLUMNAS_BASE = [
    # ── Micro ──────────────────────────────────────────────────────────────
    'timestamp', 'asset',
    'p_c', 'vwap_c', 'muro_c', 'full_c', 'nick_c',
    'p_v', 'vwap_v', 'muro_v', 'full_v', 'nick_v',
    'fuerza', 'trend_15m', 'cap_usd_real', 'ajustado',
    # ── Macro — precios ────────────────────────────────────────────────────
    'p_actual', 'min_24h', 'max_24h', 'var_24h', 'volatilidad', 'posicion_rel',
    # ── Macro — MEP / Blue ─────────────────────────────────────────────────
    'mep_avg', 'mep_gap_avg_24h', 'blue_avg',
    'brecha_mep', 'brecha_blue',
    # ── Macro — BTC / volumen ──────────────────────────────────────────────
    'corr_btc', 'vol_proxy', 'n_muestras',
    'brecha_velocity', 'distancia_media_brecha',
    'btc_vol_15m', 'btc_vol_1h',
    # ── Contexto temporal (calculado en Macro._get_contexto_temporal) ─────
    # Nota: es_fin_de_mes (bool) ≠ dias_fin_mes (int). Son dos columnas distintas.
    'hora', 'dia_semana', 'es_finde', 'es_feriado', 'es_horario_bancario',
    'es_fin_de_mes',    # bool  — día >= 25 del mes            (Macro)
    'es_lunes_manana',  # bool  — lunes entre 9-12h            (Macro)
    'semana_del_mes',   # int   — semana del mes 1-5           (Macro)
    # ── Capas de inteligencia (calculadas en Brain) ────────────────────────
    'itc_score', 'confianza_ejec', 'liq_brecha_ratio', 'tension_spread',
    'dias_fin_mes',     # int   — días enteros hasta fin de mes (Brain)
    'semana_mes',       # int   — semana del mes (Brain, legacy)
    'macro_disponible',
    # ── Señales y targets ──────────────────────────────────────────────────
    'scalper_usd', 'swing_usd', 'estrategica_usd',
    'target_scalper_usd', 'target_swing_usd', 'target_estrategica_usd',
    'tipo_estrategica',
    # ── Métricas al momento de la señal ───────────────────────────────────
    'spread_actual', 'factor_itc', 'factor_btc', 'factor_total',
]


MAX_EDAD_CONTEXTO_SEG = 7200  # 2 horas


class OrquestadorSystemBrain:

    def __init__(self, cfg):
        self.cfg = cfg
        self.asset = cfg.get('asset', 'USDT').upper()
        self.data_root = cfg['DATA_ROOT']
        self.base_path = os.path.join(self.data_root, self.asset)
        self.path_completo = os.path.join(self.base_path, 'signals.csv')
        self.path_decision_log = os.path.join(self.base_path, 'log.csv')
        self.path_estado = os.path.join(self.data_root, 'status.json')
        self.path_config_optima = os.path.join(self.base_path, 'best.json')
        self.tz = pytz.timezone('America/Argentina/Buenos_Aires')

        defaults = {
            'itc_peligro': 65,
            'itc_oportunidad': 35,
            'confianza_minima': 50,
            'diff_gap_alerta': 0.7,
            'fuerza_baja': 0.4,
            'var_umbral': 0.5,
            'spread_min_scalper': 0.15,
            'trend_min_swing': 0.1,
            'posicion_max_estrategica': 20,
            'itc_threshold_scalper': 65,
            'itc_threshold_swing': 40,
            'itc_threshold_estrategica': 30,
            'itc_threshold_estrategica_a': 45,
            'distancia_brecha_min_a': -0.5,
            'distancia_brecha_min_b': -1.5,
            'itc_threshold_estrategica_b': 35,
            'factor_scalper': 1.0,
            'factor_swing': 1.0,
            'factor_estrategica': 1.0,
            'target_scalper_pct': 0.2,
            'target_swing_pct': 0.4,
            'target_estrategica_pct': 0.8,
            'riesgo_minimo_factor': 0.2,
            'btc_vol_threshold': 0.005,
            'btc_penalty_factor': 0.7,
        }
        self.umbrales = {**defaults, **cfg.get('umbrales', {})}

        os.makedirs(self.base_path, exist_ok=True)
        self.cargar_configuracion_optima()

    def cargar_configuracion_optima(self):
        self.optimizacion_aplicada = False
        if not os.path.exists(self.path_config_optima):
            print("📄 No se encontró best.json. Usando valores por defecto.")
            return
        try:
            with open(self.path_config_optima, 'r') as f:
                config = json.load(f)
            ahora = datetime.now()
            for tipo in ['scalper', 'swing', 'estrategica']:
                if tipo in config:
                    fecha_str = config[tipo].get('fecha_actualizacion')
                    if fecha_str:
                        dias = (ahora - datetime.fromisoformat(fecha_str)).days
                        if dias > config[tipo].get('dias_validez', 2):
                            print(f"⚠️  Config {tipo} tiene {dias} días (excede validez).")
            if 'scalper' in config:
                self.umbrales['itc_threshold_scalper'] = config['scalper'].get('itc_threshold', 65)
                self.umbrales['spread_min_scalper'] = config['scalper'].get('spread_min', 0.15)
            if 'swing' in config:
                self.umbrales['itc_threshold_swing'] = config['swing'].get('itc_threshold', 40)
                self.umbrales['spread_min_swing'] = config['swing'].get('spread_min', 0.1)
            if 'estrategica' in config:
                self.umbrales['itc_threshold_estrategica'] = config['estrategica'].get('itc_threshold', 30)
                self.umbrales['spread_min_estrategica'] = config['estrategica'].get('spread_min', 0.05)
            self.optimizacion_aplicada = True
            print("✅ Configuración óptima cargada desde best.json.")
        except Exception as e:
            print(f"❌ Error al cargar best.json: {e}. Usando valores por defecto.")

    def verificar_estado(self):
        if not os.path.exists(self.path_estado):
            return True
        try:
            with open(self.path_estado, 'r') as f:
                estado = json.load(f)
            return estado.get('status', 'active') == 'active'
        except:
            return True

    def log_decision(self, micro, macro, tipo, monto, razon, spread_teorico, cumplio_condiciones=True):
        registro = {
            'timestamp': micro['timestamp'],
            'tipo': tipo,
            'monto_sugerido': monto,
            'razon': razon,
            'spread_teorico': spread_teorico,
            'itc': micro.get('itc_score', None),
            'confianza': micro.get('confianza_ejec', None),
            'hora_entrada': micro['timestamp'].hour,
            'dia_semana_entrada': micro['timestamp'].weekday(),
            'cumplio_condiciones': cumplio_condiciones,
            'precio_compra': micro.get('p_c', 0),
            'precio_venta': micro.get('p_v', 0),
        }
        if macro:
            registro.update({
                'brecha_mep': macro.get('brecha_mep', None),
                'btc_vol': macro.get('btc_vol_15m', None),
                'es_horario_bancario': macro.get('es_horario_bancario', None),
            })
        df_new = pd.DataFrame([registro])
        header = not os.path.exists(self.path_decision_log)
        try:
            df_new.to_csv(self.path_decision_log, mode='a', header=header, index=False)
        except Exception as e:
            print(f"❌ Error escribiendo en log.csv: {e}")

    def _obtener_contexto_reciente(self):
        contexto_path = os.path.join(self.data_root, 'contexto', 'contexto.csv')
        if not os.path.exists(contexto_path):
            print("⚠️ No hay archivo de contexto.")
            return {}
        try:
            df = pd.read_csv(contexto_path, parse_dates=['timestamp'])
            if df.empty:
                return {}
            ultimo = df.iloc[-1].to_dict()
            ts = ultimo['timestamp']
            if ts.tzinfo is None:
                ts = self.tz.localize(ts)
            edad_seg = (datetime.now(self.tz) - ts).total_seconds()
            if edad_seg > MAX_EDAD_CONTEXTO_SEG:
                print(f"⚠️ Contexto con {edad_seg/3600:.1f}h de antigüedad "
                      f"(límite: {MAX_EDAD_CONTEXTO_SEG/3600:.0f}h). No se usará.")
                return {}
            return ultimo
        except Exception as e:
            print(f"⚠️ Error leyendo contexto: {e}")
            return {}

    def calcular_capas_de_inteligencia(self, micro, macro):
        if not macro:
            macro = {}

        # ITC con brecha con signo:
        #   brecha positiva → prima normal → suma tensión
        #   brecha negativa → anomalía/oportunidad → NO suma tensión
        brecha_mep_signed = macro.get('brecha_mep', 0)
        b_norm = min(100, max(0, brecha_mep_signed * 12.5)) if brecha_mep_signed > 0 else 0

        volatilidad = macro.get('volatilidad', 0)
        v_norm = min(100, max(0, volatilidad * 45000))

        fuerza = micro.get('fuerza', 1)
        f_norm = 100 - min(100, max(0, fuerza * 50))

        itc = (b_norm * 0.4) + (v_norm * 0.3) + (f_norm * 0.3)

        muro_total = micro.get('muro_c', 0) + micro.get('muro_v', 0)
        f_muro = min(1, muro_total / 50000)
        f_volat = max(0.1, 1 - (volatilidad * 500))
        confianza = round(f_muro * f_volat * 100, 1)

        brecha_abs = abs(brecha_mep_signed) + 0.01
        liq_brecha_ratio = micro.get('muro_c', 0) / brecha_abs

        p_c = micro.get('p_c', 1)
        p_v = micro.get('p_v', 1)
        # Spread con signo: positivo = operable, negativo = mercado cruzado
        spread_actual = ((p_v / p_c) - 1) * 100
        tension_spread = spread_actual / (volatilidad * 1000 + 0.001)

        ts = micro['timestamp']
        ultimo_dia = calendar.monthrange(ts.year, ts.month)[1]
        dias_fin_mes = ultimo_dia - ts.day   # int: días hasta fin de mes
        semana_mes = (ts.day - 1) // 7 + 1  # int: semana del mes (1-5)

        return (round(itc, 2), confianza, round(liq_brecha_ratio, 2),
                round(tension_spread, 2), dias_fin_mes, semana_mes)

    def calcular_factor_riesgo_itc(self, itc):
        riesgo_min = self.umbrales.get('riesgo_minimo_factor', 0.2)
        factor = 1 - (itc / 100) * (1 - riesgo_min)
        return round(max(riesgo_min, min(1, factor)), 2)

    def calcular_factor_riesgo_btc(self, macro):
        if not macro:
            return 1.0
        btc_vol = macro.get('btc_vol_15m', 0)
        umbral = self.umbrales.get('btc_vol_threshold', 0.005)
        if btc_vol > umbral:
            exceso = min(btc_vol / umbral - 1, 3)
            return round(max(0.2, 1 - exceso * 0.2), 2)
        return 1.0

    def calcular_tamanos_ventanas(self, micro, macro, itc):
        """
        Capital: siempre 100% del total configurado (concentración, no fragmentación).
        El operador elige manualmente UNA señal entre las que se activan.
        """
        umbrales = self.umbrales
        capital_base = self.cfg['capital_usd']

        p_c = micro.get('p_c', 1)
        p_v = micro.get('p_v', 1)
        spread_actual = ((p_v / p_c) - 1) * 100   # con signo, sin abs()
        trend_15m = micro.get('trend_15m', 0)
        posicion_rel = macro.get('posicion_rel', 50) if macro else 50
        distancia_media = macro.get('distancia_media_brecha', 0) if macro else 0

        factor_itc = self.calcular_factor_riesgo_itc(itc)
        factor_btc = self.calcular_factor_riesgo_btc(macro)
        factor_total = round(factor_itc * factor_btc, 2)

        # ── SCALPER ──────────────────────────────────────────────────────────
        scalper_ok = (
            spread_actual > umbrales.get('spread_min_scalper', 0.15) and
            itc <= umbrales.get('itc_threshold_scalper', 65)
        )
        if scalper_ok:
            monto_scalper = capital_base * umbrales.get('factor_scalper', 1.0) * factor_total
            target_scalper = p_c * (1 + umbrales.get('target_scalper_pct', 0.2) / 100)
            razon_scalper = (f"Spread {spread_actual:.2f}% > umbral | "
                             f"ITC {itc:.1f} ≤ {umbrales['itc_threshold_scalper']}")
        else:
            monto_scalper, target_scalper = 0, 0
            if spread_actual <= 0:
                razon_no = "mercado cruzado"
            elif spread_actual <= umbrales.get('spread_min_scalper', 0.15):
                razon_no = f"spread {spread_actual:.2f}% ≤ umbral"
            else:
                razon_no = f"ITC {itc:.1f} > {umbrales.get('itc_threshold_scalper', 65)}"
            razon_scalper = f"Sin señal: {razon_no}"

        # ── SWING ─────────────────────────────────────────────────────────────
        swing_ok = (
            trend_15m > umbrales.get('trend_min_swing', 0.1) and
            itc <= umbrales.get('itc_threshold_swing', 40)
        )
        if swing_ok:
            monto_swing = capital_base * umbrales.get('factor_swing', 1.0) * factor_total
            target_swing = p_c * (1 + umbrales.get('target_swing_pct', 0.4) / 100)
            razon_swing = (f"Trend 15m {trend_15m:.4f}% > umbral | "
                           f"ITC {itc:.1f} ≤ {umbrales.get('itc_threshold_swing', 40)}")
        else:
            monto_swing, target_swing = 0, 0
            razon_swing = (f"Sin señal: trend {trend_15m:.4f}% o "
                           f"ITC {itc:.1f} > {umbrales.get('itc_threshold_swing', 40)}")

        # ── ESTRATÉGICA (Tipo A / Tipo B) ─────────────────────────────────────
        # Tipo A: precio en piso de 24h + ITC moderado + brecha debajo de media
        # Tipo B: anomalía pura de brecha MEP, independiente del precio nominal
        cond_estr_a = (
            posicion_rel < umbrales.get('posicion_max_estrategica', 20) and
            itc <= umbrales.get('itc_threshold_estrategica_a', 45) and
            distancia_media < umbrales.get('distancia_brecha_min_a', -0.5)
        )
        cond_estr_b = (
            distancia_media < umbrales.get('distancia_brecha_min_b', -1.5) and
            itc <= umbrales.get('itc_threshold_estrategica_b', 35)
        )

        if cond_estr_a or cond_estr_b:
            tipo_estr = 'A' if cond_estr_a else 'B'
            monto_estrategica = capital_base * umbrales.get('factor_estrategica', 1.0) * factor_total
            target_estrategica = p_c * (1 + umbrales.get('target_estrategica_pct', 0.8) / 100)
            if cond_estr_a:
                razon_estr = (f"[TIPO A] Posición {posicion_rel:.1f}% < umbral | "
                              f"ITC {itc:.1f} ≤ {umbrales.get('itc_threshold_estrategica_a', 45)} | "
                              f"Brecha {distancia_media:+.2f}%")
            else:
                razon_estr = (f"[TIPO B] Anomalía brecha: {distancia_media:+.2f}% < "
                              f"{umbrales.get('distancia_brecha_min_b', -1.5)}% | "
                              f"ITC {itc:.1f} ≤ {umbrales.get('itc_threshold_estrategica_b', 35)}")
        else:
            tipo_estr = None
            monto_estrategica, target_estrategica = 0, 0
            razon_estr = (f"Sin señal: posición {posicion_rel:.1f}% | "
                          f"brecha {distancia_media:+.2f}% | ITC {itc:.1f}")

        self.log_decision(micro, macro, 'scalper', monto_scalper, razon_scalper, spread_actual, monto_scalper > 0)
        self.log_decision(micro, macro, 'swing', monto_swing, razon_swing, spread_actual, monto_swing > 0)
        self.log_decision(micro, macro, 'estrategica', monto_estrategica, razon_estr, spread_actual, monto_estrategica > 0)

        return {
            'scalper_usd':            round(monto_scalper, 2),
            'swing_usd':              round(monto_swing, 2),
            'estrategica_usd':        round(monto_estrategica, 2),
            'target_scalper_usd':     round(target_scalper, 2)     if target_scalper     else 0,
            'target_swing_usd':       round(target_swing, 2)       if target_swing       else 0,
            'target_estrategica_usd': round(target_estrategica, 2) if target_estrategica else 0,
            'tipo_estrategica':       tipo_estr,
            'spread_actual':          round(spread_actual, 2),
            'trend_15m':              round(trend_15m, 4),
            'posicion_rel':           round(posicion_rel, 1),
            'factor_itc':             factor_itc,
            'factor_btc':             factor_btc,
            'factor_total':           factor_total,
        }

    def ejecutar_ciclo(self):
        if not self.verificar_estado():
            print("⏸️ Sistema en pausa. No se ejecuta el ciclo.")
            return None

        self.cargar_configuracion_optima()
        print(f"🧠 [ORÁCULO] | {self.asset}")

        contexto = self._obtener_contexto_reciente()

        micro = MicroEngine(self.cfg).run()
        if not micro:
            print("❌ No se pudo obtener análisis micro. Abortando.")
            return None

        macro = MacroAnalizadorSystem(self.cfg).ejecutar(timestamp_referencia=micro['timestamp'])

        itc, confianza, liq_brecha, tension_spread, dfm, semana = \
            self.calcular_capas_de_inteligencia(micro, macro)

        tamanos = self.calcular_tamanos_ventanas(micro, macro, itc)

        registro = {
            **micro,
            **(macro if macro else {}),
            'itc_score':        itc,
            'confianza_ejec':   confianza,
            'liq_brecha_ratio': liq_brecha,
            'tension_spread':   tension_spread,
            'dias_fin_mes':     dfm,       # int — días hasta fin de mes    (Brain)
            'semana_mes':       semana,    # int — semana del mes 1-5        (Brain)
            'macro_disponible': 1 if macro else 0,
            **tamanos,
        }

        # Prefijo ctx_ para contexto exterior (nunca colisiona con columnas de Macro)
        for k, v in contexto.items():
            if k != 'timestamp':
                registro[f'ctx_{k}'] = v

        self._guardar_completo(registro)
        self._reporte_maestro(micro, macro, itc, confianza, liq_brecha, tamanos)
        return registro

    def _guardar_completo(self, registro):
        """
        Escritura atómica + columnas estables.

        Garantías:
          1. Escritura atómica (.tmp → os.replace): el Brain nunca lee un CSV
             parcialmente escrito aunque el proceso sea interrumpido.
          2. Columnas nuevas en el registro actual → NaN en filas históricas.
             Las columnas NUNCA se corren de lugar: lectura siempre por nombre.
          3. Orden fijo: COLUMNAS_BASE + ctx_* alfabético.
        """
        df_new = pd.DataFrame([registro])

        cols_ctx_actuales = sorted([c for c in registro.keys() if c.startswith('ctx_')])
        columnas_target = COLUMNAS_BASE + cols_ctx_actuales

        if os.path.exists(self.path_completo):
            df_existing = pd.read_csv(self.path_completo, parse_dates=['timestamp'])

            # Unión de ctx_* conocidas (históricas) + nuevas
            cols_ctx_hist = sorted([c for c in df_existing.columns if c.startswith('ctx_')])
            todos_ctx = sorted(set(cols_ctx_hist) | set(cols_ctx_actuales))
            columnas_target = COLUMNAS_BASE + todos_ctx

            # Añadir columnas faltantes con NaN (sin correr datos existentes)
            for col in columnas_target:
                if col not in df_existing.columns:
                    df_existing[col] = pd.NA
                if col not in df_new.columns:
                    df_new[col] = pd.NA

            df_existing = df_existing.reindex(columns=columnas_target)
            df_new      = df_new.reindex(columns=columnas_target)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_new   = df_new.reindex(columns=columnas_target)
            df_final = df_new

        # Escritura atómica
        tmp_path = self.path_completo + '.tmp'
        df_final.to_csv(tmp_path, index=False)
        os.replace(tmp_path, self.path_completo)   # atómico en el mismo filesystem
        print(f"📂 Registro guardado ({len(df_final)} filas): {self.path_completo}")

    def _reporte_maestro(self, micro, macro, itc, confianza, liq_brecha, tamanos):
        print("\n" + "█" * 70)
        print(f"🦅 ORÁCULO | {self.asset} | ITC: {itc:.1f} | CONF: {confianza}%")

        if itc < self.umbrales['itc_oportunidad']:
            status = "🟢 ESTABLE"
        elif itc < self.umbrales['itc_peligro']:
            status = "🟡 TENSIÓN"
        else:
            status = "🔴 PELIGRO"
        print(f"STATUS: {status} | {micro['timestamp'].strftime('%d/%m/%Y %H:%M:%S')}")
        print(f"Factor Riesgo: ITC={tamanos['factor_itc']} | BTC={tamanos['factor_btc']} "
              f"| Total={tamanos['factor_total']}")
        estado_cfg = "✅ Optimización aplicada" if self.optimizacion_aplicada else "⚙️ Valores por defecto"
        print(f"Reglas: {estado_cfg} | Capital: ${self.cfg['capital_usd']:,.0f} USD (100%)")
        print(f"Scalper: ITC ≤ {self.umbrales.get('itc_threshold_scalper',65)} | "
              f"Spread ≥ {self.umbrales.get('spread_min_scalper',0.15)}%")
        print("─" * 70)

        spread = tamanos['spread_actual']
        spread_tag = f"{spread:+.2f}%" + (" ✅" if spread > 0 else " ❌ CRUZADO")
        print(f"📥 COMPRA: ${micro.get('p_c',0):.2f} | VWAP: ${micro.get('vwap_c',0):.2f} | "
              f"Muro: {micro.get('muro_c',0):,.0f} | {micro.get('nick_c','N/A')}")
        print(f"📤 VENTA:  ${micro.get('p_v',0):.2f} | VWAP: ${micro.get('vwap_v',0):.2f} | "
              f"Spread: {spread_tag}")

        if macro:
            print(f"💵 MEP: ${macro.get('mep_avg',0):.2f} | "
                  f"Brecha: {macro.get('brecha_mep',0):+.2f}% "
                  f"(Prom 24h: {macro.get('mep_gap_avg_24h',0):+.2f}%)")
            print(f"⚡ Vel. brecha: {macro.get('brecha_velocity',0):.4f} %/h | "
                  f"Dist. a media: {macro.get('distancia_media_brecha',0):+.2f}%")
            print(f"₿  BTC Vol 15m: {macro.get('btc_vol_15m',0):.6f}")
            alertas = []
            if macro.get('es_fin_de_mes'):
                alertas.append("📅 FIN DE MES — presión compradora esperada")
            if macro.get('es_lunes_manana'):
                alertas.append("🔔 LUNES MAÑANA — spreads artificiales posibles")
            if not macro.get('es_horario_bancario'):
                alertas.append("🌙 FUERA DE HORARIO BANCARIO")
            if macro.get('es_feriado'):
                alertas.append("⚠️ FERIADO")
            for a in alertas:
                print(f"🕒 {a}")
            print(f"   Ratio L/B: {liq_brecha:,.0f}")
        else:
            print("💵 FINANZAS: No hay datos macro disponibles.")

        print("─" * 70)
        print("💡 VEREDICTO:")

        if tamanos['scalper_usd'] > 0:
            print(f"   🕐 SCALPER:     ${tamanos['scalper_usd']:,.0f} USD | "
                  f"Spread {spread:+.2f}% | 🎯 ${tamanos['target_scalper_usd']:.2f}")
        else:
            print(f"   🕐 SCALPER:     Sin señal")

        if tamanos['swing_usd'] > 0:
            print(f"   🕒 SWING:       ${tamanos['swing_usd']:,.0f} USD | "
                  f"Trend {tamanos['trend_15m']:+.4f}% | 🎯 ${tamanos['target_swing_usd']:.2f}")
        else:
            print(f"   🕒 SWING:       Sin señal")

        if tamanos['estrategica_usd'] > 0:
            print(f"   📅 ESTRATÉGICA: ${tamanos['estrategica_usd']:,.0f} USD "
                  f"[TIPO {tamanos.get('tipo_estrategica','?')}] | "
                  f"🎯 ${tamanos['target_estrategica_usd']:.2f}")
        else:
            print(f"   📅 ESTRATÉGICA: Sin señal")

        # Alertas de riesgo
        if confianza < self.umbrales.get('confianza_minima', 50):
            print("• [RIESGO] 🚩 BAJA CONFIANZA: Puntas inestables.")
        fuerza = micro.get('fuerza', 1)
        var_24h = macro.get('var_24h', 0) if macro else 0
        if fuerza < self.umbrales.get('fuerza_baja', 0.4) and var_24h > self.umbrales.get('var_umbral', 0.5):
            print("• [MOMENTUM] ⚠️ DIVERGENCIA: Debilidad de compra vs subida de precio.")
        if macro and abs(macro.get('distancia_media_brecha', 0)) > 2.0:
            print(f"• [ALERTA] 📊 Brecha MEP muy desviada: {macro['distancia_media_brecha']:+.2f}%")
        if spread <= 0:
            print("• [MERCADO] 🚫 SPREAD NEGATIVO: Mercado cruzado. Scalper inhabilitado.")

        print("█" * 70 + "\n")


if __name__ == "__main__":
    from config import CONFIG
    brain = OrquestadorSystemBrain(CONFIG)
    brain.ejecutar_ciclo()
