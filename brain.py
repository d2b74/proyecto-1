import pandas as pd
import os
import numpy as np
import calendar
import json
from datetime import datetime
import pytz  
from micro import MicroEngine
from macro import MacroAnalizadorSystem
from config import CONFIG

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
        self.tz = pytz.timezone('America/Argentina/Buenos_Aires')  # <-- zona horaria

        # Umbrales con valores por defecto
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
            'factor_scalper': 1.0,
            'factor_swing': 0.5,
            'factor_estrategica': 0.8,
            'target_scalper_pct': 0.2,
            'target_swing_pct': 0.4,
            'target_estrategica_pct': 0.8,
            'riesgo_minimo_factor': 0.2,
            'btc_vol_threshold': 0.005,
            'btc_penalty_factor': 0.7,
        }
        self.umbrales = cfg.get('umbrales', defaults)

        os.makedirs(self.base_path, exist_ok=True)
        self.cargar_configuracion_optima()

    def cargar_configuracion_optima(self):
        # (sin cambios, igual que antes)
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
                        fecha = datetime.fromisoformat(fecha_str)
                        dias = (ahora - fecha).days
                        if dias > config[tipo].get('dias_validez', 2):
                            print(f"⚠️  Configuración para {tipo} tiene {dias} días (excede validez).")
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
            self.optimizacion_aplicada = False

    def verificar_estado(self):
        # (sin cambios)
        if not os.path.exists(self.path_estado):
            return True
        try:
            with open(self.path_estado, 'r') as f:
                estado = json.load(f)
            return estado.get('status', 'active') == 'active'
        except:
            return True

    def log_decision(self, micro, macro, tipo, monto, razon, spread_teorico, cumplio_condiciones=True):
        # (sin cambios)
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
        """Lee el archivo de contexto y devuelve el registro más cercano al momento actual."""
        contexto_path = os.path.join(self.data_root, 'contexto', 'contexto.csv')
        if not os.path.exists(contexto_path):
            print("⚠️ No hay archivo de contexto.")
            return {}
        try:
            df = pd.read_csv(contexto_path, parse_dates=['timestamp'])
            if df.empty:
                return {}
            # Tomar el último registro (asumiendo que está ordenado)
            ultimo = df.iloc[-1].to_dict()
            # Verificar antigüedad (no más de 2 horas)
            ahora = datetime.now(self.tz)
            diff = ahora - ultimo['timestamp']
            if diff.total_seconds() > 7200:  # 2 horas
                print(f"⚠️ Contexto demasiado antiguo ({diff.total_seconds()/3600:.1f}h). No se usará.")
                return {}
            return ultimo
        except Exception as e:
            print(f"⚠️ Error leyendo contexto: {e}")
            return {}

    def calcular_capas_de_inteligencia(self, micro, macro):
        # (sin cambios)
        if not macro:
            macro = {}
        brecha_mep = abs(macro.get('brecha_mep', 0))
        b_norm = min(100, max(0, brecha_mep * 12.5))
        volatilidad = macro.get('volatilidad', 0)
        v_norm = min(100, max(0, volatilidad * 45000))
        fuerza = micro.get('fuerza', 1)
        f_norm = 100 - min(100, max(0, fuerza * 50))
        itc = (b_norm * 0.4) + (v_norm * 0.3) + (f_norm * 0.3)

        muro_total = micro.get('muro_c', 0) + micro.get('muro_v', 0)
        f_muro = min(1, muro_total / 50000)
        f_volat = max(0.1, 1 - (volatilidad * 500))
        confianza = round(f_muro * f_volat * 100, 1)

        brecha_abs = abs(macro.get('brecha_mep', 0)) + 0.01
        liq_brecha_ratio = micro.get('muro_c', 0) / brecha_abs

        p_c = micro.get('p_c', 1)
        p_v = micro.get('p_v', 1)
        spread_actual = abs(((p_v / p_c) - 1) * 100)
        tension_spread = spread_actual / (volatilidad * 1000 + 0.001)

        ts = micro['timestamp']
        ultimo_dia = calendar.monthrange(ts.year, ts.month)[1]
        dias_fin_mes = ultimo_dia - ts.day
        semana_mes = (ts.day - 1) // 7 + 1

        return (round(itc, 2), confianza, round(liq_brecha_ratio, 2),
                round(tension_spread, 2), dias_fin_mes, semana_mes)

    def calcular_factor_riesgo_itc(self, itc):
        riesgo_min = self.umbrales.get('riesgo_minimo_factor', 0.2)
        factor = 1 - (itc / 100) * (1 - riesgo_min)
        return max(riesgo_min, min(1, factor))

    def calcular_factor_riesgo_btc(self, macro):
        if not macro:
            return 1.0
        btc_vol = macro.get('btc_vol_15m', 0)
        umbral = self.umbrales.get('btc_vol_threshold', 0.005)
        if btc_vol > umbral:
            exceso = min(btc_vol / umbral - 1, 3)
            factor = max(0.2, 1 - exceso * 0.2)
            return round(factor, 2)
        return 1.0

    def calcular_tamanos_ventanas(self, micro, macro, capital_base, confianza, itc):
        # (sin cambios)
        umbrales = self.umbrales
        p_c = micro.get('p_c', 1)
        p_v = micro.get('p_v', 1)
        spread_actual = ((p_v / p_c) - 1) * 100
        trend_15m = micro.get('trend_15m', 0)
        posicion_rel = macro.get('posicion_rel', 50) if macro else 50

        factor_itc = self.calcular_factor_riesgo_itc(itc)
        factor_btc = self.calcular_factor_riesgo_btc(macro)
        factor_total = factor_itc * factor_btc

        # Scalper
        if (spread_actual > umbrales.get('spread_min_scalper', 0.15) and
            itc <= umbrales.get('itc_threshold_scalper', 65)):
            monto_scalper = capital_base * (confianza / 100) * umbrales.get('factor_scalper', 1.0) * factor_total
            target_scalper = p_c * (1 + umbrales.get('target_scalper_pct', 0.2)/100)
            razon_scalper = f"Spread {spread_actual:.2f}% > umbral, ITC {itc} <= {umbrales['itc_threshold_scalper']}"
        else:
            monto_scalper = 0
            target_scalper = 0
            razon_scalper = f"Spread {spread_actual:.2f}% <= umbral o ITC {itc} > {umbrales.get('itc_threshold_scalper',65)}"

        # Swing
        itc_swing_ok = itc <= umbrales.get('itc_threshold_swing', 40)
        if trend_15m > umbrales.get('trend_min_swing', 0.1) and itc_swing_ok:
            monto_swing = capital_base * (confianza / 100) * umbrales.get('factor_swing', 0.5) * factor_total
            target_swing = p_c * (1 + umbrales.get('target_swing_pct', 0.4)/100)
            razon_swing = f"Trend {trend_15m:.4f}% > umbral, ITC {itc} <= {umbrales.get('itc_threshold_swing',40)}"
        else:
            monto_swing = 0
            target_swing = 0
            razon_swing = f"Trend {trend_15m:.4f}% <= umbral o ITC {itc} > {umbrales.get('itc_threshold_swing',40)}"

        # Estratégica
        itc_estr_ok = itc <= umbrales.get('itc_threshold_estrategica', 30)
        if posicion_rel < umbrales.get('posicion_max_estrategica', 20) and itc_estr_ok:
            monto_estrategica = capital_base * (confianza / 100) * umbrales.get('factor_estrategica', 0.8) * factor_total
            target_estrategica = p_c * (1 + umbrales.get('target_estrategica_pct', 0.8)/100)
            razon_estr = f"Posición {posicion_rel:.1f}% < umbral, ITC {itc} <= {umbrales.get('itc_threshold_estrategica',30)}"
        else:
            monto_estrategica = 0
            target_estrategica = 0
            razon_estr = f"Posición {posicion_rel:.1f}% >= umbral o ITC {itc} > {umbrales.get('itc_threshold_estrategica',30)}"

        self.log_decision(micro, macro, 'scalper', monto_scalper, razon_scalper, spread_actual, monto_scalper>0)
        self.log_decision(micro, macro, 'swing', monto_swing, razon_swing, spread_actual, monto_swing>0)
        self.log_decision(micro, macro, 'estrategica', monto_estrategica, razon_estr, spread_actual, monto_estrategica>0)

        return {
            'scalper_usd': round(monto_scalper, 2),
            'swing_usd': round(monto_swing, 2),
            'estrategica_usd': round(monto_estrategica, 2),
            'target_scalper_usd': round(target_scalper, 2) if target_scalper else 0,
            'target_swing_usd': round(target_swing, 2) if target_swing else 0,
            'target_estrategica_usd': round(target_estrategica, 2) if target_estrategica else 0,
            'spread_actual': round(spread_actual, 2),
            'trend_15m': round(trend_15m, 4),
            'posicion_rel': round(posicion_rel, 1),
            'factor_itc': round(factor_itc, 2),
            'factor_btc': round(factor_btc, 2),
            'factor_total': round(factor_total, 2)
        }

    def ejecutar_ciclo(self):
        if not self.verificar_estado():
            print("⏸️ Sistema en pausa (status.json = paused). No se ejecuta el ciclo.")
            return None

        self.cargar_configuracion_optima()
        print(f"🧠 [MODO DIOS] FUSIÓN TOTAL | {self.asset}")

        # Leer contexto exterior
        contexto = self._obtener_contexto_reciente()

        micro = MicroEngine(self.cfg).run()
        if not micro:
            print("❌ No se pudo obtener análisis micro. Abortando.")
            return None

        macro = MacroAnalizadorSystem(self.cfg).ejecutar(timestamp_referencia=micro['timestamp'])

        itc, confianza, liq_brecha, tension_spread, dfm, semana = self.calcular_capas_de_inteligencia(micro, macro)
        capital_base = micro.get('cap_usd_real', 1000)
        tamanos = self.calcular_tamanos_ventanas(micro, macro, capital_base, confianza, itc)

        registro = {
            **micro,
            **(macro if macro else {}),
            'itc_score': itc,
            'confianza_ejec': confianza,
            'liq_brecha_ratio': liq_brecha,
            'tension_spread': tension_spread,
            'dias_fin_mes': dfm,
            'semana_mes': semana,
            'macro_disponible': 1 if macro else 0,
            **tamanos,
        }

        # Agregar contexto con prefijo
        for k, v in contexto.items():
            if k != 'timestamp':  # no duplicar timestamp
                registro[f'ctx_{k}'] = v

        self._guardar_completo(registro)
        self._reporte_maestro(micro, macro, itc, confianza, liq_brecha, tamanos)
        return registro

    def _guardar_completo(self, registro):
        df_new = pd.DataFrame([registro])
        if os.path.exists(self.path_completo):
            # Leer existente
            df_existing = pd.read_csv(self.path_completo, parse_dates=['timestamp'])
            # Unión de columnas
            todas_columnas = set(df_existing.columns) | set(df_new.columns)
            # Reindexar
            df_existing = df_existing.reindex(columns=todas_columnas)
            df_new = df_new.reindex(columns=todas_columnas)
            # Concatenar
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_final = df_new
        # Guardar (sobrescribe)
        df_final.to_csv(self.path_completo, index=False)
        print(f"📂 Registro guardado en: {self.path_completo}")

    def _reporte_maestro(self, micro, macro, itc, confianza, liq_brecha, tamanos):
        # (sin cambios)
        print("\n" + "█" * 70)
        print(f"🦅 THEORACLE BRAIN | {self.asset} | ITC: {itc} | CONF: {confianza}%")
        if itc < self.umbrales['itc_oportunidad']:
            status = "🟢 ESTABLE"
        elif itc < self.umbrales['itc_peligro']:
            status = "🟡 TENSIÓN"
        else:
            status = "🔴 PELIGRO"
        print(f"STATUS: {status} | {micro['timestamp'].strftime('%d/%m/%Y %H:%M:%S')}")
        print(f"Factor Riesgo (ITC): {tamanos['factor_itc']} | Factor BTC: {tamanos['factor_btc']} | Total: {tamanos['factor_total']}")

        if self.optimizacion_aplicada:
            print("⚙️ REGLAS: Optimización del Backtest Aplicada ✅")
        else:
            print("⚙️ REGLAS: Usando valores por defecto (sin optimización)")

        print(f"Umbrales activos: Scalper ITC ≤ {self.umbrales.get('itc_threshold_scalper',65)} | Spread ≥ {self.umbrales.get('spread_min_scalper',0.15)}%")
        print("─" * 70)

        print(f"📥 COMPRA: ${micro.get('p_c',0):.2f} | VWAP: ${micro.get('vwap_c',0):.2f} | "
              f"Muro: {micro.get('muro_c',0):,.0f} | Elite: {micro.get('nick_c','N/A')}")
        print(f"📤 VENTA:  ${micro.get('p_v',0):.2f} | VWAP: ${micro.get('vwap_v',0):.2f} | "
              f"Muro: {micro.get('muro_v',0):,.0f} | Elite: {micro.get('nick_v','N/A')}")

        if macro:
            mep_avg = macro.get('mep_avg',0)
            brecha_mep = macro.get('brecha_mep',0)
            mep_gap_avg = macro.get('mep_gap_avg_24h',0)
            brecha_velocity = macro.get('brecha_velocity',0)
            distancia_media = macro.get('distancia_media_brecha',0)
            btc_vol = macro.get('btc_vol_15m',0)
            print(f"💵 FINANZAS: MEP ${mep_avg:.2f} | Brecha: {brecha_mep:+.2f}% (Prom: {mep_gap_avg:+.2f}%)")
            print(f"⚡ Velocidad brecha: {brecha_velocity:.4f} %/h | Distancia a media: {distancia_media:+.2f}%")
            print(f"₿ BTC Vol 15m: {btc_vol:.6f}")
            ctx = []
            if macro.get('es_horario_bancario'): ctx.append("🏦 BANCARIO")
            else: ctx.append("🌙 EXTRA-B")
            if macro.get('es_feriado'): ctx.append("⚠️ FERIADO")
            print(f"🕒 CONTEXTO: {' | '.join(ctx)} | Ratio L/B: {liq_brecha:,.0f}")
        else:
            print("💵 FINANZAS: No hay datos macro")

        print("─" * 70)
        print("💡 VEREDICTO POR VENTANA:")
        if tamanos['scalper_usd'] > 0:
            print(f"   🕐 SCALPER (ahora): ${tamanos['scalper_usd']:,.0f} USD  "
                  f"(Spread: {tamanos['spread_actual']:+.2f}%)  🎯 Target: ${tamanos['target_scalper_usd']:.2f}")
        else:
            print(f"   🕐 SCALPER: Sin señal (spread {tamanos['spread_actual']:+.2f}% o ITC > umbral)")

        if tamanos['swing_usd'] > 0:
            print(f"   🕒 SWING (15-60m):   ${tamanos['swing_usd']:,.0f} USD  "
                  f"(Trend: {tamanos['trend_15m']:+.4f}%)  🎯 Target: ${tamanos['target_swing_usd']:.2f}")
        else:
            print(f"   🕒 SWING: Sin señal (trend {tamanos['trend_15m']:+.4f}% o ITC > umbral)")

        if tamanos['estrategica_usd'] > 0:
            print(f"   📅 ESTRATÉGICA (24h): ${tamanos['estrategica_usd']:,.0f} USD  "
                  f"(Posición: {tamanos['posicion_rel']:.1f}%)  🎯 Target: ${tamanos['target_estrategica_usd']:.2f}")
        else:
            print(f"   📅 ESTRATÉGICA: Sin señal (posición {tamanos['posicion_rel']:.1f}% o ITC > umbral)")

        if confianza < self.umbrales.get('confianza_minima',50):
            print("• [RIESGO] 🚩 BAJA CONFIANZA: Puntas inestables.")

        fuerza = micro.get('fuerza',1)
        var_24h = macro.get('var_24h',0) if macro else 0
        if fuerza < self.umbrales.get('fuerza_baja',0.4) and var_24h > self.umbrales.get('var_umbral',0.5):
            print("• [MOMENTUM] ⚠️ DIVERGENCIA: Debilidad de compra vs subida de precio.")

        if macro and abs(macro.get('distancia_media_brecha',0)) > 2.0:
            print(f"• [ALERTA] 📊 Brecha MEP muy desviada de su media: {macro['distancia_media_brecha']:+.2f}%")

        print("█" * 70 + "\n")


if __name__ == "__main__":
    from config import CONFIG
    brain = OrquestadorSystemBrain(CONFIG)
    brain.ejecutar_ciclo()
