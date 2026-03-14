import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from itertools import product
from config import CONFIG

try:
    from brain import COLUMNAS_BASE
except ImportError:
    COLUMNAS_BASE = []


class BacktestEngine:

    def __init__(self, df, cfg):
        self.cfg       = cfg
        self.data_root = CONFIG['DATA_ROOT']
        self.asset     = CONFIG['asset']
        self.df        = df.copy()
        self.df.sort_values('timestamp', inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.resultados            = None
        self.scalper_filtrados_btc = 0
        self.scalper_totales       = 0

    # ── Diagnóstico de datos de entrada ────────────────────────────────────────
    def diagnosticar_datos(self):
        """
        Verifica que signals.csv tenga datos suficientes para el backtest.
        Imprime un resumen antes de ejecutar para detectar problemas silenciosos.
        """
        print("\n📋 DIAGNÓSTICO DE DATOS")
        print(f"   Total filas en signals.csv: {len(self.df)}")
        if self.df.empty:
            print("   ❌ El DataFrame está vacío. Abortando.")
            return False

        ts_min = self.df['timestamp'].min()
        ts_max = self.df['timestamp'].max()
        span_h = (ts_max - ts_min).total_seconds() / 3600
        print(f"   Rango temporal: {ts_min.strftime('%d/%m %H:%M')} → {ts_max.strftime('%d/%m %H:%M')}")
        print(f"   Ventana disponible: {span_h:.1f} horas")

        for tipo in ['scalper', 'swing', 'estrategica']:
            col = self.cfg.get(f'col_señal_{tipo}', f'{tipo}_usd')
            if col in self.df.columns:
                n = (self.df[col] > 0).sum()
                print(f"   Señales {tipo}: {n}")
            else:
                print(f"   ⚠️ Columna '{col}' no encontrada en signals.csv")

        timeout_max = max(self.cfg['timeouts'].values())
        if span_h < (timeout_max / 60):
            print(f"\n   ⚠️ ADVERTENCIA: La ventana de datos ({span_h:.1f}h) es menor que el "
                  f"timeout máximo ({timeout_max/60:.0f}h).")
            print(f"   Las operaciones de tipo 'estrategica' o 'swing' probablemente "
                  f"terminen por timeout con precio de cierre, no por target/stop.")
            print(f"   El backtest es válido cuando tenés al menos {timeout_max/60:.0f}h de historial.\n")
        else:
            print(f"   ✅ Ventana suficiente para simular todos los timeouts.\n")

        ctx_btc = 'ctx_btc_change_24h'
        if ctx_btc not in self.df.columns:
            print(f"   ⚠️ Columna '{ctx_btc}' no encontrada. "
                  f"El filtro BTC de scalper usará 0 como default.")

        # Diagnóstico de ventana futura por tipo
        print("\n   Ventana futura disponible por tipo:")
        ts_corte = self.df['timestamp'].max()
        for tipo_d, t_min in self.cfg['timeouts'].items():
            ts_limite = ts_corte - timedelta(minutes=t_min)
            n_con_ventana = (self.df['timestamp'] <= ts_limite).sum()
            pct = n_con_ventana / max(len(self.df), 1) * 100
            ok_d = '✅' if pct > 10 else '⚠️'
            print(f"   {ok_d} {tipo_d:12}: {n_con_ventana} señales con ventana "
                  f"completa ({pct:.0f}% del dataset)")

        return True

    def _get_future_window(self, start_ts, max_minutes):
        end_ts = start_ts + timedelta(minutes=max_minutes)
        mask   = (self.df['timestamp'] >= start_ts) & (self.df['timestamp'] <= end_ts)
        return self.df.loc[mask].copy()

    def simular_operacion(self, row, tipo):
        col_señal = self.cfg[f'col_señal_{tipo}']
        monto     = row.get(col_señal, 0)
        if monto <= 0:
            return None

        # Filtro BTC para scalper
        if tipo == 'scalper':
            btc_change = row.get('ctx_btc_change_24h', 0)
            if pd.isna(btc_change):
                btc_change = 0
            if btc_change <= -1.0:
                return None

        target_pct  = self.cfg['targets'][tipo] / 100
        stop_pct    = self.cfg['stops'][tipo]   / 100
        timeout_min = self.cfg['timeouts'][tipo]

        # Slippage: desactivado por defecto sobre VWAP (ya modela ejecución real)
        slippage = (self.cfg.get('slippage_pct', 0) / 100
                    if self.cfg.get('slippage_sobre_vwap', False) else 0.0)

        precio_entrada = row[self.cfg['precio_entrada']] * (1 + slippage)
        target_price   = precio_entrada * (1 + target_pct)
        stop_price     = precio_entrada * (1 - stop_pct)

        future_window = self._get_future_window(row['timestamp'], timeout_min)
        if len(future_window) <= 1:
            return None

        future_window    = future_window.iloc[1:]
        precio_salida    = None
        tiempo_salida    = None
        drawdown_max     = 0
        target_alcanzado = False
        stop_alcanzado   = False
        precio_min       = precio_entrada

        for _, fut_row in future_window.iterrows():
            precio_venta = fut_row['p_v'] * (1 - slippage)
            if precio_venta < precio_min:
                precio_min   = precio_venta
                drawdown_max = max(drawdown_max,
                                   (precio_entrada - precio_min) / precio_entrada * 100)
            if precio_venta <= stop_price:
                stop_alcanzado = True
                precio_salida  = precio_venta
                tiempo_salida  = (fut_row['timestamp'] - row['timestamp']).total_seconds() / 60
                break
            if precio_venta >= target_price:
                target_alcanzado = True
                precio_salida    = precio_venta
                tiempo_salida    = (fut_row['timestamp'] - row['timestamp']).total_seconds() / 60
                break

        if precio_salida is None:
            precio_salida = future_window.iloc[-1]['p_v'] * (1 - slippage)
            tiempo_salida = timeout_min

        retorno_pct  = (precio_salida / precio_entrada - 1) * 100
        ganancia_usd = monto * (retorno_pct / 100)
        fecha_salida = row['timestamp'] + timedelta(minutes=tiempo_salida)

        resultado = {
            'timestamp':          row['timestamp'],
            'tipo':               tipo,
            'monto_usd':          monto,
            'resultado':          ('target' if target_alcanzado
                                   else 'stop' if stop_alcanzado else 'timeout'),
            'precio_entrada':     round(precio_entrada, 2),
            'precio_salida':      round(precio_salida,  2),
            'retorno_pct':        round(retorno_pct,    2),
            'ganancia_usd':       round(ganancia_usd,   2),
            'tiempo_min':         round(tiempo_salida,  1),
            'duracion_horas':     round(tiempo_salida / 60, 1),
            'drawdown_max':       round(drawdown_max,   2),
            'target_alcanzado':   target_alcanzado,
            'stop_alcanzado':     stop_alcanzado,
            'exito':              target_alcanzado,
            'hora_entrada':       row['timestamp'].hour,
            'dia_semana_entrada': row['timestamp'].weekday(),
            'hora_salida':        fecha_salida.hour,
            'dia_semana_salida':  fecha_salida.weekday(),
        }
        for var in self.cfg.get('cluster_vars', []):
            if var in row:
                val = row[var]
                resultado[var] = None if pd.isna(val) else val
        return resultado

    def ejecutar_backtest(self):
        print(f"🔄 Ejecutando backtest sobre {len(self.df)} registros...")
        self.scalper_filtrados_btc = 0
        self.scalper_totales       = 0
        resultados = []

        for idx, row in self.df.iterrows():
            if idx % 500 == 0:
                print(f"   Procesando {idx}/{len(self.df)}...")

            # Conteo filtro BTC — NO usar continue acá
            if row.get('scalper_usd', 0) > 0:
                self.scalper_totales += 1
                btc_change = row.get('ctx_btc_change_24h', 0)
                if pd.isna(btc_change):
                    btc_change = 0
                if btc_change <= -1.0:
                    self.scalper_filtrados_btc += 1

            for tipo in ['scalper', 'swing', 'estrategica']:
                res = self.simular_operacion(row, tipo)
                if res:
                    resultados.append(res)

        self.resultados = pd.DataFrame(resultados) if resultados else pd.DataFrame()
        print(f"✅ Backtest completado: {len(self.resultados)} operaciones simuladas.")
        print(f"   Scalper totales: {self.scalper_totales} | Filtrados BTC: {self.scalper_filtrados_btc}")
        return self.resultados

    def guardar_resultados(self):
        """
        Guarda self.resultados en resultados_backtest.csv.
        SIEMPRE escribe el archivo — aunque esté vacío — para que
        rclone detecte el cambio por timestamp y lo suba a Drive.
        """
        path = os.path.join(self.data_root, self.asset, 'resultados_backtest.csv')
        tmp  = path + '.tmp'

        if self.resultados is None or self.resultados.empty:

            df_vacio = pd.DataFrame([{
                'timestamp':    datetime.now().isoformat(),
                'tipo':         '_info',
                'nota':         '0 operaciones simuladas en este período',
                'historia_h':   round(
                    (self.df['timestamp'].max() - self.df['timestamp'].min())
                    .total_seconds() / 3600, 1),
            }])
            df_vacio.to_csv(tmp, index=False)
            os.replace(tmp, path)
            print(f"💾 resultados_backtest.csv — 0 trades (archivo actualizado) → {path}")
            return

        self.resultados.to_csv(tmp, index=False)
        os.replace(tmp, path)
        print(f"💾 resultados_backtest.csv guardado: {len(self.resultados)} filas → {path}")

    def analizar_por_cluster(self, columna, bins=None, df=None):
        if df is None:
            if self.resultados is None or self.resultados.empty:
                raise ValueError("Ejecuta el backtest primero.")
            df = self.resultados.copy()
        else:
            df = df.copy()
        if bins is not None and pd.api.types.is_numeric_dtype(df[columna]):
            df['cluster'] = pd.cut(df[columna], bins=bins)
            grupo = df.groupby('cluster', observed=False)
        else:
            grupo = df.groupby(columna, observed=False)
        stats = []
        for nombre, g in grupo:
            stats.append({
                'cluster':          nombre,
                'operaciones':      len(g),
                'win_rate':         round(g['exito'].mean() * 100, 1),
                'ganancia_usd':     round(g['ganancia_usd'].sum(), 2),
                'drawdown_prom':    round(g['drawdown_max'].mean(), 2),
                'duracion_prom_hs': round(g['duracion_horas'].mean(), 1),
            })
        return pd.DataFrame(stats)

    def analizar_por_duracion(self, tipo=None, bins=None, labels=None):
        if self.resultados is None or self.resultados.empty:
            return None
        df     = self.resultados.copy()
        if tipo:
            df = df[df['tipo'] == tipo].copy()
            if df.empty:
                return None
        bins   = bins   or self.cfg.get('bins_duracion',  [0, 4, 12, 24, float('inf')])
        labels = labels or self.cfg.get('labels_duracion', ['0-4h', '4-12h', '12-24h', '24h+'])
        df['rango_duracion'] = pd.cut(df['duracion_horas'], bins=bins, labels=labels, right=False)
        return df.groupby('rango_duracion', observed=False).agg(
            operaciones=('exito', 'count'),
            win_rate=('exito', lambda x: round(x.mean() * 100, 1)),
            ganancia_total=('ganancia_usd', 'sum'),
            drawdown_prom=('drawdown_max', 'mean'),
            duracion_prom=('duracion_horas', 'mean')
        ).reset_index()

    def comparar_con_factor_riesgo(self, col_factor='factor_total', tipo=None):
        if self.resultados is None or self.resultados.empty:
            return None
        df_res = self.resultados.copy()
        if tipo:
            df_res = df_res[df_res['tipo'] == tipo].copy()
            if df_res.empty:
                return None
        df_factor = self.df.set_index('timestamp')[col_factor].to_dict()
        df_res['factor'] = df_res['timestamp'].map(df_factor)
        if df_res['factor'].isna().all():
            print(f"   No se encontró '{col_factor}' en los datos originales.")
            return None
        mediana = df_res['factor'].median()
        alto    = df_res[df_res['factor'] >= mediana]
        bajo    = df_res[df_res['factor'] <  mediana]
        return pd.DataFrame({
            'grupo':          ['Factor >= mediana', 'Factor < mediana'],
            'operaciones':    [len(alto), len(bajo)],
            'win_rate':       [round(alto['exito'].mean()*100,1), round(bajo['exito'].mean()*100,1)],
            'ganancia_total': [round(alto['ganancia_usd'].sum(),2), round(bajo['ganancia_usd'].sum(),2)],
            'drawdown_prom':  [round(alto['drawdown_max'].mean(),2), round(bajo['drawdown_max'].mean(),2)],
        })

    def optimizar_umbrales(self, rango_itc, rango_spread, tipo='scalper', metric='win_rate'):
        resultados = []
        for itc, spread in product(rango_itc, rango_spread):
            mask    = (self.df['itc_score'] <= itc) & (self.df['spread_actual'] >= spread)
            df_filt = self.df[mask].copy()
            if len(df_filt) < 10:
                continue
            engine_temp = BacktestEngine(df_filt, self.cfg)
            engine_temp.ejecutar_backtest()
            res = engine_temp.resultados
            if res is None or res.empty:
                continue
            res = res[res['tipo'] == tipo]
            if res.empty:
                continue
            if metric == 'win_rate':
                valor = res['exito'].mean() * 100
            elif metric == 'ganancia_total':
                valor = res['ganancia_usd'].sum()
            elif metric == 'sharpe':
                retornos = res['retorno_pct'] / 100
                valor = retornos.mean() / (retornos.std() + 1e-6)
            else:
                valor = 0
            resultados.append({
                'itc': itc, 'spread': spread, 'operaciones': len(res),
                metric: round(valor, 2),
                'win_rate': round(res['exito'].mean()*100, 1),
                'ganancia_total': round(res['ganancia_usd'].sum(), 2),
            })
        df_res = pd.DataFrame(resultados)
        if df_res.empty:
            return df_res, None
        mejor_idx   = df_res[metric].idxmax()
        mejor_combo = (df_res.loc[mejor_idx, 'itc'], df_res.loc[mejor_idx, 'spread'])
        return df_res, mejor_combo

    def actualizar_best_json(self, umbrales_opt=None):
        """
        Siempre escribe best.json con métricas reales del backtest actual.
        Si se pasan umbrales optimizados, los incorpora.
        Nunca borra datos previos de otros tipos — solo actualiza lo que tiene.
        """
        path = os.path.join(self.data_root, self.asset, 'best.json')

        # Leer best.json existente (no pisar lo que ya había)
        config = {}
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    config = json.load(f)
            except Exception:
                pass

        ahora = datetime.now().isoformat()

        # ── Métricas reales por tipo ───────────────
        for tipo in ['scalper', 'swing', 'estrategica']:
            if self.resultados is None or self.resultados.empty:
                trades_tipo = pd.DataFrame()
            else:
                trades_tipo = self.resultados[self.resultados['tipo'] == tipo]

            # Inicializar bloque si no existe
            if tipo not in config:
                config[tipo] = {}

            config[tipo]['fecha_actualizacion'] = ahora

            if trades_tipo.empty:
                config[tipo]['performance'] = {
                    'trades':       0,
                    'win_rate':     None,
                    'ganancia_usd': 0,
                    'nota':         'sin trades en este período',
                }
            else:
                config[tipo]['performance'] = {
                    'trades':           int(len(trades_tipo)),
                    'win_rate':         round(float(trades_tipo['exito'].mean() * 100), 1),
                    'ganancia_usd':     round(float(trades_tipo['ganancia_usd'].sum()), 2),
                    'drawdown_prom':    round(float(trades_tipo['drawdown_max'].mean()), 2),
                    'duracion_prom_h':  round(float(trades_tipo['duracion_horas'].mean()), 1),
                    'pct_target':       round(float(trades_tipo['target_alcanzado'].mean() * 100), 1),
                    'pct_timeout':      round(float((trades_tipo['resultado'] == 'timeout').mean() * 100), 1),
                }

        # ── Umbrales optimizados (solo si se encontraron) ─────────
        if umbrales_opt:
            for tipo, (itc, spread) in umbrales_opt.items():
                if tipo not in config:
                    config[tipo] = {}
                config[tipo]['itc_threshold'] = int(itc)
                config[tipo]['spread_min']    = float(spread)
                config[tipo]['dias_validez']  = 2

        # ── Metadata global ────────────────────────
        span_h = (self.df['timestamp'].max() - self.df['timestamp'].min()).total_seconds() / 3600
        config['_meta'] = {
            'generado':          ahora,
            'historia_horas':    round(span_h, 1),
            'historia_dias':     round(span_h / 24, 1),
            'ciclos':            int(len(self.df)),
            'desde':             self.df['timestamp'].min().isoformat(),
            'hasta':             self.df['timestamp'].max().isoformat(),
            'scalper_totales':   self.scalper_totales,
            'scalper_filtrados': self.scalper_filtrados_btc,
        }

        # ── Escritura atómica ──────────────────────
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        os.replace(tmp, path)
        print(f"✅ best.json actualizado → {path}")
        for tipo in ['scalper', 'swing', 'estrategica']:
            perf = config.get(tipo, {}).get('performance', {})
            if perf.get('trades', 0) > 0:
                print(f"   {tipo:12}: {perf['trades']} trades | WR {perf['win_rate']}% | ${perf['ganancia_usd']} USD")
            else:
                print(f"   {tipo:12}: sin trades")

    def exportar_configuracion_optima(self, resultados_opt, tipo, path_base):
        """Wrapper de compatibilidad — llama a actualizar_best_json con el umbral de un tipo."""
        if resultados_opt is None:
            return
        self.actualizar_best_json(umbrales_opt={tipo: resultados_opt})
        print(f"✅ best.json actualizado para {tipo}: ITC={resultados_opt[0]}, Spread={resultados_opt[1]}")

    def generar_reporte(self):
        if self.resultados is None:
            print("⚠️ Primero ejecuta el backtest.")
            return

        lineas = []
        def _p(*args):
            texto = ' '.join(str(a) for a in args)
            print(texto)
            lineas.append(texto)

        _p("\n" + "="*70)
        _p(" 🦅 REPORTE MAESTRO DE BACKTESTING")
        _p(f" Generado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        span_h_rep = (self.df['timestamp'].max() - self.df['timestamp'].min()).total_seconds() / 3600
        _p(f" Historia: {span_h_rep:.1f}h ({span_h_rep/24:.1f} días) — {len(self.df)} ciclos")
        _p("="*70)

        if self.resultados.empty:
            _p("\n❌ 0 operaciones simuladas.")
            _p("   Causas posibles:")
            _p("   1. signals.csv tiene muy poco historial (< timeout máximo)")
            _p("   2. Ninguna señal supera los umbrales configurados")
            _p("   3. p_v o vwap_c tienen valores nulos o cero")
        else:
            for tipo in ['scalper', 'swing', 'estrategica']:
                df_tipo = self.resultados[self.resultados['tipo'] == tipo]
                if df_tipo.empty:
                    _p(f"\n--- {tipo.upper()} --- Sin operaciones.")
                    continue
                _p(f"\n--- {tipo.upper()} ---")
                _p(f"Operaciones:             {len(df_tipo)}")
                _p(f"Win Rate:                {df_tipo['exito'].mean()*100:.1f}%")
                _p(f"Ganancia total:          ${df_tipo['ganancia_usd'].sum():,.2f} USD")
                _p(f"Drawdown prom:           {df_tipo['drawdown_max'].mean():.2f}%")
                _p(f"Duración prom:           {df_tipo['duracion_horas'].mean():.1f}h")
                _p(f"Target / Stop / Timeout: "
                   f"{df_tipo['target_alcanzado'].mean()*100:.1f}% / "
                   f"{df_tipo['stop_alcanzado'].mean()*100:.1f}% / "
                   f"{(df_tipo['resultado']=='timeout').mean()*100:.1f}%")
                df_dur = self.analizar_por_duracion(tipo=tipo)
                if df_dur is not None and not df_dur.empty:
                    _p("\n  Por Duración:")
                    _p(df_dur.to_string(index=False))

            if 'cluster_vars' in self.cfg:
                for var in self.cfg['cluster_vars']:
                    if var not in self.resultados.columns:
                        _p(f"\n--- {var} --- (columna no encontrada en resultados)")
                        continue
                    df_filtrado = (self.resultados.dropna(subset=[var])
                                   if var.startswith('ctx_') else self.resultados)
                    if df_filtrado.empty:
                        _p(f"\n--- {var} --- (sin datos)")
                        continue
                    _p(f"\n--- Análisis por {var} ---")
                    if pd.api.types.is_numeric_dtype(df_filtrado[var]):
                        df_cl = self.analizar_por_cluster(var, bins=self.cfg.get('bins', 4),
                                                          df=df_filtrado)
                    else:
                        df_cl = self.analizar_por_cluster(var, df=df_filtrado)
                    if df_cl is not None and not df_cl.empty:
                        _p(df_cl.to_string(index=False))

            if 'factor_total' in self.df.columns:
                _p("\n--- Factor de Riesgo ---")
                for tipo in ['scalper', 'swing', 'estrategica']:
                    comp = self.comparar_con_factor_riesgo('factor_total', tipo=tipo)
                    if comp is not None:
                        _p(f"\n  {tipo.upper()}:")
                        _p(comp.to_string(index=False))

        # Siempre imprimir filtros
        _p("\n--- Filtros BTC ---")
        _p(f"Scalper totales:  {self.scalper_totales}")
        _p(f"Filtrados (BTC):  {self.scalper_filtrados_btc}")
        df_sc  = (self.resultados[self.resultados['tipo'] == 'scalper']
                  if not self.resultados.empty else pd.DataFrame())
        monto_prom = float(df_sc['monto_usd'].mean()) if not df_sc.empty else 0.0
        if pd.isna(monto_prom):
            monto_prom = 0.0
        _p(f"Capital liberado: ${self.scalper_filtrados_btc * monto_prom:,.2f} USD")
        _p("\n" + "="*70)

        # Guardar reporte (escritura atómica)
        reporte_path = os.path.join(self.data_root, self.asset, 'reporte_backtest.txt')
        tmp_path     = reporte_path + '.tmp'
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lineas))
            os.replace(tmp_path, reporte_path)
            print(f"📄 reporte_backtest.txt guardado → {reporte_path}")
        except Exception as e:
            print(f"⚠️ No se pudo guardar el reporte: {e}")


# ── Punto de entrada ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config import CONFIG

    signals_path = os.path.join(CONFIG['DATA_ROOT'], CONFIG['asset'], 'signals.csv')
    if not os.path.exists(signals_path):
        print(f"❌ No se encuentra signals.csv en: {signals_path}")
        exit(1)

    df = pd.read_csv(signals_path, parse_dates=['timestamp'])
    print(f"📂 signals.csv cargado: {len(df)} filas")

    cfg_bt = {
        'slippage_sobre_vwap': False,   # VWAP ya modela ejecución real
        'slippage_pct':        0.05,
        'targets':   {'scalper': 0.2,  'swing': 0.4,   'estrategica': 0.8},
        'stops':     {'scalper': 0.5,  'swing': 1.0,   'estrategica': 2.0},
        'timeouts':  {'scalper': 240,  'swing': 10080, 'estrategica': 43200},
        'precio_entrada':        'vwap_c',
        'precio_salida':         'p_v',
        'col_señal_scalper':     'scalper_usd',
        'col_señal_swing':       'swing_usd',
        'col_señal_estrategica': 'estrategica_usd',

        'cluster_vars': [
            'hora_entrada',
            'dia_semana_entrada',
            'ctx_dxy',
            'ctx_btc_change_24h',
            'ctx_fear_greed',
        ],
        'bins_duracion':  [0, 4, 12, 24, float('inf')],
        'labels_duracion':['0-4h', '4-12h', '12-24h', '24h+'],
        'bins': 4,
    }

    engine = BacktestEngine(df, cfg_bt)

    ok = engine.diagnosticar_datos()
    if not ok:
        exit(1)

    engine.ejecutar_backtest()

    engine.guardar_resultados()

    engine.generar_reporte()

    umbrales_encontrados = {}

    if engine.resultados is not None and not engine.resultados.empty:
        rango_itc    = list(range(40, 81, 5))
        rango_spread = [0.1, 0.15, 0.2, 0.25]

        print("\n🔍 Optimizando SCALPER...")
        _, mejor_scalper = engine.optimizar_umbrales(
            rango_itc, rango_spread, tipo='scalper', metric='win_rate')
        if mejor_scalper:
            print(f"   Mejor SCALPER: ITC={mejor_scalper[0]}, Spread={mejor_scalper[1]}")
            umbrales_encontrados['scalper'] = mejor_scalper

        print("\n🔍 Optimizando SWING...")
        _, mejor_swing = engine.optimizar_umbrales(
            rango_itc, rango_spread, tipo='swing', metric='win_rate')
        if mejor_swing:
            print(f"   Mejor SWING: ITC={mejor_swing[0]}, Spread={mejor_swing[1]}")
            umbrales_encontrados['swing'] = mejor_swing

        print("\n🔍 Optimizando ESTRATÉGICA...")
        _, mejor_estrategica = engine.optimizar_umbrales(
            rango_itc, rango_spread, tipo='estrategica', metric='ganancia_total')
        if mejor_estrategica:
            print(f"   Mejor ESTRATÉGICA: ITC={mejor_estrategica[0]}, Spread={mejor_estrategica[1]}")
            umbrales_encontrados['estrategica'] = mejor_estrategica
    else:
        print("\n⚠️ Sin trades simulados — best.json se actualiza con métricas vacías.")
        print("   Necesitás más historial en signals.csv.")

    engine.actualizar_best_json(
        umbrales_opt=umbrales_encontrados if umbrales_encontrados else None
    )
