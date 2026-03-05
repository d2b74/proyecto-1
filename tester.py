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
    COLUMNAS_BASE = []  # fallback si se ejecuta en otro entorno


class BacktestEngine:

    def __init__(self, df, cfg):
        self.cfg = cfg
        self.data_root = CONFIG['DATA_ROOT']
        self.asset = CONFIG['asset']
        self.df = df.copy()
        self.df.sort_values('timestamp', inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.resultados = None
        self.scalper_filtrados_btc = 0
        self.scalper_totales = 0

    def _get_future_window(self, start_ts, max_minutes):
        end_ts = start_ts + timedelta(minutes=max_minutes)
        mask = (self.df['timestamp'] >= start_ts) & (self.df['timestamp'] <= end_ts)
        return self.df.loc[mask].copy()

    def simular_operacion(self, row, tipo):
        col_señal = self.cfg[f'col_señal_{tipo}']
        monto = row.get(col_señal, 0)
        if monto <= 0:
            return None

        if tipo == 'scalper':
            btc_change = row.get('ctx_btc_change_24h', 0)
            if pd.isna(btc_change):
                btc_change = 0
            if btc_change <= -1.0:
                return None

        target_pct  = self.cfg['targets'][tipo] / 100
        stop_pct    = self.cfg['stops'][tipo]   / 100
        timeout_min = self.cfg['timeouts'][tipo]


        if self.cfg.get('slippage_sobre_vwap', False):
            slippage = self.cfg.get('slippage_pct', 0) / 100
        else:
            slippage = 0.0

        precio_entrada = row[self.cfg['precio_entrada']] * (1 + slippage)
        target_price   = precio_entrada * (1 + target_pct)
        stop_price     = precio_entrada * (1 - stop_pct)

        future_window = self._get_future_window(row['timestamp'], timeout_min)
        if len(future_window) <= 1:
            return None

        future_window = future_window.iloc[1:]
        precio_salida     = None
        tiempo_salida     = None
        drawdown_max      = 0
        target_alcanzado  = False
        stop_alcanzado    = False
        precio_min        = precio_entrada

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
            'timestamp':         row['timestamp'],
            'tipo':              tipo,
            'monto_usd':         monto,
            'resultado':         'target' if target_alcanzado else ('stop' if stop_alcanzado else 'timeout'),
            'precio_entrada':    round(precio_entrada, 2),
            'precio_salida':     round(precio_salida,  2),
            'retorno_pct':       round(retorno_pct,    2),
            'ganancia_usd':      round(ganancia_usd,   2),
            'tiempo_min':        round(tiempo_salida,  1),
            'duracion_horas':    round(tiempo_salida / 60, 1),
            'drawdown_max':      round(drawdown_max,   2),
            'target_alcanzado':  target_alcanzado,
            'stop_alcanzado':    stop_alcanzado,
            'exito':             target_alcanzado,
            'hora_entrada':      row['timestamp'].hour,
            'dia_semana_entrada':row['timestamp'].weekday(),
            'hora_salida':       fecha_salida.hour,
            'dia_semana_salida': fecha_salida.weekday(),
        }

        for var in self.cfg.get('cluster_vars', []):
            if var in row:
                val = row[var]
                resultado[var] = val if not pd.isna(val) else None

        return resultado

    def ejecutar_backtest(self):
        resultados = []
        total = len(self.df)
        print(f"Ejecutando backtest sobre {total} registros...")
        self.scalper_filtrados_btc = 0
        self.scalper_totales = 0

        for idx, row in self.df.iterrows():
            if idx % 1000 == 0:
                print(f"  Procesando {idx}/{total}...")

            if row.get('scalper_usd', 0) > 0:
                self.scalper_totales += 1
                btc_change = row.get('ctx_btc_change_24h', 0)
                if pd.isna(btc_change):
                    btc_change = 0
                if btc_change <= -1.0:
                    self.scalper_filtrados_btc += 1
                    # Registrado. simular_operacion() aplicará el mismo filtro
                    # y devolverá None para el scalper de esta fila.

            for tipo in ['scalper', 'swing', 'estrategica']:
                res = self.simular_operacion(row, tipo)
                if res:
                    resultados.append(res)

        self.resultados = pd.DataFrame(resultados) if resultados else pd.DataFrame()
        print(f"Backtest completado. {len(self.resultados)} operaciones simuladas.")
        print(f"Scalper totales: {self.scalper_totales} | Filtradas por BTC: {self.scalper_filtrados_btc}")
        return self.resultados

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
        df = self.resultados.copy()
        if tipo:
            df = df[df['tipo'] == tipo].copy()
            if len(df) == 0:
                return None
        bins   = bins   or self.cfg.get('bins_duracion',  [0, 4, 12, 24, float('inf')])
        labels = labels or self.cfg.get('labels_duracion', ['0-4h', '4-12h', '12-24h', '24h+'])
        df['rango_duracion'] = pd.cut(df['duracion_horas'], bins=bins, labels=labels, right=False)
        res = df.groupby('rango_duracion', observed=False).agg(
            operaciones=('exito', 'count'),
            win_rate=('exito', lambda x: round(x.mean() * 100, 1)),
            ganancia_total=('ganancia_usd', 'sum'),
            drawdown_prom=('drawdown_max', 'mean'),
            duracion_prom=('duracion_horas', 'mean')
        ).reset_index()
        return res

    def comparar_con_factor_riesgo(self, col_factor='factor_total', tipo=None):
        if self.resultados is None or self.resultados.empty:
            return None
        df_res = self.resultados.copy()
        if tipo:
            df_res = df_res[df_res['tipo'] == tipo].copy()
            if len(df_res) == 0:
                return None
        df_original = self.df.set_index('timestamp')[col_factor].to_dict()
        df_res['factor'] = df_res['timestamp'].map(df_original)
        if df_res['factor'].isna().all():
            print(f"No se encontró '{col_factor}' en los datos originales.")
            return None
        mediana = df_res['factor'].median()
        alto = df_res[df_res['factor'] >= mediana]
        bajo = df_res[df_res['factor'] < mediana]
        return pd.DataFrame({
            'grupo':          ['Factor >= mediana', 'Factor < mediana'],
            'operaciones':    [len(alto), len(bajo)],
            'win_rate':       [round(alto['exito'].mean() * 100, 1), round(bajo['exito'].mean() * 100, 1)],
            'ganancia_total': [round(alto['ganancia_usd'].sum(), 2), round(bajo['ganancia_usd'].sum(), 2)],
            'drawdown_prom':  [round(alto['drawdown_max'].mean(), 2), round(bajo['drawdown_max'].mean(), 2)],
        })

    def optimizar_umbrales(self, rango_itc, rango_spread, tipo='scalper', metric='win_rate'):
        resultados = []
        for itc, spread in product(rango_itc, rango_spread):
            mask   = (self.df['itc_score'] <= itc) & (self.df['spread_actual'] >= spread)
            df_filt = self.df[mask].copy()
            if len(df_filt) < 10:
                continue
            engine_temp = BacktestEngine(df_filt, self.cfg)
            engine_temp.ejecutar_backtest()
            res = engine_temp.resultados
            if res is None or res.empty:
                continue
            res = res[res['tipo'] == tipo]
            if len(res) == 0:
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
                'itc':           itc,
                'spread':        spread,
                'operaciones':   len(res),
                metric:          round(valor, 2),
                'win_rate':      round(res['exito'].mean() * 100, 1),
                'ganancia_total':round(res['ganancia_usd'].sum(), 2),
            })
        df_res = pd.DataFrame(resultados)
        if df_res.empty:
            return df_res, None
        mejor_idx   = df_res[metric].idxmax()
        mejor_combo = (df_res.loc[mejor_idx, 'itc'], df_res.loc[mejor_idx, 'spread'])
        return df_res, mejor_combo

    def exportar_configuracion_optima(self, resultados_opt, tipo, path_base):
        if resultados_opt is None:
            return
        path = os.path.join(path_base, self.asset, 'best.json')
        config = {}
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    config = json.load(f)
            except:
                pass
        config[tipo] = {
            'itc_threshold':       int(resultados_opt[0]),
            'spread_min':          float(resultados_opt[1]),
            'fecha_actualizacion': datetime.now().isoformat(),
            'dias_validez':        2,
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"✅ Configuración para {tipo} guardada en {path}")

    def generar_reporte(self):
        if self.resultados is None:
            print("Primero ejecuta el backtest.")
            return

        lineas = []
        def _p(*args):
            texto = ' '.join(str(a) for a in args)
            print(texto)
            lineas.append(texto)

        _p("\n" + "="*70)
        _p(" 🦅 REPORTE MAESTRO DE BACKTESTING")
        _p("="*70)

        for tipo in ['scalper', 'swing', 'estrategica']:
            df_tipo = self.resultados[self.resultados['tipo'] == tipo]
            if len(df_tipo) == 0:
                _p(f"\n--- {tipo.upper()} --- Sin operaciones.")
                continue
            _p(f"\n--- {tipo.upper()} ---")
            _p(f"Operaciones: {len(df_tipo)}")
            _p(f"Win Rate: {df_tipo['exito'].mean()*100:.1f}%")
            _p(f"Ganancia total: ${df_tipo['ganancia_usd'].sum():,.2f} USD")
            _p(f"Drawdown máximo promedio: {df_tipo['drawdown_max'].mean():.2f}%")
            _p(f"Duración promedio: {df_tipo['duracion_horas'].mean():.1f} horas")
            _p(f"Target: {df_tipo['target_alcanzado'].mean()*100:.1f}% | "
               f"Stop: {df_tipo['stop_alcanzado'].mean()*100:.1f}% | "
               f"Timeout: {(df_tipo['resultado']=='timeout').mean()*100:.1f}%")
            df_dur = self.analizar_por_duracion(tipo=tipo)
            if df_dur is not None and not df_dur.empty:
                _p("\n  Por Duración:")
                _p(df_dur.to_string(index=False))

        if 'cluster_vars' in self.cfg:
            for var in self.cfg['cluster_vars']:
                if var not in self.resultados.columns:
                    continue
                df_filtrado = self.resultados.dropna(subset=[var]) if var.startswith('ctx_') \
                              else self.resultados
                if df_filtrado.empty:
                    _p(f"\n--- {var} --- (sin datos)")
                    continue
                _p(f"\n--- Análisis por {var} ---")
                if pd.api.types.is_numeric_dtype(df_filtrado[var]):
                    df_cluster = self.analizar_por_cluster(var, bins=self.cfg.get('bins', 4),
                                                           df=df_filtrado)
                else:
                    df_cluster = self.analizar_por_cluster(var, df=df_filtrado)
                if df_cluster is not None and not df_cluster.empty:
                    _p(df_cluster.to_string(index=False))

        if 'factor_total' in self.df.columns:
            _p("\n--- Factor de Riesgo ---")
            for tipo in ['scalper', 'swing', 'estrategica']:
                comp = self.comparar_con_factor_riesgo('factor_total', tipo=tipo)
                if comp is not None:
                    _p(f"\n  {tipo.upper()}:")
                    _p(comp.to_string(index=False))

        # Reporte de filtros BTC
        _p("\n--- Filtros aplicados ---")
        _p(f"Scalper totales: {self.scalper_totales} | Filtrados por BTC: {self.scalper_filtrados_btc}")
        df_scalper = self.resultados[self.resultados['tipo'] == 'scalper'] \
                     if not self.resultados.empty else pd.DataFrame()
        if not df_scalper.empty:
            monto_prom = df_scalper['monto_usd'].mean()
        else:
            monto_prom = 0.0
        # Usar 0 si monto_prom es NaN (por ejemplo, 0 operaciones scalper ejecutadas)
        if pd.isna(monto_prom):
            monto_prom = 0.0
        capital_liberado = self.scalper_filtrados_btc * monto_prom
        _p(f"Capital liberado estimado (filtro BTC): ${capital_liberado:,.2f} USD")

        _p("\n" + "="*70)

        reporte_path = os.path.join(self.data_root, self.asset, 'reporte_backtest.txt')
        try:
            with open(reporte_path, 'w') as f:
                f.write('\n'.join(lineas))
            print(f"📄 Reporte guardado en {reporte_path}")
        except Exception as e:
            print(f"⚠️ No se pudo guardar el reporte: {e}")


if __name__ == "__main__":
    from config import CONFIG

    signals_path = os.path.join(CONFIG['DATA_ROOT'], CONFIG['asset'], 'signals.csv')
    if not os.path.exists(signals_path):
        print(f"❌ No se encuentra {signals_path}")
        exit()

    df = pd.read_csv(signals_path, parse_dates=['timestamp'])

    cfg_bt = {
        # Slippage: False cuando precio_entrada es VWAP (ya modela ejecución real)
        'slippage_sobre_vwap': False,
        'slippage_pct':   0.05,
        'targets':        {'scalper': 0.2,  'swing': 0.4,   'estrategica': 0.8},
        'stops':          {'scalper': 0.5,  'swing': 1.0,   'estrategica': 2.0},
        'timeouts':       {'scalper': 240,  'swing': 10080, 'estrategica': 43200},
        'precio_entrada': 'vwap_c',
        'precio_salida':  'p_v',
        'col_señal_scalper':    'scalper_usd',
        'col_señal_swing':      'swing_usd',
        'col_señal_estrategica':'estrategica_usd',
        # Nombres de columnas: deben coincidir exactamente con lo que escribe el Brain
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
    engine.ejecutar_backtest()
    engine.generar_reporte()

    rango_itc    = list(range(40, 81, 5))
    rango_spread = [0.1, 0.15, 0.2, 0.25]

    df_opt, mejor_scalper = engine.optimizar_umbrales(rango_itc, rango_spread,
                                                       tipo='scalper', metric='win_rate')
    if mejor_scalper:
        print(f"\nMejor SCALPER: ITC={mejor_scalper[0]}, Spread={mejor_scalper[1]}")
        engine.exportar_configuracion_optima(mejor_scalper, 'scalper', CONFIG['DATA_ROOT'])

    df_opt_swing, mejor_swing = engine.optimizar_umbrales(rango_itc, rango_spread,
                                                           tipo='swing', metric='win_rate')
    if mejor_swing:
        print(f"Mejor SWING: ITC={mejor_swing[0]}, Spread={mejor_swing[1]}")
        engine.exportar_configuracion_optima(mejor_swing, 'swing', CONFIG['DATA_ROOT'])
