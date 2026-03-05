import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from itertools import product
from config import CONFIG  

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

        # Filtro de BTC para scalper (evitar caídas > 1%)
        if tipo == 'scalper':
            btc_change = row.get('ctx_btc_change_24h', 0)
            if btc_change <= -1.0:
                return None

        target_pct = self.cfg['targets'][tipo] / 100
        stop_pct = self.cfg['stops'][tipo] / 100
        timeout_min = self.cfg['timeouts'][tipo]
        slippage = self.cfg.get('slippage_pct', 0) / 100

        precio_entrada = row[self.cfg['precio_entrada']] * (1 + slippage)
        target_price = precio_entrada * (1 + target_pct)
        stop_price = precio_entrada * (1 - stop_pct)

        future_window = self._get_future_window(row['timestamp'], timeout_min)
        if len(future_window) <= 1:
            return None

        future_window = future_window.iloc[1:]
        precio_salida = None
        tiempo_salida = None
        drawdown_max = 0
        target_alcanzado = False
        stop_alcanzado = False
        precio_min = precio_entrada

        for _, fut_row in future_window.iterrows():
            precio_venta = fut_row['p_v'] * (1 - slippage)
            if precio_venta < precio_min:
                precio_min = precio_venta
                drawdown_max = max(drawdown_max, (precio_entrada - precio_min) / precio_entrada * 100)

            if precio_venta <= stop_price:
                stop_alcanzado = True
                precio_salida = precio_venta
                tiempo_salida = (fut_row['timestamp'] - row['timestamp']).total_seconds() / 60
                break
            if precio_venta >= target_price:
                target_alcanzado = True
                precio_salida = precio_venta
                tiempo_salida = (fut_row['timestamp'] - row['timestamp']).total_seconds() / 60
                break

        if precio_salida is None:
            ultimo_precio = future_window.iloc[-1]['p_v'] * (1 - slippage)
            precio_salida = ultimo_precio
            tiempo_salida = timeout_min

        retorno_pct = (precio_salida / precio_entrada - 1) * 100
        ganancia_usd = monto * (retorno_pct / 100)
        exito = target_alcanzado

        fecha_salida = row['timestamp'] + timedelta(minutes=tiempo_salida)

        resultado = {
            'timestamp': row['timestamp'],
            'tipo': tipo,
            'monto_usd': monto,
            'resultado': 'target' if target_alcanzado else ('stop' if stop_alcanzado else 'timeout'),
            'precio_entrada': round(precio_entrada, 2),
            'precio_salida': round(precio_salida, 2),
            'retorno_pct': round(retorno_pct, 2),
            'ganancia_usd': round(ganancia_usd, 2),
            'tiempo_min': round(tiempo_salida, 1),
            'duracion_horas': round(tiempo_salida / 60, 1),
            'drawdown_max': round(drawdown_max, 2),
            'target_alcanzado': target_alcanzado,
            'stop_alcanzado': stop_alcanzado,
            'exito': exito,
            'hora_entrada': row['timestamp'].hour,
            'dia_semana_entrada': row['timestamp'].weekday(),
            'hora_salida': fecha_salida.hour,
            'dia_semana_salida': fecha_salida.weekday()
        }
        for var in self.cfg.get('cluster_vars', []):
            if var in row:
                resultado[var] = row[var]
        return resultado

    def ejecutar_backtest(self):
        resultados = []
        total = len(self.df)
        print(f"Ejecutando backtest sobre {total} registros...")
        self.scalper_filtrados_btc = 0
        self.scalper_totales = 0

        for idx, row in self.df.iterrows():
            if idx % 1000 == 0:
                print(f"Procesando {idx}/{total}")

            # Contar señales de scalper
            if row.get('scalper_usd', 0) > 0:
                self.scalper_totales += 1
                # Aplicar filtro BTC (si no pasa, no se simula)
                if row.get('ctx_btc_change_24h', 0) <= -1.0:
                    self.scalper_filtrados_btc += 1
                    continue  # Saltamos esta señal

            for tipo in ['scalper', 'swing', 'estrategica']:
                res = self.simular_operacion(row, tipo)
                if res:
                    resultados.append(res)

        self.resultados = pd.DataFrame(resultados)
        print(f"Backtest completado. {len(self.resultados)} operaciones simuladas.")
        print(f"Señales de scalper totales: {self.scalper_totales}, filtradas por BTC: {self.scalper_filtrados_btc}")
        return self.resultados
    
    def analizar_por_cluster(self, columna, bins=None, df=None):
        if df is None:
            if self.resultados is None:
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
            win_rate = g['exito'].mean() * 100
            total_ops = len(g)
            ganancia = g['ganancia_usd'].sum()
            drawdown_prom = g['drawdown_max'].mean()
            tiempo_prom = g['duracion_horas'].mean()
            stats.append({
                'cluster': nombre,
                'operaciones': total_ops,
                'win_rate': round(win_rate, 1),
                'ganancia_usd': round(ganancia, 2),
                'drawdown_prom': round(drawdown_prom, 2),
                'duracion_prom_hs': round(tiempo_prom, 1)
            })
        return pd.DataFrame(stats)

    def analizar_por_duracion(self, tipo=None, bins=None, labels=None):
        if self.resultados is None:
            return None
        df = self.resultados.copy()
        if tipo:
            df = df[df['tipo'] == tipo].copy()
            if len(df) == 0:
                return None
        bins = bins or self.cfg.get('bins_duracion', [0, 4, 12, 24, float('inf')])
        labels = labels or self.cfg.get('labels_duracion', ['0-4h', '4-12h', '12-24h', '24h+'])
        df['rango_duracion'] = pd.cut(df['duracion_horas'], bins=bins, labels=labels, right=False)
        res = df.groupby('rango_duracion', observed=False).agg(
            operaciones=('exito', 'count'),
            win_rate=('exito', lambda x: round(x.mean()*100, 1)),
            ganancia_total=('ganancia_usd', 'sum'),
            drawdown_prom=('drawdown_max', 'mean'),
            duracion_prom=('duracion_horas', 'mean')
        ).reset_index()
        return res

    def comparar_con_factor_riesgo(self, col_factor='factor_total', tipo=None):
        """
        Compara el desempeño de las operaciones según el valor del factor de riesgo.
        Útil para validar si el factor realmente mejora los resultados.
        """
        if self.resultados is None:
            return None

        # Necesitamos unir los resultados con el factor de riesgo original de cada fila
        # Como el factor está en el df original, lo agregamos a resultados
        df_res = self.resultados.copy()
        if tipo:
            df_res = df_res[df_res['tipo'] == tipo].copy()
            if len(df_res) == 0:
                return None

        df_original = self.df.set_index('timestamp')[col_factor].to_dict()
        df_res['factor'] = df_res['timestamp'].map(df_original)

        if df_res['factor'].isna().all():
            print("No se encontró la columna de factor en los datos.")
            return None

        mediana = df_res['factor'].median()
        alto = df_res[df_res['factor'] >= mediana]
        bajo = df_res[df_res['factor'] < mediana]

        comparacion = pd.DataFrame({
            'grupo': ['Factor >= mediana', 'Factor < mediana'],
            'operaciones': [len(alto), len(bajo)],
            'win_rate': [alto['exito'].mean()*100, bajo['exito'].mean()*100],
            'ganancia_total': [alto['ganancia_usd'].sum(), bajo['ganancia_usd'].sum()],
            'drawdown_prom': [alto['drawdown_max'].mean(), bajo['drawdown_max'].mean()]
        })
        return comparacion

    def optimizar_umbrales(self, rango_itc, rango_spread, tipo='scalper', metric='win_rate'):
        resultados = []
        for itc, spread in product(rango_itc, rango_spread):
            mask = (self.df['itc_score'] <= itc) & (self.df['spread_actual'] >= spread)
            df_filt = self.df[mask].copy()
            if len(df_filt) < 10:
                continue
            engine_temp = BacktestEngine(df_filt, self.cfg)
            engine_temp.ejecutar_backtest()
            res = engine_temp.resultados
            if res is None or len(res) == 0:
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
                'itc': itc,
                'spread': spread,
                'operaciones': len(res),
                metric: round(valor, 2),
                'win_rate': round(res['exito'].mean() * 100, 1),
                'ganancia_total': round(res['ganancia_usd'].sum(), 2)
            })
        df_res = pd.DataFrame(resultados)
        if len(df_res) == 0:
            return df_res, None
        mejor_idx = df_res[metric].idxmax()
        mejor_combo = (df_res.loc[mejor_idx, 'itc'], df_res.loc[mejor_idx, 'spread'])
        return df_res, mejor_combo

    def exportar_configuracion_optima(self, resultados_opt, tipo, path_base):
        if resultados_opt is None:
            return
        path = os.path.join(path_base, self.asset, 'best.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        config[tipo] = {
            'itc_threshold': int(resultados_opt[0]),
            'spread_min': float(resultados_opt[1]),
            'fecha_actualizacion': datetime.now().isoformat(),
            'dias_validez': 2
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"✅ Configuración para {tipo} guardada en {path}")

    def generar_reporte(self):
        if self.resultados is None:
            print("Primero ejecuta el backtest.")
            return
        print("\n" + "="*70)
        print(" 🦅 REPORTE MAESTRO DE BACKTESTING")
        print("="*70)
        for tipo in ['scalper', 'swing', 'estrategica']:
            mask = self.resultados['tipo'] == tipo
            df_tipo = self.resultados[mask]
            if len(df_tipo) == 0:
                print(f"\n--- {tipo.upper()} --- Sin operaciones.")
                continue
            win_rate = df_tipo['exito'].mean() * 100
            ganancia = df_tipo['ganancia_usd'].sum()
            drawdown = df_tipo['drawdown_max'].mean()
            tiempo = df_tipo['duracion_horas'].mean()
            target_rate = df_tipo['target_alcanzado'].mean() * 100
            stop_rate = df_tipo['stop_alcanzado'].mean() * 100
            timeout_rate = (df_tipo['resultado'] == 'timeout').mean() * 100
            print(f"\n--- {tipo.upper()} ---")
            print(f"Operaciones: {len(df_tipo)}")
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Ganancia total: ${ganancia:,.2f} USD")
            print(f"Drawdown máximo promedio: {drawdown:.2f}%")
            print(f"Duración promedio: {tiempo:.1f} horas")
            print(f"Alcanzaron target: {target_rate:.1f}% | Stop: {stop_rate:.1f}% | Timeout: {timeout_rate:.1f}%")
            print("\n  Análisis por Duración:")
            df_duracion = self.analizar_por_duracion(tipo=tipo)
            if df_duracion is not None and len(df_duracion) > 0:
                print(df_duracion.to_string(index=False))

        # Análisis por clusters
        if 'cluster_vars' in self.cfg:
            for var in self.cfg['cluster_vars']:
                if var in self.resultados.columns:
                    print(f"\n--- Análisis por {var} ---")
                    # Filtrar filas con NaN si la variable es de contexto
                    if var.startswith('ctx_'):
                        df_filtrado = self.resultados.dropna(subset=[var]).copy()
                        if df_filtrado.empty:
                            print("   (sin datos con contexto)")
                            continue
                    else:
                        df_filtrado = self.resultados
        
                    if pd.api.types.is_numeric_dtype(df_filtrado[var]):
                        bins = self.cfg.get('bins', 4)
                        # Asegurar que haya al menos 2 puntos para crear bins
                        if len(df_filtrado) < 2:
                            print("   (insuficientes datos para agrupar)")
                            continue
                        df_cluster = self.analizar_por_cluster(var, bins=bins, df=df_filtrado)
                    else:
                        df_cluster = self.analizar_por_cluster(var, df=df_filtrado)
                    if df_cluster is not None and not df_cluster.empty:
                        print(df_cluster.to_string(index=False))

        # Comparación por factor de riesgo
        if 'factor_total' in self.df.columns:
            print("\n--- Comparación por Factor de Riesgo ---")
            for tipo in ['scalper', 'swing', 'estrategica']:
                comp = self.comparar_con_factor_riesgo('factor_total', tipo=tipo)
                if comp is not None:
                    print(f"\n  {tipo.upper()}:")
                    print(comp.to_string(index=False))

        # Nuevo: reporte de filtros aplicados
        if hasattr(self, 'scalper_filtrados_btc'):
            print("\n--- Filtros aplicados ---")
            print(f"Señales de scalper totales: {self.scalper_totales}")
            print(f"Señales de scalper filtradas por caída de BTC >1%: {self.scalper_filtrados_btc}")
            # Estimación de capital liberado (suponiendo monto promedio de scalper)
            monto_promedio_scalper = self.resultados[self.resultados['tipo']=='scalper']['monto_usd'].mean() if not self.resultados.empty else 20000*0.5
            capital_liberado = self.scalper_filtrados_btc * monto_promedio_scalper
            print(f"Capital liberado estimado: ${capital_liberado:,.2f} USD")

        print("\n" + "="*70)

if __name__ == "__main__":
    from config import CONFIG

    signals_path = os.path.join(CONFIG['DATA_ROOT'], CONFIG['asset'], 'signals.csv')
    if not os.path.exists(signals_path):
        print(f"❌ No se encuentra {signals_path}")
        exit()

    df = pd.read_csv(signals_path, parse_dates=['timestamp'])

    cfg_bt = {
        'slippage_pct': 0.05,
        'targets': {'scalper': 0.2, 'swing': 0.4, 'estrategica': 0.8},
        'stops': {'scalper': 0.5, 'swing': 1.0, 'estrategica': 2.0},
        'timeouts': {'scalper': 240, 'swing': 10080, 'estrategica': 43200},  # scalper 4h
        'precio_entrada': 'vwap_c',
        'precio_salida': 'p_v',
        'col_señal_scalper': 'scalper_usd',
        'col_señal_swing': 'swing_usd',
        'col_señal_estrategica': 'estrategica_usd',
        'cluster_vars': ['hora_entrada','dia_semana_entrada','ctx_dxy','ctx_btc_change_24h','ctx_fear_greed'],
        'bins_duracion': [0, 4, 12, 24, float('inf')],
        'labels_duracion': ['0-4h', '4-12h', '12-24h', '24h+'],
        'bins': 4
    }

    engine = BacktestEngine(df, cfg_bt)
    engine.ejecutar_backtest()
    engine.generar_reporte()

    # Optimizar umbrales para scalper
    rango_itc = list(range(40, 81, 5))
    rango_spread = [0.1, 0.15, 0.2, 0.25]
    df_opt, mejor_scalper = engine.optimizar_umbrales(rango_itc, rango_spread, tipo='scalper', metric='win_rate')
    if mejor_scalper:
        print("\nMejor combinación para SCALPER:", mejor_scalper)
        engine.exportar_configuracion_optima(mejor_scalper, 'scalper', CONFIG['DATA_ROOT'])

    # Optimizar para swing
    df_opt_swing, mejor_swing = engine.optimizar_umbrales(rango_itc, rango_spread, tipo='swing', metric='win_rate')
    if mejor_swing:
        print("Mejor combinación para SWING:", mejor_swing)
        engine.exportar_configuracion_optima(mejor_swing, 'swing', CONFIG['DATA_ROOT'])
