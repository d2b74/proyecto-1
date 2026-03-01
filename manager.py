import pandas as pd
import glob
import os
from datetime import datetime

class DataConsolidator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.asset = cfg.get('asset', 'USDT').upper()
        self.data_root = cfg['DATA_ROOT']
        self.raw_path = cfg['path_datasets']          
        self.base_path = os.path.join(self.data_root, self.asset)
        self.master_file = os.path.join(self.base_path, 'master.parquet') 
        self.control_file = os.path.join(self.base_path, '.last')

        os.makedirs(self.base_path, exist_ok=True)

    def _get_last_processed(self):
        if not os.path.exists(self.control_file):
            return None
        with open(self.control_file, 'r') as f:
            return f.read().strip()

    def _update_last_processed(self, filepath):
        rel_path = os.path.relpath(filepath, self.raw_path)
        with open(self.control_file, 'w') as f:
            f.write(rel_path)

    def _get_new_files(self):
        # Búsqueda recursiva en todas las subcarpetas de raw
        pattern = os.path.join(self.raw_path, '**', 'p2p_*.parquet')
        all_files = sorted(glob.glob(pattern, recursive=True))
        last = self._get_last_processed()
        if last is None:
            return all_files
        last_abs = os.path.join(self.raw_path, last)
        return [f for f in all_files if f > last_abs]

    def run_update(self, force_all=False):
        print(f"🔍 [DataConsolidator] Buscando actualizaciones para {self.asset}...")

        if force_all:
            pattern = os.path.join(self.raw_path, '**', 'p2p_*.parquet')
            files_to_process = sorted(glob.glob(pattern, recursive=True))
        else:
            files_to_process = self._get_new_files()

        if not files_to_process:
            print("✅ No hay archivos nuevos.")
            return

        new_data = []
        for f in files_to_process:
            try:
                df_temp = pd.read_parquet(f)
            except Exception as e:
                print(f"⚠️ Error leyendo {f}: {e}")
                continue

            if 'moneda' in df_temp.columns:
                df_temp = df_temp[df_temp['moneda'] == self.asset].copy()
            else:
                print(f"⚠️ Archivo {f} no tiene columna 'moneda', se salta.")
                continue

            if df_temp.empty:
                continue

            if df_temp['timestamp'].dtype == 'object':
                df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])

            if 'month_finish_rate' in df_temp.columns:
                min_rate = self.cfg.get('min_rate_confianza', 0.95)
                df_temp = df_temp[df_temp['month_finish_rate'] >= min_rate].copy()
            else:
                print(f"⚠️ Archivo {f} no tiene 'month_finish_rate', se omite filtro de calidad.")

            if df_temp.empty:
                continue

            if 'mep_venta' in df_temp.columns:
                df_temp['ratio_p2p_mep'] = df_temp['precio'] / df_temp['mep_venta']

            df_temp['hora'] = df_temp['timestamp'].dt.hour
            df_temp['dia_semana'] = df_temp['timestamp'].dt.dayofweek

            new_data.append(df_temp)

        if not new_data:
            print("✅ No se encontraron datos nuevos después de filtrar.")
            return

        df_new = pd.concat(new_data, ignore_index=True)

        if os.path.exists(self.master_file):
            df_existing = pd.read_parquet(self.master_file)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            print(f"📥 Se agregaron {len(df_new)} nuevos registros (total: {len(df_final)}).")
        else:
            df_final = df_new
            print(f"✨ Memoria creada con {len(df_final)} registros.")

        df_final.to_parquet(self.master_file, compression='snappy', index=False)

        if files_to_process:
            self._update_last_processed(files_to_process[-1])

        print(f"✅ Consolidación completada. Master en: {self.master_file}")
