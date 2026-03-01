import pandas as pd
import glob
import os
from datetime import datetime

class DataConsolidator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.asset = cfg.get('asset', 'USDT').upper()
        self.data_root = cfg['DATA_ROOT']
        self.raw_path = cfg['path_datasets']          # debe apuntar a la carpeta raw
        self.base_path = os.path.join(self.data_root, self.asset)
        self.master_file = os.path.join(self.base_path, 'master.parquet')
        self.control_file = os.path.join(self.base_path, '.last')

        os.makedirs(self.base_path, exist_ok=True)

    def _get_last_master_timestamp(self):
        """Obtiene el timestamp máximo del master actual."""
        if not os.path.exists(self.master_file):
            return None
        try:
            df = pd.read_parquet(self.master_file, columns=['timestamp'])
            return df['timestamp'].max()
        except:
            return None

    def _get_new_files_by_timestamp(self):
        """
        Busca archivos raw que contengan datos con timestamp posterior
        al último del master.
        """
        last_ts = self._get_last_master_timestamp()
        pattern = os.path.join(self.raw_path, '**', 'p2p_*.parquet')
        all_files = sorted(glob.glob(pattern, recursive=True))
        new_files = []

        for f in all_files:
            try:
                # Leer solo la columna timestamp para ser eficiente
                df_ts = pd.read_parquet(f, columns=['timestamp'])
                file_max_ts = df_ts['timestamp'].max()
                if last_ts is None or file_max_ts > last_ts:
                    new_files.append(f)
            except Exception as e:
                print(f"⚠️ Error leyendo timestamp de {f}: {e}")
                continue

        return new_files

    def run_update(self, force_all=False):
        print(f"🔍 [DataConsolidator] Buscando actualizaciones para {self.asset}...")

        if force_all:
            pattern = os.path.join(self.raw_path, '**', 'p2p_*.parquet')
            files_to_process = sorted(glob.glob(pattern, recursive=True))
        else:
            files_to_process = self._get_new_files_by_timestamp()

        print(f"📁 Archivos raw encontrados: {len(files_to_process)}")

        if not files_to_process:
            print("✅ No hay archivos nuevos.")
            return

        new_data = []
        for f in files_to_process:
            print(f"📄 Procesando {f}...")
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

        # Opcional: actualizar el archivo de control con el último archivo procesado
        if files_to_process:
            with open(self.control_file, 'w') as f:
                f.write(files_to_process[-1])

        print(f"✅ Consolidación completada. Master en: {self.master_file}")


if __name__ == "__main__":
    import sys
    from config import CONFIG
    force = '--force-all' in sys.argv
    consolidator = DataConsolidator(CONFIG)
    consolidator.run_update(force_all=force)
