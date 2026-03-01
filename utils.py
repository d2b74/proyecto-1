import os
import pandas as pd

def obtener_ventana_fresca(base_path, asset, horas_atras=2):

    master_file = os.path.join(base_path, asset, 'master.parquet')
    if not os.path.exists(master_file):
        print(f"❌ No se encontró master.parquet en {master_file}")
        return None

    # Leer todo el master (es incremental, pero para ventanas pequeñas es rápido)
    df = pd.read_parquet(master_file)

    # Asegurar timestamp como datetime
    if df['timestamp'].dtype == 'object':
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filtrar por ventana de tiempo
    ts_max = df['timestamp'].max()
    ventana = df[df['timestamp'] >= (ts_max - pd.Timedelta(hours=horas_atras))].copy()

    print(f"✅ Ventana obtenida ({horas_atras}h): {len(ventana)} registros (Desde: {ventana['timestamp'].min().strftime('%d/%m %H:%M')})")
    return ventana
