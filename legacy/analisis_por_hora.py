import csv
from collections import defaultdict
from datetime import datetime
import pytz

CSV_FILE = "p2p_historial.csv"

# Zona horaria Argentina
ARG_TZ = pytz.timezone("America/Argentina/Buenos_Aires")

# datos[moneda][hora] = lista de spreads
datos = defaultdict(lambda: defaultdict(list))

with open(CSV_FILE, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        moneda = row["moneda"]

        # Construimos datetime desde fecha + hora
        dt_str = f'{row["fecha"]} {row["hora"]}'
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        dt = ARG_TZ.localize(dt)

        hora = dt.strftime("%H")
        spread = float(row["spread_pct"])

        datos[moneda][hora].append(spread)

print("\n📊 ANÁLISIS DE SPREAD PROMEDIO POR HORA (ARG)\n")

for moneda in sorted(datos.keys()):
    print(f"🪙 {moneda}")
    print("Hora | Promedio Spread | Muestras")
    print("-" * 40)

    for hora in sorted(datos[moneda].keys()):
        spreads = datos[moneda][hora]
        avg_spread = sum(spreads) / len(spreads)
        print(f"{hora}:00 | {avg_spread:>8.3f}% | {len(spreads)}")

    print()
