import pandas as pd
import numpy as np
import os
from utils import obtener_ventana_fresca
from config import CONFIG

class MicroEngine:

    def __init__(self, cfg):
        self.cfg = cfg
        self.asset = cfg.get('asset', 'USDT').upper()
        self.base_path = os.path.join(cfg['DATA_ROOT'], self.asset)
        self.hist_file = os.path.join(self.base_path, 'micro.csv')
        os.makedirs(self.base_path, exist_ok=True)

    def _get_operable_amount(self, universe, desired_usd, ref_price):
        ars_needed = desired_usd * ref_price
        valid = universe[universe['min_single_amount'] <= ars_needed]
        if not valid.empty:
            return desired_usd, ars_needed, False
        else:
            min_req = universe['min_single_amount'].min()
            return (min_req / ref_price), min_req, True

    def _save_record(self, record):
        df_new = pd.DataFrame([record])
        if not os.path.exists(self.hist_file):
            df_new.to_csv(self.hist_file, index=False)
        else:
            df_new.to_csv(self.hist_file, mode='a', header=False, index=False)

    def run(self):
        df = obtener_ventana_fresca(self.cfg['DATA_ROOT'], self.asset, horas_atras=2)
        if df is None or df.empty:
            print(f"❌ Sin datos para {self.asset}")
            return None

        ts_abs = df['timestamp'].max()
        df_now = df[df['timestamp'] >= (ts_abs - pd.Timedelta(minutes=5))].copy()
        df_prev = df[(df['timestamp'] < (ts_abs - pd.Timedelta(minutes=10))) &
                     (df['timestamp'] >= (ts_abs - pd.Timedelta(minutes=15)))].copy()

        def process_side(side):
            universe = df_now[df_now['lado'] == side].sort_values('precio', ascending=(side == 'SELL'))
            if universe.empty:
                return None

            base_price = universe.iloc[0]['precio']
            cap_usd, cap_ars, adjusted = self._get_operable_amount(
                universe, self.cfg['capital_usd'], base_price
            )
            valid_universe = universe[universe['min_single_amount'] <= cap_ars].copy()
            if valid_universe.empty:
                return None

            ref_price = valid_universe.iloc[0]['precio']

            if side == 'SELL':
                wall_mask = valid_universe['precio'] <= ref_price * 1.01
            else:
                wall_mask = valid_universe['precio'] >= ref_price * 0.99
            wall_vol = valid_universe[wall_mask]['max_single_amount'].sum() / ref_price

            valid_universe['cum_vol'] = valid_universe['max_single_amount'].cumsum()
            full_mask = valid_universe['cum_vol'] <= cap_ars
            slices = valid_universe[full_mask]
            vol_accum = slices['max_single_amount'].sum()

            if vol_accum < cap_ars and len(valid_universe) > len(slices):
                last = valid_universe.iloc[len(slices)]
                vwap = ((slices['precio'] * slices['max_single_amount']).sum() +
                        (last['precio'] * (cap_ars - vol_accum))) / cap_ars
                full_fill = True
            else:
                vwap = (slices['precio'] * slices['max_single_amount']).sum() / vol_accum if vol_accum > 0 else ref_price
                full_fill = vol_accum >= cap_ars

            rep_vol = (valid_universe['max_single_amount'] * valid_universe['month_finish_rate']).sum()

            return {
                'price': ref_price,
                'vwap': vwap,
                'wall': wall_vol,
                'nick': valid_universe.iloc[0]['nick_name'],
                'rate': valid_universe.iloc[0]['positive_rate'],
                'rep_vol': rep_vol,
                'cap_usd': cap_usd,
                'full_fill': full_fill,
                'adjusted': adjusted
            }

        buy = process_side('BUY')
        sell = process_side('SELL')
        if not buy or not sell:
            return None

        prev_sell = df_prev[df_prev['lado'] == 'SELL'].sort_values('precio')
        trend = ((sell['price'] / prev_sell.iloc[0]['precio']) - 1) * 100 if not prev_sell.empty else 0
        strength = (sell['rep_vol'] / buy['rep_vol']) if buy['rep_vol'] > 0 else 0

        # Diccionario con nombres compatibles con el orquestador
        record = {
            'timestamp': ts_abs,
            'asset': self.asset,
            'p_c': sell['price'],
            'vwap_c': sell['vwap'],
            'muro_c': sell['wall'],
            'full_c': sell['full_fill'],
            'nick_c': sell['nick'],
            'p_v': buy['price'],
            'vwap_v': buy['vwap'],
            'muro_v': buy['wall'],
            'full_v': buy['full_fill'],
            'nick_v': buy['nick'],
            'fuerza': strength,
            'trend_15m': trend,
            'cap_usd_real': sell['cap_usd'],
            'ajustado': sell['adjusted']
        }

        self._save_record(record)

        print(f"=== 🦅 MICRO | {self.asset} | {ts_abs.strftime('%H:%M')} ===")
        print(f"💰 CAP. REAL: ${sell['cap_usd']:,.2f} USD {'⚠️' if sell['adjusted'] else ''}")
        print(f"📥 COMPRA: ${sell['price']:.2f} | VWAP: ${sell['vwap']:.2f} | Muro: {sell['wall']:,.0f} | Elite: {sell['nick']}")
        print(f"📤 VENTA:  ${buy['price']:.2f} | VWAP: ${buy['vwap']:.2f} | Spread: {((buy['price']/sell['price'])-1)*100:.2f}% | Elite: {buy['nick']}")
        print(f"📊 FUERZA: {strength:.2f} | TREND 15m: {trend:+.4f}%")
        print("="*60)

        return record
