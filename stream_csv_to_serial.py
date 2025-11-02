#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Envia una trayectoria (CSV) a Arduino por Serial como comandos 'R,theta1,theta2' en radianes.
Requiere: pyserial, pandas
"""
import argparse
import time
import sys
import pandas as pd
import serial

def main():
    parser = argparse.ArgumentParser(description='Stream de trayectoria CSV a Arduino (R,theta1,theta2).')
    parser.add_argument('--port', required=True, help='Puerto serial (p.ej., COM3 o /dev/ttyACM0)')
    parser.add_argument('--baud', type=int, default=115200, help='Baudios (default: 115200)')
    parser.add_argument('--csv', required=True, help='Archivo CSV con columnas time_s,theta1_rad,theta2_rad')
    parser.add_argument('--start-index', type=int, default=None, help='Índice de inicio (opcional)')
    parser.add_argument('--first-cycle-only', action='store_true', help='Enviar solo la primera vuelta si hay metadatos')
    parser.add_argument('--dry-run', action='store_true', help='No enviar por serial, solo imprimir')
    args = parser.parse_args()

    df = pd.read_csv(args.csv, comment='#')
    # Leer metadatos de cabecera si existen
    start_index = args.start_index
    cycle_frames = None
    try:
        with open(args.csv, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.startswith('#'):
                    break
                if 'start_index=' in line:
                    start_index = int(line.split('=')[1]) if start_index is None else start_index
                if 'cycle_frames=' in line:
                    cycle_frames = int(line.split('=')[1])
    except Exception:
        pass

    t = df['time_s'].values
    th1 = df['theta1_rad'].values
    th2 = df['theta2_rad'].values

    i0 = start_index or 0
    i1 = len(df)
    if args.first_cycle_only and cycle_frames is not None:
        i1 = min(i0 + cycle_frames, len(df))

    if not args.dry_run:
        ser = serial.Serial(args.port, args.baud, timeout=0.05)
        time.sleep(0.2)
        ser.write(b'Z\n')  # cero rápido
        ser.flush()
    else:
        ser = None

    print(f"Streaming {i1 - i0} puntos de {args.csv} -> {args.port} @ {args.baud} baudios")

    t0 = time.perf_counter()
    t_csv0 = t[i0]
    for i in range(i0, i1):
        # Esperar según el tiempo del CSV (relativo al punto inicial)
        t_target = (t[i] - t_csv0)
        while True:
            if (time.perf_counter() - t0) >= t_target - 1e-4:
                break
            time.sleep(0.0005)

        cmd = f"R,{th1[i]:.6f},{th2[i]:.6f}\n"
        if args.dry_run:
            sys.stdout.write(cmd)
        else:
            ser.write(cmd.encode('ascii'))

    if ser is not None:
        ser.write(b'S\n')  # stop
        ser.flush()
        ser.close()

    print("Listo.")

if __name__ == '__main__':
    main()


