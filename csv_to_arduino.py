#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversor CSV a Array de Arduino
Convierte el CSV de trayectoria a código C/Arduino listo para copiar.
"""

import sys
import pandas as pd
import os

def csv_to_arduino_array(csv_path, output_path=None, decimate=1):
    """
    Convierte CSV a array de Arduino.
    
    Args:
        csv_path: Ruta al archivo CSV
        output_path: Ruta del archivo de salida (opcional)
        decimate: Factor de diezmado (1=todos los puntos, 2=1 de cada 2, etc.)
    """
    
    # Leer CSV
    df = pd.read_csv(csv_path, comment='#')
    
    # Diezmar si es necesario
    if decimate > 1:
        df = df.iloc[::decimate, :].reset_index(drop=True)
    
    # Generar código
    code_lines = []
    code_lines.append("/*")
    code_lines.append(f" * Trayectoria generada desde: {os.path.basename(csv_path)}")
    code_lines.append(f" * Número de puntos: {len(df)}")
    code_lines.append(f" * Factor de diezmado: {decimate}")
    code_lines.append(" * Convención:")
    code_lines.append(" *   theta1: ángulo del eslabón 1 desde horizontal [rad]")
    code_lines.append(" *   theta2: ángulo relativo del eslabón 2 [rad] (0 = colineales)")
    code_lines.append(" */")
    code_lines.append("")
    code_lines.append("struct TrajectoryPoint {")
    code_lines.append("  float time_s;")
    code_lines.append("  float theta1_rad;")
    code_lines.append("  float theta2_rad;")
    code_lines.append("};")
    code_lines.append("")
    code_lines.append(f"const int NUM_TRAJECTORY_POINTS = {len(df)};")
    code_lines.append("")
    code_lines.append("const TrajectoryPoint trajectory[NUM_TRAJECTORY_POINTS] PROGMEM = {")
    
    # Agregar datos
    for i, row in df.iterrows():
        time_s = row['time_s']
        theta1 = row['theta1_rad']
        theta2 = row['theta2_rad']
        
        # Formato con coma al final excepto en el último
        comma = "," if i < len(df) - 1 else ""
        code_lines.append(f"  {{{time_s:.6f}, {theta1:.6f}, {theta2:.6f}}}{comma}")
    
    code_lines.append("};")
    code_lines.append("")
    code_lines.append("// Función helper para leer desde PROGMEM")
    code_lines.append("TrajectoryPoint getTrajectoryPoint(int index) {")
    code_lines.append("  TrajectoryPoint point;")
    code_lines.append("  memcpy_P(&point, &trajectory[index], sizeof(TrajectoryPoint));")
    code_lines.append("  return point;")
    code_lines.append("}")
    
    # Unir líneas
    code = "\n".join(code_lines)
    
    # Guardar o imprimir
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"✓ Archivo generado: {output_path}")
        print(f"  Puntos: {len(df)}")
        print(f"  Tamaño estimado: {len(code)} bytes de código")
        print(f"  RAM necesaria: ~{len(df) * 12} bytes")
    else:
        print(code)
    
    # Advertencia si es muy grande
    if len(df) > 2000:
        print("\n⚠ ADVERTENCIA: La trayectoria tiene muchos puntos.")
        print("  Considera usar:")
        print("  1. Mayor factor de diezmado (--decimate=4)")
        print("  2. Reducir cycles o fps en Python")
        print("  3. Usar tarjeta SD en lugar de array en RAM")
        print("  4. Usar streaming desde PC (método recomendado)")

def main():
    if len(sys.argv) < 2:
        print("Uso: python csv_to_arduino.py <archivo_csv> [opciones]")
        print("\nOpciones:")
        print("  -o, --output <archivo>    Archivo de salida (ej: trajectory_data.h)")
        print("  -d, --decimate <N>        Diezmar por factor N (1=sin diezmar)")
        print("\nEjemplos:")
        print("  python csv_to_arduino.py trayectorias/trajectory_20250102_143022.csv")
        print("  python csv_to_arduino.py trajectory.csv -o trajectory_data.h -d 2")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_path = None
    decimate = 1
    
    # Parsear argumentos
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ['-o', '--output']:
            if i + 1 < len(sys.argv):
                output_path = sys.argv[i + 1]
                i += 2
            else:
                print("Error: -o requiere un argumento")
                sys.exit(1)
        elif arg in ['-d', '--decimate']:
            if i + 1 < len(sys.argv):
                try:
                    decimate = int(sys.argv[i + 1])
                    if decimate < 1:
                        print("Error: --decimate debe ser >= 1")
                        sys.exit(1)
                    i += 2
                except ValueError:
                    print("Error: --decimate debe ser un número entero")
                    sys.exit(1)
            else:
                print("Error: -d requiere un argumento")
                sys.exit(1)
        else:
            print(f"Error: Argumento desconocido '{arg}'")
            sys.exit(1)
    
    try:
        csv_to_arduino_array(csv_path, output_path, decimate)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{csv_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

