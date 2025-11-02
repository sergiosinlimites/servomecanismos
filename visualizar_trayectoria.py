#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador de Trayectorias 2R
Permite visualizar los archivos CSV generados por gemelo_digital_2R_trebol_pt2.py
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_trajectory(csv_path):
    """Visualiza la trayectoria desde un archivo CSV."""
    
    # Leer CSV (ignorar líneas de comentario)
    df = pd.read_csv(csv_path, comment='#')
    
    # Extraer datos
    t = df['time_s'].values
    theta1 = df['theta1_rad'].values
    theta2 = df['theta2_rad'].values
    
    # Calcular derivadas (velocidades y aceleraciones)
    dt = np.mean(np.diff(t))
    omega1 = np.gradient(theta1, dt)
    omega2 = np.gradient(theta2, dt)
    alpha1 = np.gradient(omega1, dt)
    alpha2 = np.gradient(omega2, dt)
    
    # Crear figura con 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    fig.suptitle(f'Análisis de Trayectoria: {csv_path}', fontsize=14, fontweight='bold')
    
    # 1. Ángulos
    axes[0].plot(t, theta1, 'b-', label='θ₁', linewidth=1.5)
    axes[0].plot(t, theta2, 'r-', label='θ₂', linewidth=1.5)
    axes[0].set_ylabel('Ángulo [rad]', fontsize=11)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Posiciones Angulares', fontsize=11, fontweight='bold')
    
    # 2. Velocidades angulares
    axes[1].plot(t, omega1, 'b-', label='ω₁', linewidth=1.5)
    axes[1].plot(t, omega2, 'r-', label='ω₂', linewidth=1.5)
    axes[1].set_ylabel('Velocidad [rad/s]', fontsize=11)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Velocidades Angulares', fontsize=11, fontweight='bold')
    
    # 3. Aceleraciones angulares
    axes[2].plot(t, alpha1, 'b-', label='α₁', linewidth=1.5)
    axes[2].plot(t, alpha2, 'r-', label='α₂', linewidth=1.5)
    axes[2].set_ylabel('Aceleración [rad/s²]', fontsize=11)
    axes[2].legend(loc='upper right', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Aceleraciones Angulares', fontsize=11, fontweight='bold')
    
    # 4. Magnitudes (norma de velocidad y aceleración)
    omega_norm = np.sqrt(omega1**2 + omega2**2)
    alpha_norm = np.sqrt(alpha1**2 + alpha2**2)
    axes[3].plot(t, omega_norm, 'g-', label='|ω|', linewidth=1.5)
    axes[3].set_ylabel('|ω| [rad/s]', fontsize=11, color='g')
    axes[3].tick_params(axis='y', labelcolor='g')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title('Magnitudes de Velocidad y Aceleración', fontsize=11, fontweight='bold')
    
    ax3_twin = axes[3].twinx()
    ax3_twin.plot(t, alpha_norm, 'm-', label='|α|', linewidth=1.5)
    ax3_twin.set_ylabel('|α| [rad/s²]', fontsize=11, color='m')
    ax3_twin.tick_params(axis='y', labelcolor='m')
    
    axes[3].set_xlabel('Tiempo [s]', fontsize=11)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Estadísticas
    print("\n" + "="*60)
    print("ESTADÍSTICAS DE LA TRAYECTORIA")
    print("="*60)
    print(f"Archivo: {csv_path}")
    print(f"\nDuración total: {t[-1]:.3f} s")
    print(f"Número de puntos: {len(t)}")
    print(f"Frecuencia de muestreo: {1/dt:.1f} Hz")
    print(f"\nÁngulos:")
    print(f"  θ₁: min={np.min(theta1):.4f} rad, max={np.max(theta1):.4f} rad, rango={np.ptp(theta1):.4f} rad")
    print(f"  θ₂: min={np.min(theta2):.4f} rad, max={np.max(theta2):.4f} rad, rango={np.ptp(theta2):.4f} rad")
    print(f"\nVelocidades angulares:")
    print(f"  ω₁: max={np.max(np.abs(omega1)):.4f} rad/s")
    print(f"  ω₂: max={np.max(np.abs(omega2)):.4f} rad/s")
    print(f"  |ω|: max={np.max(omega_norm):.4f} rad/s")
    print(f"\nAceleraciones angulares:")
    print(f"  α₁: max={np.max(np.abs(alpha1)):.4f} rad/s²")
    print(f"  α₂: max={np.max(np.abs(alpha2)):.4f} rad/s²")
    print(f"  |α|: max={np.max(alpha_norm):.4f} rad/s²")
    print("="*60 + "\n")
    
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Uso: python visualizar_trayectoria.py <archivo_csv>")
        print("\nEjemplo:")
        print("  python visualizar_trayectoria.py trayectorias/trajectory_20250102_143022.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    try:
        visualize_trajectory(csv_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{csv_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

