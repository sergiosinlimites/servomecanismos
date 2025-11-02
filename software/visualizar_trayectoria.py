#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador de Trayectorias 2R
Permite visualizar los archivos CSV generados por gemelo_digital_2R_trebol_pt2.py
Actualizado: Gráficas en grados, opción de skip inicial, análisis de caracterización
"""

import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def _parse_csv_metadata(csv_path):
    meta = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.startswith('#'):
                    break
                m = re.search(r"start_index\s*=\s*(\d+)", line)
                if m:
                    meta['start_index'] = int(m.group(1))
                m = re.search(r"cycle_frames\s*=\s*(\d+)", line)
                if m:
                    meta['cycle_frames'] = int(m.group(1))
                m = re.search(r"fps\s*=\s*(\d+)", line)
                if m:
                    meta['fps'] = int(m.group(1))
    except Exception:
        pass
    return meta

def visualize_trajectory(csv_path, skip_initial=False, show_analysis=True, first_cycle_only=False):
    """Visualiza la trayectoria desde un archivo CSV."""
    
    # Leer CSV (ignorar líneas de comentario)
    df = pd.read_csv(csv_path, comment='#')
    
    # Extraer datos
    t = df['time_s'].values
    theta1_rad = df['theta1_rad'].values
    theta2_rad = df['theta2_rad'].values
    
    # Calcular derivadas (velocidades y aceleraciones)
    dt = np.mean(np.diff(t))
    omega1_rad = np.gradient(theta1_rad, dt)
    omega2_rad = np.gradient(theta2_rad, dt)
    alpha1_rad = np.gradient(omega1_rad, dt)
    alpha2_rad = np.gradient(omega2_rad, dt)
    
    # Metadatos (si existen) para recortar inicio y longitud de ciclo
    meta = _parse_csv_metadata(csv_path)
    idx_start = 0
    if skip_initial:
        if 'start_index' in meta:
            idx_start = int(meta['start_index'])
        else:
            # Heurística de respaldo si no hay metadatos
            if len(t) > 120:
                omega_norm = np.sqrt(omega1_rad**2 + omega2_rad**2)
                threshold = np.percentile(omega_norm, 10)
                stable_start = np.where(omega_norm > threshold * 2)[0]
                if len(stable_start) > 0:
                    idx_start = max(60, stable_start[0] - 10)
    
    # Determinar fin (solo primer ciclo si se solicita)
    if first_cycle_only and 'cycle_frames' in meta:
        idx_end = min(len(t), idx_start + int(meta['cycle_frames']))
    else:
        idx_end = len(t)

    # Recortar datos
    t_plot = t[idx_start:idx_end]
    
    # Convertir a grados para visualización
    theta1 = np.rad2deg(theta1_rad[idx_start:idx_end])
    theta2 = np.rad2deg(theta2_rad[idx_start:idx_end])
    omega1 = np.rad2deg(omega1_rad[idx_start:idx_end])
    omega2 = np.rad2deg(omega2_rad[idx_start:idx_end])
    alpha1 = np.rad2deg(alpha1_rad[idx_start:idx_end])
    alpha2 = np.rad2deg(alpha2_rad[idx_start:idx_end])
    
    # Crear figura con 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    fig.suptitle(f'Análisis de Trayectoria: {csv_path}', fontsize=14, fontweight='bold')
    
    # 1. Ángulos
    axes[0].plot(t_plot, theta1, 'b-', label='θ₁', linewidth=1.5)
    axes[0].plot(t_plot, theta2, 'r-', label='θ₂', linewidth=1.5)
    axes[0].set_ylabel('Ángulo [°]', fontsize=11)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Posiciones Angulares', fontsize=11, fontweight='bold')
    
    # 2. Velocidades angulares
    axes[1].plot(t_plot, omega1, 'b-', label='ω₁', linewidth=1.5)
    axes[1].plot(t_plot, omega2, 'r-', label='ω₂', linewidth=1.5)
    axes[1].set_ylabel('Velocidad [°/s]', fontsize=11)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Velocidades Angulares', fontsize=11, fontweight='bold')
    
    # 3. Aceleraciones angulares
    axes[2].plot(t_plot, alpha1, 'b-', label='α₁', linewidth=1.5)
    axes[2].plot(t_plot, alpha2, 'r-', label='α₂', linewidth=1.5)
    axes[2].set_ylabel('Aceleración [°/s²]', fontsize=11)
    axes[2].legend(loc='upper right', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Aceleraciones Angulares', fontsize=11, fontweight='bold')
    
    # 4. Magnitudes (norma de velocidad y aceleración)
    omega_norm = np.sqrt(omega1**2 + omega2**2)
    alpha_norm = np.sqrt(alpha1**2 + alpha2**2)
    axes[3].plot(t_plot, omega_norm, 'g-', label='|ω|', linewidth=1.5)
    axes[3].set_ylabel('|ω| [°/s]', fontsize=11, color='g')
    axes[3].tick_params(axis='y', labelcolor='g')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title('Magnitudes de Velocidad y Aceleración', fontsize=11, fontweight='bold')
    
    ax3_twin = axes[3].twinx()
    ax3_twin.plot(t_plot, alpha_norm, 'm-', label='|α|', linewidth=1.5)
    ax3_twin.set_ylabel('|α| [°/s²]', fontsize=11, color='m')
    ax3_twin.tick_params(axis='y', labelcolor='m')
    
    axes[3].set_xlabel('Tiempo [s]', fontsize=11)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Estadísticas en consola
    print("\n" + "="*60)
    print("ESTADÍSTICAS DE LA TRAYECTORIA")
    print("="*60)
    print(f"Archivo: {csv_path}")
    print(f"\nDuración total: {t[-1]:.3f} s")
    print(f"Número de puntos: {len(t)}")
    if skip_initial:
        print(f"Puntos analizados (sin inicial): {len(t_plot)}")
    print(f"Frecuencia de muestreo: {1/dt:.1f} Hz")
    print(f"\nÁngulos (en datos analizados):")
    print(f"  θ₁: min={np.min(theta1):.2f}°, max={np.max(theta1):.2f}°, rango={np.ptp(theta1):.2f}°")
    print(f"  θ₂: min={np.min(theta2):.2f}°, max={np.max(theta2):.2f}°, rango={np.ptp(theta2):.2f}°")
    print(f"\nVelocidades angulares:")
    print(f"  ω₁: max={np.max(np.abs(omega1)):.2f}°/s")
    print(f"  ω₂: max={np.max(np.abs(omega2)):.2f}°/s")
    print(f"  |ω|: max={np.max(omega_norm):.2f}°/s")
    print(f"\nAceleraciones angulares:")
    print(f"  α₁: max={np.max(np.abs(alpha1)):.2f}°/s²")
    print(f"  α₂: max={np.max(np.abs(alpha2)):.2f}°/s²")
    print(f"  |α|: max={np.max(alpha_norm):.2f}°/s²")
    
    # Ángulos más comunes
    hist1, bins1 = np.histogram(theta1, bins=40)
    most_common_th1 = (bins1[np.argmax(hist1)] + bins1[np.argmax(hist1) + 1]) / 2
    hist2, bins2 = np.histogram(theta2, bins=40)
    most_common_th2 = (bins2[np.argmax(hist2)] + bins2[np.argmax(hist2) + 1]) / 2
    print(f"\nÁngulos más comunes:")
    print(f"  θ₁: {most_common_th1:.2f}°")
    print(f"  θ₂: {most_common_th2:.2f}°")
    
    # Condiciones más demandantes
    effort = omega_norm + 0.1 * alpha_norm
    idx_max_omega = np.argmax(omega_norm)
    idx_max_alpha = np.argmax(alpha_norm)
    idx_max_effort = np.argmax(effort)
    print(f"\nCondiciones más demandantes:")
    print(f"  Mayor velocidad: θ₁={theta1[idx_max_omega]:.2f}°, θ₂={theta2[idx_max_omega]:.2f}°")
    print(f"  Mayor aceleración: θ₁={theta1[idx_max_alpha]:.2f}°, θ₂={theta2[idx_max_alpha]:.2f}°")
    print(f"  Mayor esfuerzo: θ₁={theta1[idx_max_effort]:.2f}°, θ₂={theta2[idx_max_effort]:.2f}°")
    print("="*60 + "\n")
    
    # Análisis adicional (histogramas y caracterización)
    if show_analysis:
        show_angle_analysis(theta1, theta2, omega1, omega2, alpha1, alpha2, csv_path)
    
    plt.show()

def show_angle_analysis(theta1, theta2, omega1, omega2, alpha1, alpha2, csv_path):
    """Muestra ventana adicional con análisis de caracterización."""
    # Usar constrained_layout para mejorar el ajuste y evitar solapes
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    fig.canvas.manager.set_window_title("Análisis de Caracterización")
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.25])
    
    # Métricas de demanda
    omega_norm = np.sqrt(omega1**2 + omega2**2)
    alpha_norm = np.sqrt(alpha1**2 + alpha2**2)
    effort = omega_norm + 0.1 * alpha_norm
    
    # ===== 1. HISTOGRAMA θ1 =====
    ax_hist1 = fig.add_subplot(gs[0, 0])
    counts1, bins1, _ = ax_hist1.hist(theta1, bins=40, alpha=0.7, color='tab:blue', edgecolor='black')
    ax_hist1.set_xlabel('θ₁ [°]', fontsize=10)
    ax_hist1.set_ylabel('Frecuencia', fontsize=10)
    ax_hist1.set_title('Distribución de θ₁', fontsize=11, fontweight='bold')
    ax_hist1.grid(True, alpha=0.3)
    most_common_idx1 = np.argmax(counts1)
    most_common_th1 = (bins1[most_common_idx1] + bins1[most_common_idx1 + 1]) / 2
    ax_hist1.axvline(most_common_th1, color='red', linestyle='--', linewidth=2, label=f'Más común: {most_common_th1:.1f}°')
    ax_hist1.legend(fontsize=9)
    
    # ===== 2. HISTOGRAMA θ2 =====
    ax_hist2 = fig.add_subplot(gs[0, 1])
    counts2, bins2, _ = ax_hist2.hist(theta2, bins=40, alpha=0.7, color='tab:orange', edgecolor='black')
    ax_hist2.set_xlabel('θ₂ [°]', fontsize=10)
    ax_hist2.set_ylabel('Frecuencia', fontsize=10)
    ax_hist2.set_title('Distribución de θ₂', fontsize=11, fontweight='bold')
    ax_hist2.grid(True, alpha=0.3)
    most_common_idx2 = np.argmax(counts2)
    most_common_th2 = (bins2[most_common_idx2] + bins2[most_common_idx2 + 1]) / 2
    ax_hist2.axvline(most_common_th2, color='red', linestyle='--', linewidth=2, label=f'Más común: {most_common_th2:.1f}°')
    ax_hist2.legend(fontsize=9)
    
    # ===== 3. HISTOGRAMA 2D (θ1 vs θ2) =====
    ax_hist2d = fig.add_subplot(gs[0, 2])
    h, xedges, yedges, im = ax_hist2d.hist2d(theta1, theta2, bins=30, cmap='YlOrRd')
    ax_hist2d.set_xlabel('θ₁ [°]', fontsize=10)
    ax_hist2d.set_ylabel('θ₂ [°]', fontsize=10)
    ax_hist2d.set_title('Combinaciones θ₁-θ₂', fontsize=11, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax_hist2d)
    cbar.set_label('Frecuencia', fontsize=9)
    max_idx = np.unravel_index(np.argmax(h), h.shape)
    combo_th1 = (xedges[max_idx[0]] + xedges[max_idx[0] + 1]) / 2
    combo_th2 = (yedges[max_idx[1]] + yedges[max_idx[1] + 1]) / 2
    ax_hist2d.plot(combo_th1, combo_th2, 'b*', markersize=15, label=f'Más común')
    ax_hist2d.legend(fontsize=8)
    
    # ===== 4. DEMANDA DE VELOCIDAD =====
    ax_omega = fig.add_subplot(gs[1, 0])
    ax_omega.scatter(theta1, theta2, c=omega_norm, s=5, cmap='viridis', alpha=0.6)
    cbar_w = plt.colorbar(ax_omega.collections[0], ax=ax_omega)
    cbar_w.set_label('|ω| [°/s]', fontsize=9)
    ax_omega.set_xlabel('θ₁ [°]', fontsize=10)
    ax_omega.set_ylabel('θ₂ [°]', fontsize=10)
    ax_omega.set_title('Demanda de Velocidad', fontsize=11, fontweight='bold')
    ax_omega.grid(True, alpha=0.3)
    top_omega_idx = np.argsort(omega_norm)[-5:]
    ax_omega.scatter(theta1[top_omega_idx], theta2[top_omega_idx], 
                     c='red', s=50, marker='x', linewidths=2, label='Top 5')
    ax_omega.legend(fontsize=8)
    
    # ===== 5. DEMANDA DE ACELERACIÓN =====
    ax_alpha = fig.add_subplot(gs[1, 1])
    ax_alpha.scatter(theta1, theta2, c=alpha_norm, s=5, cmap='plasma', alpha=0.6)
    cbar_a = plt.colorbar(ax_alpha.collections[0], ax=ax_alpha)
    cbar_a.set_label('|α| [°/s²]', fontsize=9)
    ax_alpha.set_xlabel('θ₁ [°]', fontsize=10)
    ax_alpha.set_ylabel('θ₂ [°]', fontsize=10)
    ax_alpha.set_title('Demanda de Aceleración', fontsize=11, fontweight='bold')
    ax_alpha.grid(True, alpha=0.3)
    top_alpha_idx = np.argsort(alpha_norm)[-5:]
    ax_alpha.scatter(theta1[top_alpha_idx], theta2[top_alpha_idx], 
                     c='red', s=50, marker='x', linewidths=2, label='Top 5')
    ax_alpha.legend(fontsize=8)
    
    # ===== 6. ESFUERZO COMBINADO =====
    ax_effort = fig.add_subplot(gs[1, 2])
    ax_effort.scatter(theta1, theta2, c=effort, s=5, cmap='coolwarm', alpha=0.6)
    cbar_e = plt.colorbar(ax_effort.collections[0], ax=ax_effort)
    cbar_e.set_label('Esfuerzo', fontsize=9)
    ax_effort.set_xlabel('θ₁ [°]', fontsize=10)
    ax_effort.set_ylabel('θ₂ [°]', fontsize=10)
    ax_effort.set_title('Esfuerzo Combinado', fontsize=11, fontweight='bold')
    ax_effort.grid(True, alpha=0.3)
    top_effort_idx = np.argsort(effort)[-5:]
    ax_effort.scatter(theta1[top_effort_idx], theta2[top_effort_idx], 
                      c='red', s=50, marker='x', linewidths=2, label='Top 5')
    ax_effort.legend(fontsize=8)
    
    # ===== 7. TABLA RESUMEN =====
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    stats_data = [
        ['PARÁMETRO', 'θ₁', 'θ₂', 'UNIDAD'],
        ['─' * 30, '─' * 15, '─' * 15, '─' * 10],
        ['Más común', f'{most_common_th1:.2f}', f'{most_common_th2:.2f}', '°'],
        ['Combinación frecuente', f'{combo_th1:.2f}', f'{combo_th2:.2f}', '°'],
        ['Promedio', f'{np.mean(theta1):.2f}', f'{np.mean(theta2):.2f}', '°'],
        ['Desv. estándar', f'{np.std(theta1):.2f}', f'{np.std(theta2):.2f}', '°'],
        ['Rango', f'{np.ptp(theta1):.2f}', f'{np.ptp(theta2):.2f}', '°'],
    ]
    table = ax_stats.table(cellText=stats_data, cellLoc='left', loc='center',
                           colWidths=[0.4, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    # Reducir escala para evitar que tape las gráficas
    table.scale(1, 1.3)
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

def main():
    if len(sys.argv) < 2:
        print("Uso: python visualizar_trayectoria.py <archivo_csv> [opciones]")
        print("\nOpciones:")
        print("  --exclude-initial    Ocultar posicionamiento inicial (por defecto se muestra)")
        print("  --include-initial    Forzar mostrar posicionamiento inicial (por defecto)")
        print("  --first-cycle        Mostrar solo la primera vuelta")
        print("  --all-cycles         Mostrar todas las vueltas (por defecto)")
        print("  --no-analysis        No mostrar ventana de caracterización")
        print("\nEjemplos:")
        print("  python visualizar_trayectoria.py trayectorias/trajectory_20250102_143022.csv")
        print("  python visualizar_trayectoria.py trajectory.csv --exclude-initial --first-cycle")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    # Defaults: mostrar TODO (incluye inicial, todas las vueltas)
    skip_initial = ('--exclude-initial' in sys.argv) and ('--include-initial' not in sys.argv)
    show_analysis = ('--no-analysis' not in sys.argv)
    first_cycle_only = ('--first-cycle' in sys.argv) and ('--all-cycles' not in sys.argv)
    
    try:
        visualize_trajectory(csv_path, skip_initial=skip_initial, show_analysis=show_analysis, first_cycle_only=first_cycle_only)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{csv_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

