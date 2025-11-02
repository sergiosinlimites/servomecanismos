# Changelog

## [2.2-serial] - 2025-10-29
### Added
- Streaming Serial a Arduino (comando `R,θ1,θ2`), lectura de telemetría `Y,...`.
- CSV de telemetría y compare_sim_vs_real con métricas y gráficas.
- Límite de rotación ±45° y factor de escala ≤ ×1.2 (CLI).

### Changed
- Pose inicial de “park” a la izquierda del bounding box y y ≤ 50% de la altura.

### Fixed
- Continuidad de rama en IK para evitar “flips” de codo.

## [2.1-debug] - previo
- Banner de versión, `endpoint=False` en la rosa, `avoid_center+alpha`, fallback de amplitud.
