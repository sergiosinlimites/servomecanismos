# Simulación (Rosa / Trébol)

## Curva rosa
Se usa `r = A * cos(kφ)` o `sin`, con `k` según el número de hojas; opción **avoid_center** y **alpha** para no pasar por el origen.

## Encaje en lienzo 20×20
`fit_inside_canvas()` ajusta `A` si es necesario.

## Reparametrización por arco
Se interpola la trayectoria por longitud de arco para lograr **velocidad lineal ~ constante** (v ∈ [1,10] cm/s).

## IK y continuidad
Se intenta elbow up y down, eligiendo la solución más cercana a la del frame anterior (minimiza saltos).

## Pose inicial (park)
Se fija a la izquierda del bounding box del trébol y a ≤ 50% de la altura del mismo.
