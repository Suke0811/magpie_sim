# Material, Force Sensitivity and Range


## Sensible Force Range and Force Sensitivity
<div style="text-align: center">
<img src="../material.jpg" alt="model gif" style="width:1000%;">
<br>
</div>


!!! note

    Sensible force range is limited by the flexure strength.

    By adjusting $\Gamma_{\mathcal{M}}$ we can improve the range.

    i.e. reduce the range of deflection, but configure $\Gamma_{\mathcal{M}}$ to detect smaller deflections


### Sweeping Parameters
$\Gamma_{b},\Gamma_{\mathcal{M}}$

Here, we sweeped 5,000 different parameters per material, totaling 30,000 parameter combinations.

| Parameter                             | Values                                      |
|---------------------------------------|---------------------------------------------|
| L (Length of beam)                    | 10, 30, 50, 100 (mm)                        |
| b (Width of beam)                     | 5, 10 (mm)                                  |
| h (Thickness of beam)                 | 1, 2, 5, 8, 10 (mm)                         |
| magnet_D (Diameter of magnet)         | 0.001, 0.003, 0.005, 0.01, 0.02 (m)         |
| magnet_L (Length of magnet)           | 0.001, 0.003, 0.005, 0.01, 0.02 (m)         |
| magnet_distance (Distance from sensor) | 0.001, 0.003, 0.005, 0.01, 0.015 (m)        |



### Material Parameters

| Material         | Young's Modulus (E) (MPa) | Poisson's Ratio | Yield Strength (MPa) | Fatigue Strength (MPa) |
|------------------|---------------------------|------------------|----------------------|------------------------|
| Aluminum         | 69,000                    | 0.33             | 276                  | 96                     |
| Stainless Steel  | 200,000                   | 0.30             | 520                  | 240                    |
| Brass            | 100,000                   | 0.34             | 200                  | 100                    |
| PLA              | 3,500                     | 0.36             | 60                   | 30                     |
| ABS              | 2,100                     | 0.35             | 40                   | 20                     |
| Nylon 12         | 1,650                     | 0.39             | 48                   | 24                     |

Mostly from [Engineering Toolbox](https://www.engineeringtoolbox.com/engineering-materials-properties-d_1225.html)
