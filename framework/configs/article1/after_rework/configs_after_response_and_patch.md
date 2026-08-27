# Полный список конфигов для прогона (after\_response\_and\_patch)

Все конфиги в формате JSON, совместимом с `SimConfig.from\_json()` / `--config`.
Каждый блок — отдельный файл. `out\_dir` уже проставлен по схеме
`.\\models\\article1\\after\_response\_and\_patch\\<TYPE\_NAME>\\`.



## 1\. Mesh screen (short, физический размер поля фиксирован)

Короткие прогоны (t\_total=10) на последовательности сеток при неизменном физическом размере домена Lx=Ly=160 и физическом радиусе пятна=8. dt масштабируется как dx^2 относительно эталона dx=1, dt=0.0015. Цель — быстро отбраковать неустойчивые сетки перед long-run.

### `mesh\_screen/MS0\_nx080\_dx2p000.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 80,
  "ny": 80,
  "dx": 2.0,
  "dt": 0.006,
  "t\_total": 10.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 40.0,
        "cy": 40.0,
        "radius": 4.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 400,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\mesh\_screen\\\\\\\\",
  "run\_id": "MS0\_nx080\_dx2p000",
  "tags": \[
    "after\_response\_and\_patch",
    "mesh\_screen",
    "MS0\_nx080\_dx2p000"
  ]
}
```

### `mesh\_screen/MS1\_nx160\_dx1p000.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 160,
  "ny": 160,
  "dx": 1.0,
  "dt": 0.0015,
  "t\_total": 10.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 80.0,
        "cy": 80.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 400,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\mesh\_screen\\\\\\\\",
  "run\_id": "MS1\_nx160\_dx1p000",
  "tags": \[
    "after\_response\_and\_patch",
    "mesh\_screen",
    "MS1\_nx160\_dx1p000"
  ]
}
```

### `mesh\_screen/MS2\_nx320\_dx0p500.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 320,
  "ny": 320,
  "dx": 0.5,
  "dt": 0.000375,
  "t\_total": 10.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 160.0,
        "cy": 160.0,
        "radius": 16.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 400,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\mesh\_screen\\\\\\\\",
  "run\_id": "MS2\_nx320\_dx0p500",
  "tags": \[
    "after\_response\_and\_patch",
    "mesh\_screen",
    "MS2\_nx320\_dx0p500"
  ]
}
```

### `mesh\_screen/MS3\_nx640\_dx0p250.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 640,
  "ny": 640,
  "dx": 0.25,
  "dt": 9.375e-05,
  "t\_total": 10.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 320.0,
        "cy": 320.0,
        "radius": 32.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 400,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\mesh\_screen\\\\\\\\",
  "run\_id": "MS3\_nx640\_dx0p250",
  "tags": \[
    "after\_response\_and\_patch",
    "mesh\_screen",
    "MS3\_nx640\_dx0p250"
  ]
}
```



## 2\. Mesh long (только для устойчивых сеток из screen)

Длинные прогоны (t\_total=800) на тех же сетках, что и mesh\_screen. Запускать только для тех run\_id, у которых screen дал solver\_status=completed.

### `mesh\_long/ML0\_nx080\_dx2p000.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 80,
  "ny": 80,
  "dx": 2.0,
  "dt": 0.006,
  "t\_total": 800.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 40.0,
        "cy": 40.0,
        "radius": 4.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 4000,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\mesh\_long\\\\\\\\",
  "run\_id": "ML0\_nx080\_dx2p000",
  "tags": \[
    "after\_response\_and\_patch",
    "mesh\_long",
    "ML0\_nx080\_dx2p000"
  ]
}
```

### `mesh\_long/ML1\_nx160\_dx1p000.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 160,
  "ny": 160,
  "dx": 1.0,
  "dt": 0.0015,
  "t\_total": 800.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 80.0,
        "cy": 80.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 4000,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\mesh\_long\\\\\\\\",
  "run\_id": "ML1\_nx160\_dx1p000",
  "tags": \[
    "after\_response\_and\_patch",
    "mesh\_long",
    "ML1\_nx160\_dx1p000"
  ]
}
```

### `mesh\_long/ML2\_nx320\_dx0p500.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 320,
  "ny": 320,
  "dx": 0.5,
  "dt": 0.000375,
  "t\_total": 800.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 160.0,
        "cy": 160.0,
        "radius": 16.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 4000,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\mesh\_long\\\\\\\\",
  "run\_id": "ML2\_nx320\_dx0p500",
  "tags": \[
    "after\_response\_and\_patch",
    "mesh\_long",
    "ML2\_nx320\_dx0p500"
  ]
}
```

### `mesh\_long/ML3\_nx640\_dx0p250.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 640,
  "ny": 640,
  "dx": 0.25,
  "dt": 9.375e-05,
  "t\_total": 800.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 320.0,
        "cy": 320.0,
        "radius": 32.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 4000,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\mesh\_long\\\\\\\\",
  "run\_id": "ML3\_nx640\_dx0p250",
  "tags": \[
    "after\_response\_and\_patch",
    "mesh\_long",
    "ML3\_nx640\_dx0p250"
  ]
}
```



## 3\. Timestep sensitivity (short)

Короткие прогоны (t\_total=10) при фиксированной сетке nx=ny=160, dx=1.0, варьируется только dt. Цель — найти порог устойчивости по времени.

### `timestep\_sensitivity/TS0\_dt0p003000.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 160,
  "ny": 160,
  "dx": 1.0,
  "dt": 0.003,
  "t\_total": 10.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 80.0,
        "cy": 80.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 400,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\timestep\_sensitivity\\\\\\\\",
  "run\_id": "TS0\_dt0p003000",
  "tags": \[
    "after\_response\_and\_patch",
    "timestep\_sensitivity",
    "TS0\_dt0p003000"
  ]
}
```

### `timestep\_sensitivity/TS1\_dt0p001500.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 160,
  "ny": 160,
  "dx": 1.0,
  "dt": 0.0015,
  "t\_total": 10.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 80.0,
        "cy": 80.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 400,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\timestep\_sensitivity\\\\\\\\",
  "run\_id": "TS1\_dt0p001500",
  "tags": \[
    "after\_response\_and\_patch",
    "timestep\_sensitivity",
    "TS1\_dt0p001500"
  ]
}
```

### `timestep\_sensitivity/TS2\_dt0p000750.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 160,
  "ny": 160,
  "dx": 1.0,
  "dt": 0.00075,
  "t\_total": 10.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 80.0,
        "cy": 80.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 400,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\timestep\_sensitivity\\\\\\\\",
  "run\_id": "TS2\_dt0p000750",
  "tags": \[
    "after\_response\_and\_patch",
    "timestep\_sensitivity",
    "TS2\_dt0p000750"
  ]
}
```

### `timestep\_sensitivity/TS3\_dt0p000375.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 160,
  "ny": 160,
  "dx": 1.0,
  "dt": 0.000375,
  "t\_total": 10.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 80.0,
        "cy": 80.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 400,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\timestep\_sensitivity\\\\\\\\",
  "run\_id": "TS3\_dt0p000375",
  "tags": \[
    "after\_response\_and\_patch",
    "timestep\_sensitivity",
    "TS3\_dt0p000375"
  ]
}
```



## 4\. Timestep long (для устойчивых dt)

Длинные прогоны (t\_total=800) с теми же dt, что в timestep\_sensitivity, только для устойчивых вариантов.

### `timestep\_long/TL0\_dt0p003000.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 160,
  "ny": 160,
  "dx": 1.0,
  "dt": 0.003,
  "t\_total": 800.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 80.0,
        "cy": 80.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 4000,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\timestep\_long\\\\\\\\",
  "run\_id": "TL0\_dt0p003000",
  "tags": \[
    "after\_response\_and\_patch",
    "timestep\_long",
    "TL0\_dt0p003000"
  ]
}
```

### `timestep\_long/TL1\_dt0p001500.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 160,
  "ny": 160,
  "dx": 1.0,
  "dt": 0.0015,
  "t\_total": 800.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 80.0,
        "cy": 80.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 4000,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\timestep\_long\\\\\\\\",
  "run\_id": "TL1\_dt0p001500",
  "tags": \[
    "after\_response\_and\_patch",
    "timestep\_long",
    "TL1\_dt0p001500"
  ]
}
```

### `timestep\_long/TL2\_dt0p000750.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 160,
  "ny": 160,
  "dx": 1.0,
  "dt": 0.00075,
  "t\_total": 800.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 80.0,
        "cy": 80.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 4000,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\timestep\_long\\\\\\\\",
  "run\_id": "TL2\_dt0p000750",
  "tags": \[
    "after\_response\_and\_patch",
    "timestep\_long",
    "TL2\_dt0p000750"
  ]
}
```

### `timestep\_long/TL3\_dt0p000375.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 160,
  "ny": 160,
  "dx": 1.0,
  "dt": 0.000375,
  "t\_total": 800.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 80.0,
        "cy": 80.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 4000,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\timestep\_long\\\\\\\\",
  "run\_id": "TL3\_dt0p000375",
  "tags": \[
    "after\_response\_and\_patch",
    "timestep\_long",
    "TL3\_dt0p000375"
  ]
}
```



## 5\. Domain size (short)

Короткие прогоны (t\_total=10) с фиксированными dx=1.0, dt=0.003, физическим радиусом пятна=8, варьируется размер домена nx=ny (значит и Lx=Ly). Проверка на конечно-размерные эффекты периодических границ.

### `domain\_size/DS0\_nx080\_L080.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 80,
  "ny": 80,
  "dx": 1.0,
  "dt": 0.003,
  "t\_total": 10.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 40.0,
        "cy": 40.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 400,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\domain\_size\\\\\\\\",
  "run\_id": "DS0\_nx080\_L080",
  "tags": \[
    "after\_response\_and\_patch",
    "domain\_size",
    "DS0\_nx080\_L080"
  ]
}
```

### `domain\_size/DS1\_nx120\_L120.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 120,
  "ny": 120,
  "dx": 1.0,
  "dt": 0.003,
  "t\_total": 10.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 60.0,
        "cy": 60.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 400,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\domain\_size\\\\\\\\",
  "run\_id": "DS1\_nx120\_L120",
  "tags": \[
    "after\_response\_and\_patch",
    "domain\_size",
    "DS1\_nx120\_L120"
  ]
}
```

### `domain\_size/DS2\_nx160\_L160.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 160,
  "ny": 160,
  "dx": 1.0,
  "dt": 0.003,
  "t\_total": 10.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 80.0,
        "cy": 80.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 400,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\domain\_size\\\\\\\\",
  "run\_id": "DS2\_nx160\_L160",
  "tags": \[
    "after\_response\_and\_patch",
    "domain\_size",
    "DS2\_nx160\_L160"
  ]
}
```

### `domain\_size/DS3\_nx240\_L240.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 240,
  "ny": 240,
  "dx": 1.0,
  "dt": 0.003,
  "t\_total": 10.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 120.0,
        "cy": 120.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 400,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\domain\_size\\\\\\\\",
  "run\_id": "DS3\_nx240\_L240",
  "tags": \[
    "after\_response\_and\_patch",
    "domain\_size",
    "DS3\_nx240\_L240"
  ]
}
```

### `domain\_size/DS4\_nx320\_L320.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 320,
  "ny": 320,
  "dx": 1.0,
  "dt": 0.003,
  "t\_total": 10.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 160.0,
        "cy": 160.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 400,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\domain\_size\\\\\\\\",
  "run\_id": "DS4\_nx320\_L320",
  "tags": \[
    "after\_response\_and\_patch",
    "domain\_size",
    "DS4\_nx320\_L320"
  ]
}
```



## 6\. Domain size long

Длинные прогоны (t\_total=800) для устойчивых размеров домена из domain\_size.

### `domain\_size\_long/DL0\_nx080\_L080.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 80,
  "ny": 80,
  "dx": 1.0,
  "dt": 0.003,
  "t\_total": 800.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 40.0,
        "cy": 40.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 4000,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\domain\_size\_long\\\\\\\\",
  "run\_id": "DL0\_nx080\_L080",
  "tags": \[
    "after\_response\_and\_patch",
    "domain\_size\_long",
    "DL0\_nx080\_L080"
  ]
}
```

### `domain\_size\_long/DL1\_nx120\_L120.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 120,
  "ny": 120,
  "dx": 1.0,
  "dt": 0.003,
  "t\_total": 800.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 60.0,
        "cy": 60.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 4000,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\domain\_size\_long\\\\\\\\",
  "run\_id": "DL1\_nx120\_L120",
  "tags": \[
    "after\_response\_and\_patch",
    "domain\_size\_long",
    "DL1\_nx120\_L120"
  ]
}
```

### `domain\_size\_long/DL2\_nx160\_L160.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 160,
  "ny": 160,
  "dx": 1.0,
  "dt": 0.003,
  "t\_total": 800.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 80.0,
        "cy": 80.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 4000,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\domain\_size\_long\\\\\\\\",
  "run\_id": "DL2\_nx160\_L160",
  "tags": \[
    "after\_response\_and\_patch",
    "domain\_size\_long",
    "DL2\_nx160\_L160"
  ]
}
```

### `domain\_size\_long/DL3\_nx240\_L240.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 240,
  "ny": 240,
  "dx": 1.0,
  "dt": 0.003,
  "t\_total": 800.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 120.0,
        "cy": 120.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 4000,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\domain\_size\_long\\\\\\\\",
  "run\_id": "DL3\_nx240\_L240",
  "tags": \[
    "after\_response\_and\_patch",
    "domain\_size\_long",
    "DL3\_nx240\_L240"
  ]
}
```

### `domain\_size\_long/DL4\_nx320\_L320.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 320,
  "ny": 320,
  "dx": 1.0,
  "dt": 0.003,
  "t\_total": 800.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 160.0,
        "cy": 160.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.0,
    "noise\_seed": 202,
    "noise\_lowpass": 0.0,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 4000,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\domain\_size\_long\\\\\\\\",
  "run\_id": "DL4\_nx320\_L320",
  "tags": \[
    "after\_response\_and\_patch",
    "domain\_size\_long",
    "DL4\_nx320\_L320"
  ]
}
```



## 7\. Init robustness (seed/noise)

Длинные прогоны (t\_total=800) при фиксированной геометрии nx=ny=160, dx=1.0, dt=0.003 с малым шумом (noise\_amplitude=0.01, noise\_lowpass=0.05) и разными seed. Проверка, что режим не зависит от конкретной случайной реализации.

### `init\_robustness/IR101\_seed101.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 160,
  "ny": 160,
  "dx": 1.0,
  "dt": 0.003,
  "t\_total": 800.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 80.0,
        "cy": 80.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.01,
    "noise\_seed": 101,
    "noise\_lowpass": 0.05,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 4000,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\init\_robustness\\\\\\\\",
  "run\_id": "IR101\_seed101",
  "tags": \[
    "after\_response\_and\_patch",
    "init\_robustness",
    "IR101\_seed101"
  ]
}
```

### `init\_robustness/IR202\_seed202.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 160,
  "ny": 160,
  "dx": 1.0,
  "dt": 0.003,
  "t\_total": 800.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 80.0,
        "cy": 80.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.01,
    "noise\_seed": 202,
    "noise\_lowpass": 0.05,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 4000,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\init\_robustness\\\\\\\\",
  "run\_id": "IR202\_seed202",
  "tags": \[
    "after\_response\_and\_patch",
    "init\_robustness",
    "IR202\_seed202"
  ]
}
```

### `init\_robustness/IR303\_seed303.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 160,
  "ny": 160,
  "dx": 1.0,
  "dt": 0.003,
  "t\_total": 800.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 80.0,
        "cy": 80.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.01,
    "noise\_seed": 303,
    "noise\_lowpass": 0.05,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 4000,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\init\_robustness\\\\\\\\",
  "run\_id": "IR303\_seed303",
  "tags": \[
    "after\_response\_and\_patch",
    "init\_robustness",
    "IR303\_seed303"
  ]
}
```

### `init\_robustness/IR404\_seed404.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 160,
  "ny": 160,
  "dx": 1.0,
  "dt": 0.003,
  "t\_total": 800.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 80.0,
        "cy": 80.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.01,
    "noise\_seed": 404,
    "noise\_lowpass": 0.05,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 4000,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\init\_robustness\\\\\\\\",
  "run\_id": "IR404\_seed404",
  "tags": \[
    "after\_response\_and\_patch",
    "init\_robustness",
    "IR404\_seed404"
  ]
}
```

### `init\_robustness/IR505\_seed505.json`

```json
{
  "eps": 2.8,
  "h\_bg": 0.533,
  "D\_psi": 0.0,
  "phi\_lo": 0.533,
  "phi\_hi": 3.9766096853487105,
  "kappa": 10.0,
  "theta1": 4.0,
  "theta2": 16.0,
  "boundary": "periodic",
  "gamma\_bg": 0.0,
  "h\_custom": null,
  "gamma\_custom": null,
  "attractors": \[],
  "monitor\_every": 10,
  "t\_warm": 0.0,
  "seed\_n\_snaps": 16,
  "nx": 160,
  "ny": 160,
  "dx": 1.0,
  "dt": 0.003,
  "t\_total": 800.0,
  "init": {
    "npz\_path": null,
    "spots": \[
      {
        "cx": 80.0,
        "cy": 80.0,
        "radius": 8.0,
        "amp": null,
        "shape": "disk",
        "phase\_v": 0.0
      }
    ],
    "noise\_amplitude": 0.01,
    "noise\_seed": 505,
    "noise\_lowpass": 0.05,
    "phi\_background": null
  },
  "snap": {
    "every\_steps": 4000,
    "t\_start": 0.0,
    "t\_stop": -1.0,
    "save\_phi": true,
    "save\_psi": true,
    "save\_v": true,
    "max\_snaps": -1,
    "labels\_save": \[]
  },
  "out\_dir": ".\\\\models\\\\article1\\\\after\_response\_and\_patch\\\\init\_robustness\\\\\\\\",
  "run\_id": "IR505\_seed505",
  "tags": \[
    "after\_response\_and\_patch",
    "init\_robustness",
    "IR505\_seed505"
  ]
}
```

