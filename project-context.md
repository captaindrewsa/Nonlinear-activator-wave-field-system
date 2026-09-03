# Контекстный документ по проекту: nonlinear activator–wave-field system

Этот документ предназначен как автономная рабочая база для нового чата или нового цикла работы над проектом. Его задача — быстро восстановить контекст по первой статье, численному solver, ключевым наблюдениям, критике рецензентов, найденным уязвимостям, уже выполненным проверкам и приоритетным следующим шагам. Основа документа собрана по текущей рукописи `cnsns_autosoliton_wave-v4.tex`, вспомогательным manifest-файлам и численным диагностическим данным. [file:167][file:170][file:177][file:183][file:175]

## Что за модель

Рассматривается двухполевой activator–inhibitor тип системы, где активатор \\(\phi\\) подчиняется нелинейному уравнению первого порядка по времени, а ингибитор \\(\psi\\) — затухающему уравнению второго порядка по времени, то есть wave-mediated inhibitor, а не классической first-order relaxation dynamics. Именно это отличие переводит линейную характеристическую задачу для однородного состояния из квадратичной в кубическую и допускает комплексно-сопряжённые спектральные ветви и осцилляторные линейные моды. [file:167]

Базовая система в статье записана как

\\[
\partial_t \phi = M(\phi) + h + \psi,
\\]
\\[
\partial_{tt}\psi + \varepsilon\partial_t\psi = D_\psi \Delta \psi + \Delta \phi,
\\]

где нелинейность \\(M(\phi)\\) задаётся через две сигмоиды с параметрами \\(\kappa=10\\), \\(\theta_1=4\\), \\(\theta_2=16\\), а для \\(h=0.533\\) имеются два устойчивых однородных состояния \\(\phi_{lo}\approx0.533\\) и \\(\phi_{hi}\approx3.977\\). [file:167]

## Главная идея первой статьи

Первая статья для CNSNS посвящена не общему полевому структурообразованию, а более узкой задаче: показать, что в минимальной activator–wave-field системе могут существовать локализованные breathing autosolitons, separation-dependent two-spot outcomes и long-lived transient displacement асимметричных пар. Работа строится как сочетание линейного анализа однородного состояния и прямого численного моделирования локализованных начальных условий. [file:167]

Ключевой тезис не должен формулироваться как «второй порядок автоматически создаёт breathing», потому что это было бы слишком сильным утверждением. Корректная версия: second-order inhibitor dynamics меняет временную структуру линейного спектра, а в исследованном диапазоне параметров это численно ассоциируется с breathing localised states и нетривиальными interaction effects. [file:167]

## Что именно уже утверждается в статье

В текущей версии рукописи показаны следующие результаты:

- выведено кубическое дисперсионное соотношение для линейзации около homogeneous low state; [file:167]
- получены условия Routh–Hurwitz и критическое волновое число \\(k_c\\) для устойчивости однородного состояния; [file:167]
- в численных экспериментах наблюдается breathing regime для локализованного spot initial condition; [file:167]
- при увеличении damping coefficient \\(\varepsilon\\) наблюдается breathing-to-collapse transition, который сформулирован только как Hopf-type в описательном, а не строгом бифуркационном смысле; [file:167]
- observed spatial tail structure интерпретируется осторожно, без ложного утверждения, что tail eigenvalues уже выведены аналитически; [file:167]
- для двух пятен показаны separation-dependent outcomes и long-lived transient displacement, но без claim о существовании постоянного travelling branch. [file:167]

## Что именно НЕ утверждается

Чтобы не потерять аккуратность формулировок, важно помнить список запретных overclaims:

- не утверждается строгая Hopf bifurcation локализованной ветви; [file:167]
- не утверждается continuation/localized-spectrum result; [file:167]
- не утверждается полная mesh convergence при \\(\Delta x < 1\\); [file:167][file:177]
- не утверждается, что strong damping limit автоматически даёт классическую first-order inhibitor model; вместо этого в тексте отдельно поясняется, что требуется специальная модель вида Eq. (16); [file:167]
- не утверждается, что observed drift есть stable glider; он сформулирован как long-lived transient displacement; [file:167]
- не утверждается прямая биофизическая идентификация с GABA current или прямой gamma calibration. [file:167]

## Solver: как он устроен

Основная реализация находится в `sim_framework.py`. Система приводится к first-order form через введение вспомогательного поля \\(v=\partial_t\psi\\), после чего интегрируются три уравнения: для \\(\phi\\), \\(\psi\\) и \\(v\\). Пространственный лапласиан по умолчанию считается через isotropic nine-point stencil, а time stepping в рабочей ветке реализован явно. [file:175][file:167]

С точки зрения кода шаг solver выглядит так:

- вычисляется \\(\Delta \phi\\);
- вычисляется \\(\Delta \psi\\);
- обновляется \\(\phi\\) по явной схеме через \\(M(\phi)+h+\psi\\);
- обновляется \\(v\\) через \\(\Delta\phi + D_\psi\Delta\psi - \varepsilon v\\);
- обновляется \\(\psi\\) через новое значение \\(v\\). [file:175]

В solver встроены safeguards: проверка nonfinite значений и safety bounds по \\(\phi\\), \\(\psi\\), \\(v\\), по умолчанию на уровне 1000. При превышении порога run маркируется как `numerically_unstable` или, в старых manifest-ветках, как completed с заполненным `failure_reason`, что и было источником логической несогласованности некоторых старых manifest-файлов. [file:175][file:98]

## Важная архитектура solver

`SimConfig` управляет всеми основными параметрами: `eps`, `h_bg`, `D_psi`, `nx`, `ny`, `dx`, `dt`, `t_total`, boundary conditions, attractors, init configuration и snapshot policy. В коде есть полезная функция `suggest_dt(dx, dx_ref=1.0, dt_ref=0.0015)`, которая масштабирует шаг времени как \\(\Delta t\propto \Delta x^2\\); это эвристика для explicit integration, а не доказанная оптимальная граница устойчивости. [file:175]

Поддерживаются разные типы начальных условий и forcing-структур:

- одиночный диск;
- несколько дисков;
- gaussian spots;
- начальный `phase_v` для «пинка» волнового поля;
- spatial attractors: `pump`, `sink`, `hsink`;
- формы attractor-профиля: `gaussian`, `disk`, `ring`;
- шум с возможным low-pass filtering. [file:175]

## Примеры встроенных сценариев solver

В `sim_framework.py` уже имеются example-конфиги, которые можно использовать как отправную точку: `singlespot`, `gliderpair`, `triadattractors`, `sinkbarrier`, `channelring`, `fullfieldnoise`. Это важно для второй статьи, потому что проект уже технически вышел за рамки одного autosoliton и может работать с полевым структурообразованием на всём домене. [file:175]

### Example: baseline single spot

```json
{
  "eps": 2.8,
  "hbg": 0.533,
  "nx": 160,
  "ny": 160,
  "dx": 1.0,
  "dt": 0.003,
  "ttotal": 600.0,
  "init": {
    "spots": [{"cx": 80, "cy": 80, "radius": 8, "shape": "disk"}]
  }
}
```

Это минимальный сценарий для локализованного spot run из логики первой статьи. [file:175]

### Example: asymmetric/glider-like pair

```json
{
  "eps": 2.8,
  "hbg": 0.545,
  "nx": 160,
  "ny": 160,
  "dx": 1.0,
  "dt": 0.003,
  "ttotal": 1400.0,
  "init": {
    "spots": [
      {"cx": 73, "cy": 78, "radius": 8, "shape": "disk"},
      {"cx": 87, "cy": 82, "radius": 8, "shape": "disk", "phase_v": 0.06}
    ]
  }
}
```

Этот сценарий соответствует линии поиска transient drift / glider-like dynamics. [file:175]

### Example: triad attractors

```json
{
  "eps": 2.8,
  "hbg": 0.50,
  "gammabg": 0.15,
  "nx": 160,
  "ny": 160,
  "dx": 1.0,
  "dt": 0.003,
  "ttotal": 800.0,
  "attractors": [
    {"kind": "pump", "profile": "gaussian", "strength": 0.04, "sigma": 10.0, "cx": 80.0, "cy": 45.0},
    {"kind": "pump", "profile": "gaussian", "strength": 0.04, "sigma": 10.0, "cx": 49.7, "cy": 97.5},
    {"kind": "pump", "profile": "gaussian", "strength": 0.04, "sigma": 10.0, "cx": 110.3, "cy": 97.5}
  ]
}
```

Этот класс сценариев концептуально связан уже со второй статьёй, где возникает идея competition, nucleation, crystallization points и emergence of multiple structures. [file:175]

### Example: full-field noise

```json
{
  "eps": 2.8,
  "hbg": 0.533,
  "nx": 256,
  "ny": 256,
  "dx": 1.0,
  "dt": 0.003,
  "ttotal": 1200.0,
  "init": {
    "phi_background": 0.54,
    "noise_amplitude": 0.03,
    "noise_seed": 7,
    "noise_lowpass": 0.05
  }
}
```

Это ключевой тип запуска для изучения самозарождения структур не из заранее заданного spots, а из возмущённого поля. [file:175]

## Как устроен анализ результатов

В первой статье использовался operational classification pipeline: после initial transient анализируется центральный сигнал \\(\phi_c(t)\\), по окну наблюдения считается peak-to-peak amplitude \\(A_{pp}\\), выделяется dominant frequency через FFT и оценивается prominence ratio \\(r_{FFT}\\). Именно по этим критериям run классифицируется как `collapsed`, `breathing`, `weakly_modulated_or_unresolved` или `numerically_unstable`. [file:167][file:98]

Базовые operational thresholds в первой статье:

- collapsed, если \\(\phi_{max} \le \phi_{lo}+0.3\\);
- breathing, если localised spot present, \\(A_{pp} > 0.15\\) и \\(r_{FFT} \ge 5\\);
- weakly modulated or unresolved — если spot есть, но эти критерии не выполнены;
- numerically unstable — при nonfinite или safety-bound event. [file:167]

Важное уточнение: позднее мы исправляли старую логическую ошибку в manifest-логике. Для short-run и unresolved cases нельзя записывать осмысленные `frequency` и `period`, если `r_fft=0` или `fft_significant=false`. В новом global manifest это было починено: у short-run sensitivity cases частота и период стали `NaN`, а сам режим остался `weakly_modulated_or_unresolved`. [file:98]

## Что было найдено и исправлено в тексте первой статьи

В процессе редактирования первой статьи были найдены и устранены следующие важные содержательные слабости:

### 1. Линейный анализ был сделан существенно строже

В текущей версии:

- явно выписаны коэффициенты кубического полинома;
- дан критерий Routh–Hurwitz;
- отдельно разобран случай \\(k=0\\) с нейтральным корнем;
- введён предел по \\(D_\psi < (1+\varepsilon)/\varepsilon\\);
- для \\(\varepsilon=3, D_\psi=0\\) выписан discriminant и указано, где появляются complex-conjugate roots. [file:167]

Это было принципиально важно, потому что рецензент указывал на недостаточную полноту linear stability analysis. [file:167]

### 2. Убран overclaim про Hopf bifurcation

Текущая формулировка — только `Hopf-type transition`, а не доказанная Hopf bifurcation. В тексте прямо сказано, что без continuation и спектра локализованной ветви нельзя говорить о transversality, criticality и normal-form coefficient. [file:167]

### 3. Уточнён strong-damping limit

Мы отдельно проговорили, что large \\(\varepsilon\\) в текущем second-order equation не даёт автоматически classical first-order inhibitor equation с локальным decay term. Для честного сравнения нужна отдельно введённая модель Eq. (16). [file:167]

### 4. Нейронная интерпретация стала безопасной

Вместо прямых или расплывчатых заявлений теперь используется аккуратная phenomenological neural-field reading: \\(\phi\\) как local activity, \\(\psi\\) как inhibitory input with finite rise/decay times; добавлены references на Wang–Buzsáki и Buzsáki–Wang, но без притязания на direct physiological calibration. [file:167]

## Главные замечания рецензентов и как мы на них отвечали

### Reviewer #2: математика и аккуратность интерпретаций

Ключевые претензии касались:

- недостаточно полного линейного анализа;
- необоснованного использования термина Hopf;
- слабой связи с first-order comparison;
- чрезмерной уверенности в tail interpretation и neural analogy. [file:167]

Ответная стратегия:

- усиление Section 3;
- везде замена языка на more conservative;
- явное вынесение limitations в Discussion;
- separate statement, что first-order model — это future work, а не выполненный benchmark. [file:167]

### Reviewer #3: вычислительная строгость

Основной удар пришёлся по numerical methods:

- требование finer meshes;
- требование convergence of amplitude/period/tails/drift;
- недоверие к phase diagram как к полноценной bifurcation map. [file:167]

Ответная стратегия была не «симулировать всё подряд», а развести claims по уровням надёжности:

- Table 3 и Table 4 подтверждают robustness по времени, домену и seed perturbations при \\(\Delta x=1\\); [file:167]
- Table 2 честно показывает, что coarsest grid qualitatively wrong, а finer-than-1 explicit runs fail early; [file:167][file:177]
- phase map интерпретируется как operational regime survey for a fixed initial condition, not a complete phase diagram. [file:167]

## Самая важная численная находка после ревизии

После обсуждения критики мы отдельно проверили, что происходит при \\(\Delta x=0.5\\), если уменьшать \\(\Delta t\\) в 8 раз. В файле `total-2.json` четыре прогона `DT05-0`–`DT05-3` показали почти одинаковое время сбоя: примерно 23.505–23.511, и во всех случаях причиной был `v_bound`. [file:170]

Это чрезвычайно важный результат. Он означает, что early breakdown при \\(\Delta x=0.5\\) не является просто тривиальным time-step artifact или банальным CFL violation, который исчезнет при уменьшении \\(\Delta t\\). Наоборот, failure time практически \\(\Delta t\\)-independent, что указывает на grid-scale growth / short-wavelength instability в поле \\(v\\) и делает проблему структурной для данной discretized model, а не только алгоритмической для time integrator. [file:170]

### Практическая интерпретация этой находки

Это не «доказательство специфичности» модели в строгом математическом смысле, но это сильный диагностический аргумент, что в системе может существовать нетривиальная short-scale dynamics, не редуцируемая к обычной жалобе на explicit scheme. Такая картина особенно правдоподобна в режиме \\(D_\psi=0\\), где у ингибитора нет собственного диффузионного сглаживания, а поле \\(v=\partial_t\psi\\) питается от \\(\Delta\phi\\). [file:175][file:170][file:167]

Для Discussion это можно оформлять как: the breakdown at fine mesh is not primarily attributable to time-step size, since the failure time remains essentially unchanged under an eightfold reduction of \\(\Delta t\\); this suggests unresolved short-wavelength growth associated with the structure of the \\(D_\psi=0\\) model and the explicit discretization of the wave-mediated inhibitor dynamics. [file:170][file:167]

## Что показал D_psi control

Дополнительные manifest-файлы с `D_psi` control важны для следующего этапа. В `processed_dpsi_control_manifest.json` видно, что при fine grid \\(\Delta x=0.5\\) несколько запусков `dpsi_fine_*` всё равно остаются numerically unstable с `v_bound`, причём failure times лежат примерно около 25.28, а в одном случае около 26.57. [file:177]

При этом runs `dpsi_resolved_*` на \\(\Delta x=1.0\\) завершаются корректно и классифицируются как `breathing`, с типичными периодами порядка 5.56–6.15 и выраженным FFT peak. Это подтверждает, что рабочий resolved regime на \\(\Delta x=1\\) реален, но finer-mesh pathology полностью не исчезла. [file:177]

Отдельный `processed_dpsi_addition_manifest.json` содержит по крайней мере два resolved reference-style запуска на \\(\Delta x=1\\), оба breathing. Это полезно как внутренняя проверка воспроизводимости baseline signal. [file:183]

## Текущий честный вывод по численной части

На данный момент можно утверждать следующее:

- \\(\Delta x=2\\) qualitatively inadequate, так как меняет regime outcome по сравнению с \\(\Delta x=1\\); [file:167]
- при \\(\Delta x=1\\) single-spot breathing robust по \\(\Delta t\\), domain size и random seed perturbations; [file:167]
- при \\(\Delta x=0.5\\) early instability сохраняется даже при многократном уменьшении \\(\Delta t\\), так что проблема не сводится к простому CFL-fix; [file:170]
- therefore, полноценная spatial convergence below \\(\Delta x=1\\) для первой статьи не доказана; [file:167][file:177]
- но есть серьёзное основание считать fine-grid breakdown самостоятельным объектом исследования, особенно для второй статьи. [file:170][file:175]

## Как правильно позиционировать это в первой статье

Для первой статьи цель не должна смещаться в сторону заявления «мы доказали сложную неклассическую PDE-специфику». Это будет слишком сильно и легко атакуется. Безопасная и продуктивная стратегия такая:

- сохранить claims первой статьи локальными: breathing localised states, interaction effects, operational robustness at \\(\Delta x=1\\); [file:167]
- включить dt-independence fine-mesh breakdown как дополнительный numerical diagnostic, если он будет внесён в текст; [file:170]
- трактовать unresolved fine-grid issue как limitation of current modelling/numerical framework and as motivation for future work. [file:167]

То есть для CNSNS это должно звучать не как «classical PDE methods fail because the phenomenon is deep», а как «the observed fine-grid breakdown is not removed by time-step refinement and may reflect short-wavelength dynamics specific to the present model/discretization; clarifying this requires a dedicated study beyond the scope of the present paper». [file:170][file:167]

## Как развивать тему во второй статье

Идеи competition between activator and inhibitor fields, spontaneous structure formation, nucleation, crystallization points и emergence from noisy/full-field initial states лучше развивать уже во второй статье. Для этого у проекта есть техническая база: full-field noise runs, attractor-driven runs (`triadattractors`, `sinkbarrier`, `channelring`) и large locale/json outputs для структурного анализа. [file:175]

Во второй статье важно будет не ограничиться метафорами, а ввести operational definitions:

- что такое nucleation event;
- что такое crystallization point;
- что считать отдельной структурой;
- как измерять birth rate, lifetime, merger, annihilation, drift, cluster formation;
- как отличать long transient from asymptotic structured regime. [file:175]

Хорошая логика второй статьи:

1. Ввести field-wide initial conditions без жёстко заданного spot.
2. Определить детектор локальных структур на пороге или по connected components.
3. Построить кинетику числа структур во времени.
4. Измерять spatial pair-correlation, nearest-neighbour distance, cluster-size distribution.
5. Проверить роль \\(\varepsilon\\), `pump/sink` architecture, noise amplitude и \\(D_\psi\\).
6. Отдельно диагностировать режимы, где solver показывал dt-independent fine-grid breakdown. [file:175][file:170]

## Рабочая гипотеза о «специфичности» модели

Если нужен единый conceptual frame для будущей работы, его можно формулировать так:

> Специфичность модели заключается не просто во втором порядке по времени, а в сочетании wave-mediated inhibitory memory, отсутствия собственного diffusive smoothing при \\(D_\psi=0\\), сильной нелинейной bistable activator dynamics и пространственного источника \\(\Delta\phi\\) в уравнении на \\(v\\). Это сочетание может порождать локализованное breathing, separation-sensitive interactions и short-scale growth phenomena, не редуцируемые к стандартной first-order activator–inhibitor intuition.

Эта гипотеза пока не доказана как theorem, но она уже поддерживается набором численных симптомов: cubic temporal spectrum, resolved breathing at \\(\Delta x=1\\), two-spot nontrivial outcomes, transient drift и dt-independent fine-grid breakdown. [file:167][file:170][file:177]

## Что делать дальше: приоритеты

### Приоритет A: минимально усилить первую статью

1. Исправить оставшиеся текстовые мелочи в рукописи, если ещё не исправлены, особенно численные несогласованности типа \\(k_c\\) для \\(\varepsilon=3\\). [file:167]
2. Ослабить overly strong sentence в abstract про “validated against spatial resolution”, если он ещё не ослаблен. [file:167]
3. Если есть возможность, добавить в manuscript короткий dt-sweep diagnostic paragraph по \\(\Delta x=0.5\\) с опорой на `total-2.json`. [file:170]
4. Не раздувать claims beyond available evidence. [file:167]

### Приоритет B: прямой first-order control

Если будет время, крайне полезно сделать хотя бы один matched illustrative comparison с genuine first-order inhibitor model из Eq. (16). Это не обязан быть полный phase diagram; уже одна figure с \\(\phi_c(t)\\) и profile comparison при одинаковом initial disk была бы сильным аргументом против претензии “you did not show what is specific to second-order dynamics”. [file:167]

Но такой comparison надо делать только при явно описанном matching rule между параметрами second-order и first-order model. Нельзя просто подставить произвольные \\(\tau_\psi, D_\psi^{(1)}, \mu_\psi\\) и назвать это fair comparison. [file:167]

### Приоритет C: dedicated second article

Для второй статьи приоритетнее не continuation, а правильно поставленная статистика структур в поле:

- noise-driven emergence;
- attractor-controlled nucleation;
- competition and coarsening;
- lattice-like or crystallization-like organisation;
- sensitivity to \\(\varepsilon\\), forcing geometry и inhibitor regularization. [file:175]

## Полезные файлы и зачем они нужны

### Рукопись

- `cnsns_autosoliton_wave-v4.tex` — основная текущая версия первой статьи. [file:167]
- `cnsns_autosoliton_wave-v4-2.pdf` — соответствующий PDF. [file:166]

### Solver и анализ

- `sim_framework.py` — основной solver/framework с examples и hooks. [file:175]
- `analysis_validation.py` — скрипт анализа validation runs. [file:168]
- `glider_search.py` — вспомогательный поиск pair/drift-like trajectories. [file:141]
- `make_phase_diagram.py` — построение phase/regime map. [file:156]

### Manifest- и diagnostic-файлы первой статьи

- `processed_first-global-screen_manifest.json` — сводка baseline/short-long checks; новая версия уже исправляет nonsense frequency/period in unresolved short runs. [file:98]
- `processed_long_sensetivity_manifest*.json` — long-time sensitivity checks. [file:99][file:100][file:101]
- `processed_mesh_doagnostic_manifest-4.json` — mesh diagnostic summary. [file:102]
- `threshold_sweep-*.csv` — threshold sensitivity для классификатора. [file:103][file:104][file:105]
- `drift_threshold_comparison-2.json` и `drift_threshold_twospot-3.txt` — вспомогательные данные по two-spot drift classification. [file:149][file:151]

### Новые fine-grid diagnostics

- `total-2.json` — ключевой файл с dt-sweep at \\(\Delta x=0.5\\); главная находка: failure time почти не зависит от \\(\Delta t\\). [file:170]
- `all_tracks.txt` — треки для набора fine-grid прогонов. [file:171]
- `track_end-3.csv` — финальный фрагмент одного из unstable tracks. [file:174]
- `processed_dpsi_control_manifest.json` — сводка по D_psi control runs. [file:177]
- `processed_dpsi_fine_early_manifest-2.json` — дополнительный early-stage manifest fine-grid runs. [file:178]
- `processed_dpsi_addition_manifest.json` — дополнительная пара resolved reference-style runs. [file:183]

## Рекомендованный prompt для нового чата

Если этот проект переносится в новый чат, полезно сразу давать такой стартовый контекст:

> Я работаю над PDE-проектом по activator–wave-field system, где \\(\phi\\) — activator, а \\(\psi\\) — inhibitor с second-order damped wave dynamics. Первая статья для CNSNS уже почти готова и посвящена breathing autosolitons, linear stability of the homogeneous state, two-spot interactions and transient drift. Главные ограничения: нет continuation/localized spectrum, нет полноценного first-order benchmark, а fine-grid runs при \\(\Delta x=0.5\\) показывают dt-independent breakdown in \\(v\\), so this may reflect short-wavelength dynamics rather than a trivial CFL issue. У меня есть solver `sim_framework.py`, manifests по validation и новая идея второй статьи про full-field structure formation, nucleation and crystallization-like organisation. Помоги продолжить работу, не теряя этот контекст.

## Быстрый operational checklist

Перед следующим этапом работы полезно каждый раз проверять:

- Что именно я сейчас хочу доказать: локальную устойчивость claim, solver pathology или новую физику структуры? [file:167][file:175]
- Утверждение относится к первой статье или уже ко второй? [file:167][file:175]
- Есть ли у меня operational metric для наблюдаемого феномена? [file:98][file:170]
- Не пытаюсь ли я выдать exploratory numerical evidence за строгий PDE result? [file:167]
- Не смешал ли я limitation solver с limitation model, не имея диагностического теста? [file:170][file:177]

## Краткий итог

Состояние проекта на сейчас такое: первая статья стала существенно более аккуратной и defensible после ревизии текста и анализа, а новый dt-sweep на fine mesh дал важнейший диагностический аргумент, что observed breakdown не является простым артефактом выбора шага времени. Это не снимает вычислительные ограничения статьи, но превращает слабое место в осмысленный исследовательский вопрос. На этой базе можно либо аккуратно доподать первую статью, либо параллельно запускать вторую — уже про emergence of structures in the field, competition, nucleation и crystallization-like dynamics. [file:167][file:170][file:177][file:175]
