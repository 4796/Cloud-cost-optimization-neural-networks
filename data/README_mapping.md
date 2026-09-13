# Poreklo i dekodiranje dataseta

## Izvor

[IBM/multi-cloud-configuration-dataset](https://github.com/IBM/multi-cloud-configuration-dataset),
prateći dataset uz rad Lazuka et al., *"Search-based Methods for
Multi-Cloud Configuration"*, IEEE CLOUD 2022 (arXiv:2204.09437).

Originalni CSV (`multi-cloud-configuration.csv`) anonimizuje cloud
provajdera (A/B/C) i njegove konfiguracione parametre (family/type/vcpu)
pomoću label-encoding-a (celobrojni kodovi 0/1/2 bez objašnjenja šta
predstavljaju), da bi se sprečila konkurentska analiza cena provajdera.

## Postupak dekodiranja

Pošto je `target_cost` konstruisan kao
`cost = runtime_h × cena_po_satu × nodes` (navedeno u originalnom README-u),
za svaku (provider, family, type, vcpu) konfiguraciju izračunata je
implicitna cena po satu:

```
implied_price = target_cost / (target_runtime_h × nodes)
```

Ova vrednost je, unutar svake konfiguracije, konstantna (std ≈ 0,
razlika je samo floating-point šum) — što potvrđuje da smo tačno
rekonstruisali skrivenu satnicu za svih 88 konfiguracija
(`scripts/decode_pricing.py`).

Konfiguracije su zatim rangirane po ceni unutar svakog provajdera i
upoređene sa poznatim hijerarhijama cena realnih instanci:

- **AWS (A)**: family kodovi odgovaraju `c4 < m4 < r4` (po ceni, pri
  istom vcpu kodu), vcpu kod `0/1` = `large/xlarge` (dupliranje cene i
  broja jezgara potvrđuje ovo tačno: xlarge cena = 2× large cena za
  svaku family).
- **Azure (B)**: `D_v3 < D_v2` po ceni (v3 serija ima hyperthreaded
  vCPU-ove pa je jeftinija po vCPU-u), vcpu kod `0/1` = `2/4 vCPU`.
- **GCP (C)**: `e2 < n1` po ceni (e2 je noviji, cost-optimized custom
  machine family), type kod `0/1/2` = `standard/highcpu/highmem`
  (highcpu najjeftiniji, highmem najskuplji — konzistentno u obe
  family grupe), vcpu kod `0/1` = `2/4 vCPU`.

**Napomena o pouzdanosti**: redosled porodica (koja vrednost je "0" a
koja "1"/"2") je rekonstruisan na osnovu relativnog poretka cena i
opšte poznatih cenovnih hijerarhija, ne iz zvaničnog izvora koji
eksplicitno navodi mapiranje. Za GCP deo (jedini koji se koristi u
finalnom datasetu — videti niže) uklapanje je posebno čvrsto: 6
kombinacija (family×type) unutar vcpu=0 grupe imaju striktno rastući
redosled cena koji se **identično ponavlja** za obe family vrednosti
(highcpu < standard < highmem), što je jak signal da je dekodiranje
tačno.

## Zašto samo GCP (provider C)?

Cilj projekta je da se provajder potpuno ignoriše i da model uči
isključivo iz (vCPU, RAM, broj čvorova, workload). Kad se to uradi na
sva tri provajdera zajedno, ~12% kombinacija (vcpu, ram, nodes,
workload) se poklapa preko različitih provajdera (npr. i AWS m4.large
i Azure D2_v3 i GCP e2-standard-2 su 2 vCPU / 8 GB RAM), a njihov
`target_cost` se razlikuje i do 240% (prosečno ~140%) jer cena po satu
zavisi pretežno od provajdera/pricing politike, a ne od hardvera.
`target_runtime` je mnogo stabilniji (~30% prosečno) jer zavisi od
realne računarske snage.

Rešenje: dataset je ograničen na **provider C (GCP)**, koji ima
najviše redova (1486 od 2507 sirovih / 1282 od 2202 uspešnih), i kod
kog je mapiranje (vcpu, ram_gb) → instance_type **bijektivno** — svih
12 GCP instance tipova ima različitu RAM vrednost, pa nema kolizija
(potvrđeno programski, 0/1237). Time (vcpu, ram) postaju čist,
determinisan zamenik za "koja instanca", i cost/runtime su čisti,
nezašumljeni target-i.

## Finalno mapiranje (GCP)

| instance_type    | vcpu | ram_gb |
|------------------|------|--------|
| e2-standard-2    | 2    | 8.0    |
| e2-standard-4    | 4    | 16.0   |
| e2-highcpu-2     | 2    | 2.0    |
| e2-highcpu-4     | 4    | 4.0    |
| e2-highmem-2     | 2    | 16.0   |
| e2-highmem-4     | 4    | 32.0   |
| n1-standard-2    | 2    | 7.5    |
| n1-standard-4    | 4    | 15.0   |
| n1-highcpu-2     | 2    | 1.8    |
| n1-highcpu-4     | 4    | 3.6    |
| n1-highmem-2     | 2    | 13.0   |
| n1-highmem-4     | 4    | 26.0   |

## Finalni dataset

`data/cloud_cost_dataset_gcp.csv` — 1282 reda (samo `status == ok`),
kolone: `vcpu, ram_gb, nodes, dataset, task, target_cost,
target_runtime, instance_type` (`instance_type` je referentna kolona,
ne koristi se kao model feature jer je potpuno određena sa vcpu+ram).

- `nodes`: 2-5
- `dataset`: `buzz`, `creditcard`, `santander`
- `task`: 10 Dask ML operacija (kmeans, logistic_regression, ...)
- pokriveno 28 od mogućih 30 (dataset, task) kombinacija (2 nedostaju,
  verovatno usled kvota providera pri prikupljanju originalnog
  dataseta — navedeno u originalnom README-u)

## Reprodukcija

```
python scripts/decode_pricing.py   # rekonstrukcija implicitne cene, provera dekodiranja
python scripts/build_dataset.py    # gradi data/cloud_cost_dataset_gcp.csv
```
