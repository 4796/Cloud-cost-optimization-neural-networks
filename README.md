# Primena neuronskih mreža za predviđanje i optimizaciju troškova na cloud-u (Cloud cost optimization)



Kompletan tok rada (priprema podataka, treniranje, analiza, evaluacija) nalazi se u jednom Jupyter notebook-u: [`cloud_cost_optimization.ipynb`](cloud_cost_optimization.ipynb). Ovaj README sumira projekat po traženoj strukturi; za pun kod i objašnjenja korak-po-korak videti notebook.

## Kako pokrenuti

```
pip install numpy pandas matplotlib torch scikit-learn jupyter
jupyter notebook cloud_cost_optimization.ipynb
```

Notebook se oslanja isključivo na fajlove iz `data/` (uključeni u repozitorijum, nema potrebe za internet konekcijom pri pokretanju) i pokreće se iz korenskog direktorijuma projekta (koristi relativne putanje `data/...`).

---

## 1. Opis problema

Pri pokretanju posla u cloud okruženju (npr. treniranje ML modela, obrada podataka), korisnik bira konfiguraciju resursa - tip virtuelne mašine (broj vCPU-a, količina RAM-a) i broj čvorova u klasteru. Ova odluka direktno određuje dve suprotstavljene veličine: **trošak** izvršavanja i **vreme izvršavanja** (latenciju).

Cilj projekta je dvojak:

1. **Predviđanje** - neuronske mreže koje na osnovu opisa posla (tip zadatka, ulazni dataset) i konfiguracije resursa (vCPU, RAM, broj čvorova) predviđaju očekivani trošak i vreme izvršavanja.
2. **Optimizacija** - korišćenje istreniranih modela kao "orakla" unutar algoritma lokalne (greedy) pretrage koji za dati posao i ograničenje (npr. "završi za manje od X sekundi") pronalazi konfiguraciju koja minimizuje trošak, bez iscrpnog isprobavanja svih konfiguracija.

## 2. Podaci

**Izvor**: [IBM multi-cloud-configuration dataset](https://github.com/IBM/multi-cloud-configuration-dataset) (Lazuka et al., IEEE CLOUD 2022, arXiv:2204.09437) - 88 konfiguracija resursa preko 3 cloud provajdera, izmerene na 30 kombinacija (10 Dask ML zadataka × 3 ulazna dataseta). Provajder i njegovi konfiguracioni parametri su u originalnom CSV-u anonimizovani label-encoding-om.

**Dekodiranje**: anonimizacija je rešena empirijski - implicitna satnica po konfiguraciji (`cost / (runtime_h × nodes)`) je konstantna (std≈0), pa je rangiranje tih vrednosti upoređeno sa poznatim javnim cenovnicima realnih instanci da se odredi šta kodovi 0/1/2 predstavljaju. Puna metodologija: [`data/README_mapping.md`](data/README_mapping.md); reprodukcija u notebook-u, sekcija 2.2.

**Identitet provajdera** (A/B/C) otkriven je iz same strukture parametara, bez računanja cena: provider A ima 3×2=6 kombinacija (family×vcpu) - poklapa se samo sa AWS ponudom (m4/r4/c4 × large/xlarge); provider B ima 2×2=4 - Azure (D_v2/D_v3 × 2/4 vCPU); provider C ima 2×3×2=12 - GCP, jedini sa dodatnom "tip mašine" dimenzijom (e2/n1 × standard/highcpu/highmem × 2/4 vCPU). Koja tačno vrednost (0/1) odgovara kojoj familiji/tipu utvrđeno je poređenjem redosleda cena sa poznatim cenovnicima - puna provera u `data/README_mapping.md`.

**Ključna odluka - samo GCP**: da bi model učio isključivo iz (vCPU, RAM) bez obzira na provajdera, ideja je bila izbaciti kolonu provajdera iz sva 3 dekodirana provajdera. Problem: različiti provajderi dele iste (vcpu, ram) kombinacije (npr. i AWS m4.large i Azure D2_v3 i GCP e2-standard-2 su 2 vCPU/8GB RAM) sa drastično različitom cenom, jer cena zavisi od pricing politike provajdera, ne (samo) hardvera. Zato je dataset ograničen na **GCP (provider C)**, koji ima i najviše redova (1486/2507 sirovih, 1282/2202 uspešnih od sva 3 provajdera) - i kod kog `(vcpu, ram_gb)` bijektivno određuju instancu (0 kolizija na 1237 kombinacija, potvrđeno u notebook-u), pa su cost/runtime tamo čisti, nezašumljeni target-i. Detalji: notebook, sekcija 2.2–2.3.

**Finalni dataset**: `data/cloud_cost_dataset_gcp.csv` - 1282 reda (samo uspešna izvršavanja), kolone `vcpu, ram_gb, nodes, dataset, task, target_cost, target_runtime, instance_type` (poslednja je referentna, ne koristi se kao model feature - potpuno je određena sa vcpu+ram_gb). Pokriva 12 realnih GCP instanci (e2/n1 × standard/highcpu/highmem × 2/4 vCPU), `nodes ∈ {2,3,4,5}`, 28 od mogućih 30 (dataset, task) kombinacija.

**EDA**: oba targeta (`target_cost`, `target_runtime`) su jako desno-skewed. Koristi se **`log`** transformacija, ne `log1p` - `target_cost` ima vrednosti dosta manje od 1 (0.0002–0.17), a `log1p(x) = ln(1+x) ≈ x` upravo za tako male vrednosti, pa bi `log1p` na cost-u skoro ništa ne promenio. Pošto su obe mete striktno pozitivne (nema nula), obična `log` transformacija je ispravan izbor. Detalji: notebook, sekcija 2.4.

**Preprocesiranje**: `log` transformacija oba targeta (razlog iznad), `StandardScaler` na numeričke ulaze (`vcpu, ram_gb, nodes`, fit samo na train), one-hot na `task`/`dataset`. Split 70/15/15 (train/val/test), **grupisan** po `(instance_type, nodes, task, dataset)` ključu (`GroupShuffleSplit`) da ponovljena merenja istog config-a (45/1237 kombinacija, 3.6%, ~7% redova) ne cure između split-ova - u notebook-u je i eksperimentalno potvrđeno da bi obično nasumičan split doveo do merljivog "curenja" (test redovi sa blizancem u train-u imaju ~2x manju grešku predikcije od ostalih). Detalji: notebook, sekcija 2.5.

## 3. Arhitektura modela

Dva odvojena MLP regresora (PyTorch) - jedan za `target_cost`, jedan za `target_runtime` - umesto jednog multi-output modela (izbegava balansiranje gubitaka različitih skala):

```
Linear(input_dim, 64) → ReLU → Dropout(0.2) → Linear(64, 32) → ReLU → Linear(32, 1)
```

`input_dim = 16` (3 numerička + 10 one-hot task + 3 one-hot dataset). Izlazni sloj bez aktivacije (regresija na `log(target)`). Detalji: notebook, sekcija 3.

## 4. Trening

Adam optimizator, `MSELoss` na `log`-transformisanom targetu, mini-batch trening (`batch_size=32`). **Early stopping**: prati validacioni gubitak, čuva težine najboljeg modela, prekida posle `patience=15` epoha bez poboljšanja (`max_epochs=250`, dovoljno visoko da early stopping praktično uvek prekine ranije). Detalji i krive učenja: notebook, sekcija 4.

## 5. Analiza osetljivosti i hiperparametarska optimizacija

**Hiperparametarska pretraga**: manji, ručno biran skup od 8 kombinacija (veličine skrivenih slojeva, dropout, learning rate), evaluiranih po validacionom gubitku na cost modelu; ista konfiguracija se primenjuje i na runtime model. Namerno ograničen obim (ne iscrpna/automatizovana pretraga poput Optuna) - dovoljan za demonstraciju uticaja hiperparametara na ovom malom datasetu. Najbolja pronađena konfiguracija (jedan konkretan run): `hidden1=64, hidden2=32, dropout=0.1, lr=0.001` (val_loss≈0.095); pun rang svih 8 kombinacija: notebook, sekcija 5.1 (brojevi mogu blago varirati između pokretanja zbog nedeterminizma treninga).

**Analiza osetljivosti**: permutation importance na test skupu, isključivo `vcpu, ram_gb, nodes, task, dataset` - **bez** `instance_type`, koji je potpuno redundantan sa vcpu+ram_gb i bi iskrivio rezultat (videti Diskusija). Detalji: notebook, sekcija 5.2.

## 6. Rezultati evaluacije

**Tačnost modela na test skupu** (jedan konkretan run; tačne vrednosti variraju blago između pokretanja zbog nedeterminizma treninga - videti notebook, sekcija 6.1, za tekuće vrednosti):

| Model | R² | MAE | RMSE |
|---|---|---|---|
| COST | 0.844 | 0.00276 | 0.00630 |
| RUNTIME | 0.930 | 11.13 | 27.50 |

**Optimizacija (6.2)** - constraint-based greedy/lokalna pretraga (hill climbing, 3 nasumična restarta, graf suseda po hardverskoj semantici) nad prostorom od 12 GCP instanci × `nodes∈{2,3,4,5}` = 48 kandidata. Primer rezultata za 4 test-workload-a (pragovi izabrani tako da budu realno dostižni za dati posao - videti napomenu ispod):

| workload | prag (s) | predložena konfiguracija | pred. cost | pred. runtime | izvodljivo | poziva modela |
|---|---|---|---|---|---|---|
| kmeans/santander | 90 | n1-highcpu-4, 4 nodova | 0.0147 | 88.62 | **da** | 35 |
| logistic_regression/buzz | 150 | n1-highcpu-4, 5 nodova | 0.0219 | 133.37 | **da** | 35 |
| xgboost/creditcard | 30 | n1-highcpu-2, 3 nodova | 0.0015 | 24.54 | **da** | 30 |
| naive_bayes/santander | 20 | n1-standard-2, 2 nodova | 0.0011 | 19.14 | **da** | 33 |

Svaki scenario je izvodljiv uz znatno manje od 48 poziva modela (30-35). Zanimljiv nalaz pri biranju pragova: za `naive_bayes/santander`, stvarni (izmereni) podaci sadrže konfiguraciju sa runtime-om od svega 6.9s (`e2-highmem-4`, 4 noda), ali **model** ni za jednu od 48 konfiguracija ne predviđa runtime ispod ~15s za taj posao - dakle prag ispod toga (npr. 10s) ispada neizvodljiv prema modelu bez obzira na pravu vrednost. Ovo je ograničenje tačnosti modela za taj konkretan posao, ne greška pretrage (greedy je ispravno pronašao najbolje što model uopšte predviđa), i vredi ga imati na umu pri tumačenju rezultata. Neizvodljiv slučaj (kad ni model ni pretraga ne mogu zadovoljiti prag) posebno je demonstriran u notebook-u za `logistic_regression/santander` sa pragom od 1s.

## 7. Diskusija

**Cost je teže predvideti nego runtime.** Pošto je `cost = runtime × cena_po_satu(instance) × nodes`, cost model mora implicitno da nauči dva efekta odjednom - koliko dugo posao traje na datom hardveru, i koliko taj hardver košta po satu (nelinearna, stepenasta funkcija tipa instance) - dok runtime model uči samo prvi efekat. To se i vidi u rezultatima iz 6.1 (runtime model ima viši R² na test skupu).

Ovo otvara pitanje da li je poseban cost model uopšte neophodan. Pošto je u ovom datasetu cena konstruisana upravo kao `runtime × poznata_cena_po_satu`, teorijski najprecizniji pristup bio bi da se trenira samo model za runtime, a cost izračuna analitički (`runtime_pred × cena_instance × nodes`) - bez ikakve dodatne greške na strani cost-a. Ipak, zadržavamo i eksplicitan cost model, jer u realnom cloud okruženju cena retko zavisi isključivo od runtime-a i cene po satu - tu su i troškovi prenosa podataka, popusti na rezervisane/spot instance, troškovi skladištenja i slično, koje analitička formula ne bi obuhvatila. Eksplicitan cost model je time na *ovom* datasetu manje precizan od pukog analitičkog izračuna, ali predstavlja opštiji pristup koji bi se bolje preneo na realističniji scenario u kom cena nije čista funkcija runtime-a.

**Ograničenja pristupa**:
- Dataset je ograničen na jedan cloud provajder (GCP) i 30 specifičnih (task, dataset) kombinacija - model se ne generalizuje na potpuno nove tipove poslova van ovog skupa.
- Na prostoru od 48 kandidata iscrpna pretraga bi i inače bila jeftina, i ne poredimo je eksplicitno sa greedy rešenjem (6.2) - ne možemo izmeriti koliko je pronađeno rešenje blizu pravog optimuma; oslanjamo se na hardverski smislen dizajn suseda i 3 nasumična restarta kao heurističku garanciju kvaliteta. Prava vrednost greedy pristupa bi se pokazala na mnogo većem prostoru (više provajdera/regiona/tipova instanci), gde iscrpna pretraga ne bi bila izvodljiva.
- 45 od 1237 (instance, nodes, task, dataset) kombinacija u datasetu su pokrenute dvaput - potpuno ista konfiguracija, ista mašina, isti posao. Runtime tih ponovljenih merenja se ipak razlikuje u proseku ~9% (medijalno ~6%), a u najgorem slučaju i do 63% (npr. `n1-highmem-2`, 5 nodova, `xgboost`/`santander`: 142.9s naspram 74.4s) - verovatno usled deljene cloud infrastrukture u trenutku merenja. Ovo postavlja gornju granicu tačnosti bilo kog modela: čak i savršen model ne može tačno da predvidi pojedinačno merenje kad sam podatak nosi ovoliki šum.
- Hiperparametarska pretraga je namerno ograničena na manji, ručno biran skup kombinacija, a ne iscrpna/automatizovana pretraga (npr. Optuna) - dovoljno za demonstraciju uticaja hiperparametara u okviru ovog projekta.

## 8. Zaključak

Projekat pokazuje da se, uz pažljivo rešavanje nekoliko praktičnih problema (dekodiranje anonimizovanog dataseta, izbor konzistentnog podskupa provajdera bez unakrsne kolizije cena, transformacija skewed targeta, izbegavanje redundantnih feature-a), neuronska mreža može naučiti da predvidi trošak i vreme izvršavanja cloud radnih opterećenja sa zadovoljavajućom tačnošću (videti 6.1). Istrenirani modeli, korišćeni kao orakl unutar constraint-based greedy pretrage, pronalaze jeftinu konfiguraciju resursa uz zadati vremenski prag koristeći znatno manje poziva modela nego što bi zahtevalo iscrpno isprobavanje svih 48 kombinacija - pristup dizajniran da se skalira na realnije, veće prostore konfiguracija (više provajdera, regiona i tipova instanci) gde bi iscrpna pretraga bila neizvodljiva.

---

## Struktura repozitorijuma

```
cloud_cost_optimization.ipynb   - glavni notebook (kompletan tok rada)
data/
  multi-cloud-configuration.csv - originalni (anonimizovani) dataset
  cloud_cost_dataset_gcp.csv    - finalni, dekodirani GCP-only dataset za modeliranje
  README_mapping.md             - puna metodologija dekodiranja anonimizacije
  README_dataset.md             - originalni README izvornog dataseta
scripts/
  decode_pricing.py             - pomoćna skripta, rekonstrukcija implicitne cene
  build_dataset.py              - pomoćna skripta, gradi cloud_cost_dataset_gcp.csv
```

---

## Korišćenje AI alata

U izradi projekta korišćen je AI asistent (Claude), i to prevashodno za zadatke koji ne zahtevaju samostalno domensko rasuđivanje ili donošenje odluka o projektu, već izvršavanje/podršku već donetih odluka:

- **Pretraga odgovarajućeg dataseta** - pomoć pri pronalaženju javno dostupnog dataseta pogodnog za problem predviđanja cloud troškova, pošto samostalna pretraga nije dala odgovarajući rezultat.
- **Komplikovano prilagođavanje i mapiranje podataka** - dekodiranje anonimizovanog dataseta (rekonstrukcija implicitne cene po satu, mapiranje anonimizovanih kodova na stvarne vCPU/RAM specifikacije realnih instanci).
- **Osmišljavanje algoritma za pronalaženje optimalnog rešenja** - dizajn constraint-based greedy/lokalne pretrage (definicija prostora kandidata, suseda, kriterijuma poređenja, ponašanja u neizvodljivom slučaju).
- **Otkrivanje grešaka** - identifikacija bagova i metodoloških propusta tokom razvoja (npr. curenje informacija kroz redundantan feature, neodgovarajuća transformacija targeta, greške u kodu).
- **Dosadni zadaci koji ne zahtevaju razmišljanje** - rutinski/mehanički delovi izrade (npr. pomoćni kod za vizualizacije, formatiranje, ponovljena podešavanja).
- **Sastavljanje opisa koda i ovog README fajla.**

Sve odluke koje zahtevaju znanje i razumevanje problema - izbor arhitekture i hiperparametara, definisanje cilja i ograničenja optimizacije, tumačenje i ocena ispravnosti rezultata, i sve druge suštinske odluke o projektu - donete su samostalno.
