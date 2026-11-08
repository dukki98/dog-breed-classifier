# Faza 0: Okruženje i provera GPU-a

## Cilj

Pripremiti okruženje na laptopu i proveriti da PyTorch koristi GPU i podržava BF16,
pre nego što krenemo sa podacima i treningom.

## Polazno stanje

| | |
|---|---|
| OS | Windows 10 (build 19045) |
| GPU | NVIDIA GeForce RTX 3060 Laptop, 6 GB VRAM (6144 MiB), do 95 W |
| Arhitektura | Ampere (GA106), compute capability 8.6 |
| Drajver | 610.88, podržava CUDA do 13.3 |
| Python | 3.13.17 |
| Git | 2.52.0 |

## Odluke

### 1. Python 3.13 umesto 3.11 ili 3.12

- PyTorch na Windows-u podržava Python od 3.10 do 3.14, tako da bi i 3.11 radio.
- Problem su naučne biblioteke (NumPy, pandas, scikit-learn, matplotlib). One po pravilu
  [SPEC 0](https://scientific-python.org/specs/spec-0000/) podržavaju verziju Pythona oko 3 godine od izlaska.
- Za 3.11 je taj rok istekao krajem 2025, a za 3.12 u oktobru 2026. Na starijem Python-u pip ne prijavljuje grešku,
  nego bez upozorenja instalira starije verzije tih biblioteka.
- 3.13 je u roku podrške do oktobra 2027 i dobija sigurnosne zakrpe do oktobra 2029.
- 3.14 je noviji, ali je veća šansa da neki paket (npr. za INT8 u fazi 6) za njega još nema verziju.

### 2. PyTorch 2.14.1 + torchvision 0.29.1, build za CUDA 13.2

- **CUDA 13.2:** drajver podržava CUDA do 13.3, pa biramo najnoviji build koji nije noviji od toga.
  Postoji i build za 13.4, ali je noviji nego što drajver podržava.
- **Zašto ne najnoviji torch 2.15.0:** za njega još ne postoji odgovarajući torchvision (0.30).
- **Zašto tačne verzije:** torchvision 0.29.1 u svojim zavisnostima traži samo `torch>=2.14.0`.
  Bez navedenih verzija pip bi spojio torch 2.15.0 i torchvision 0.29.1. To je par koji nije pravljen
  zajedno i može da pokvari torchvision-ove C++ operacije (npr. `nms`).
- **Zašto poseban indeks:** PyPI za Windows nudi samo CPU verziju PyTorch-a. Verzije sa CUDA-om su na
  `download.pytorch.org`.
- Proveren je dry-run za Windows i Python 3.13: instaliraju se `torch-2.14.1+cu132`,
  `torchvision-0.29.1+cu132` i `numpy-2.5.4`.

### 3. Bez instalacije CUDA Toolkit-a

„CUDA Version“ u `nvidia-smi` je najveća verzija koju drajver podržava, a ne nešto što je instalirano.
PyTorch donosi svoj CUDA runtime i cuDNN, pa CUDA Toolkit treba samo za pisanje sopstvenih CUDA ekstenzija.

### 4. Projekat u `C:\dev`, van OneDrive-a

- Desktop i Documents se na Windows-u često sinhronizuju sa OneDrive-om.
- Ne želimo da se sinhronizuju gigabajti slika, checkpoint-i i `.venv`.
- Kratka putanja smanjuje i probleme sa ograničenjem od 260 znakova na Windows-u.

### 5. Šta ne ide u git

`.gitignore` je GitHub-ov šablon za Python (pokriva `.venv`, `__pycache__`, …), dopunjen sa
`data/`, `checkpoints/` i `runs/`. To su podaci, težine modela i logovi treninga: preveliki su i mogu da se
ponovo naprave.

## Koraci (Windows, cmd)

```bat
mkdir C:\dev
cd C:\dev
git clone https://github.com/dukki98/dog-breed-classifier.git
cd dog-breed-classifier
git switch faza-0-setup

python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-torch.txt

python scripts\check_gpu.py
```

`.venv` se aktivira u svakom novom terminalu. Kad je aktivan, na početku reda piše `(.venv)`.

## Provera GPU-a: `scripts/check_gpu.py`

Skripta ispisuje verzije, ime GPU-a, VRAM i podršku za BF16, a zatim meri množenje dve matrice 4096×4096
u tri preciznosti:

| Preciznost | Kako se računa |
|---|---|
| FP32 | obična CUDA jezgra, mantisa od 23 bita |
| TF32 | tensor jezgra. Ulaz je FP32, ali se množi sa mantisom od 10 bitova. Opseg je isti kao kod FP32. |
| BF16 | tensor jezgra. Podaci su 16-bitni (mantisa od 7 bitova, isti opseg kao FP32), a sabira se u FP32. |

Dve stvari na koje treba paziti pri merenju na GPU-u:
- **Zagrevanje:** prva pokretanja su sporija (inicijalizacija, cuBLAS bira algoritam), pa ih ne merimo.
- **Asinhronost:** Python samo zakaže posao na GPU-u i odmah nastavlja. Bez `torch.cuda.synchronize()`
  pre i posle merenja izmerili bismo samo zakazivanje, a ne računanje.

### Hipoteza pre merenja

Iz specifikacije Ampere GeForce čipova (GA10x), teoretski maksimumi stoje ovako:
- TF32 na tensor jezgrima ima **približno isti** maksimum kao FP32 na običnim jezgrima.
- BF16 (sa sabiranjem u FP32) ima oko **2×** veći maksimum.

Zato očekujemo da TF32 bude tek nešto brži od FP32, a BF16 oko 2× brži.
(Kod data-centar kartica kao A100 razlika je mnogo veća. Ovo je specifičnost GeForce serije.)

### Rezultati

> Biće popunjeno posle pokretanja na laptopu, na punjaču.

| Stavka | Vrednost |
|---|---|
| PyTorch / CUDA build | — |
| VRAM ukupno / slobodno | — |
| BF16 podrška | — |
| FP32 | — TFLOPS |
| TF32 | — TFLOPS (—× u odnosu na FP32) |
| BF16 | — TFLOPS (—× u odnosu na FP32) |

**Zaključak:** —

## Pitanja za proveru razumevanja

1. Šta znači „CUDA Version: 13.3“ u `nvidia-smi` i zašto zbog toga ne instaliramo CUDA Toolkit?
2. Zašto u `requirements-torch.txt` navodimo tačne verzije torch-a i torchvision-a?
3. Šta bi izmerila skripta da nema `torch.cuda.synchronize()`?
4. Zašto je BF16 bolji izbor za trening od FP16 na ovoj kartici?
   (Pomoć: opseg brojeva i GradScaler.)
5. Zašto TF32 na RTX 3060 donosi manje ubrzanja nego na A100?

## Sledeći korak

Faza 1: preuzimanje Kaggle skupa podataka, analiza (raspodela rasa, veličine slika) i podela na train i val.
