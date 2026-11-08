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

Izmereno 8. 11. 2026, dok je laptop bio na punjaču (baterija 100%). Pre merenja GPU je bio slobodan
(0 MiB zauzeto, 0% opterećenja, 53 °C), a `nvidia-smi` je prijavio nametnut limit snage od 95 W.

Instalacija iz `requirements-torch.txt` je prošla bez grešaka. Instalirani su `torch-2.14.1+cu132`,
`torchvision-0.29.1+cu132` i `numpy-2.5.4`, kao u dry-run-u.

| Stavka | Vrednost |
|---|---|
| PyTorch / CUDA build | 2.14.1+cu132 / CUDA 13.2, cuDNN 9.24.0 (`92400`) |
| GPU | RTX 3060 Laptop, 30 SM jedinica, compute capability 8.6 |
| VRAM ukupno / slobodno | 6.00 GiB / 5.01 GiB (posle inicijalizacije CUDA-e) |
| BF16 podrška | da |

`scripts/check_gpu.py` je pokrenut tri puta zaredom (svako pokretanje je prosek 20 množenja):

| Preciznost | 1. pokretanje | 2. pokretanje | 3. pokretanje | U odnosu na FP32 |
|---|---|---|---|---|
| FP32 | 6.27 TFLOPS | 6.25 TFLOPS | 6.24 TFLOPS | 1.0× |
| TF32 | 12.47 TFLOPS | 12.21 TFLOPS | 12.21 TFLOPS | 2.0× |
| BF16 | 25.60 TFLOPS | 25.38 TFLOPS | 25.20 TFLOPS | 4.0–4.1× |

Pokretanja se razlikuju za najviše ~2%, pa su razlike između preciznosti stvarne, a ne šum.

### Poređenje sa hipotezom

| | Hipoteza | Izmereno |
|---|---|---|
| TF32 / FP32 | ~1× (tek nešto brže) | 2.0× |
| BF16 / FP32 | ~2× | 4.0–4.1× |

**Hipoteza nije potvrđena.** Odnos TF32 : BF16 = 1 : 2 odgovara specifikaciji, ali je FP32 oko 2× sporiji nego
što je hipoteza pretpostavila. Hipoteza je poredila teoretske maksimume, a to prećutno podrazumeva dve stvari:
da sve tri preciznosti rade na istom taktu i da svaka postiže isti deo svog maksimuma. Ovde ne važi ni jedno ni drugo.

Da bismo to proverili, izmerili smo takt GPU-a pod dužim opterećenjem. Privremena skripta (nije u repou) je
za svaku preciznost 8 s neprekidno množila matrice 4096×4096. Prve 2 s se ne računaju, jer se takt tada još
podešava. U preostalih ~6 s je 12 puta pročitala `nvidia-smi`: takt SM-a, snagu, temperaturu i razlog
ograničenja takta.

Teoretski maksimum pri izmerenom taktu = 30 SM × operacija po SM-u po taktu × takt. Broj operacija po SM-u
po taktu za GA10x je izveden iz NVIDIA-inog dokumenta o arhitekturi GA102: FP32 256 (128 jedinica × FMA,
tj. 2 operacije), TF32 256, BF16 sa sabiranjem u FP32 512.

| Preciznost | Protok (~6 s) | Takt SM, medijana (opseg) | Snaga | Teoretski maksimum pri tom taktu | Iskorišćenost |
|---|---|---|---|---|---|
| FP32 | 6.16 TFLOPS | 1260 MHz (1215–1275) | 94–95 W | 9.68 TFLOPS | 64% |
| TF32 | 12.09 TFLOPS | 1676 MHz (1665–1695) | 95 W | 12.87 TFLOPS | 94% |
| BF16 | 24.06 TFLOPS | 1687 MHz (1672–1702) | 95 W | 25.91 TFLOPS | 93% |

Temperatura je bila 67–80 °C. Razlog ograničenja takta je u svim merenjima bio `0x4` (SW Power Cap), dakle
limit snage, a nijednom temperatura.

Odstupanje dolazi od dva efekta koji se množe:

1. **Limit snage različito spušta takt za različite preciznosti.** GPU stalno radi na 95 W. FP32 na običnim
   jezgrima troši mnogo više energije po operaciji (95 W / 6.16 TFLOPS ≈ 15 pJ po operaciji, a za BF16
   95 W / 24.06 TFLOPS ≈ 4 pJ). Zato drajver za FP32 spušta takt na ~1260 MHz, a tensor jezgra drži na
   ~1680 MHz, što je oko 1.33× više.
2. **FP32 množenje ostaje daleko od svog maksimuma.** Tensor jezgra postižu 93–94% teoretskog maksimuma,
   a FP32 samo 64%. To nismo posebno merili, ali verovatan razlog je ovaj: na GA10x polovina FP32 jedinica
   deli putanju sa INT32 operacijama, a teoretski maksimum računa kao da sve jedinice rade FMA u svakom taktu.
   To daje još oko 1.47× u korist tensor jezgara.

Zajedno: TF32 1.33 × 1.48 ≈ 1.97 (izmereno 12.09 / 6.16 = 1.96), a BF16 2 × 1.34 × 1.46 ≈ 3.9
(izmereno 24.06 / 6.16 = 3.91).

Kratko merenje iz `check_gpu.py` daje za BF16 nešto više (25.2–25.6 TFLOPS) od dužeg merenja (24.06).
Verovatan razlog je to što u prvoj sekundi GPU još nije zagrejan i takt mu je viši.

**Zaključak:**

- PyTorch vidi GPU, CUDA build radi i BF16 je podržan. Okruženje je spremno.
- Množenje matrica u BF16 je ~4× brže od pravog FP32. To potvrđuje izbor BF16 za trening.
- TF32 na ovoj kartici daje ~2× za FP32 operacije koje ostanu van `autocast`-a. Da li u treningu uključiti
  `torch.set_float32_matmul_precision("high")`, odlučićemo kad uvodimo mešovitu preciznost (faza 3).
- Za trening je na raspolaganju ~5 GiB VRAM-a, a ne 6. Pre pokretanja `nvidia-smi` pokazuje 0 MiB zauzeto,
  a posle inicijalizacije CUDA-e slobodno je 5.01 GiB. Razliku zauzimaju CUDA kontekst i drajver.
- Brzina je ograničena snagom (95 W), pa zavisi od napajanja. Merenja brzine (posebno u fazi 6) raditi
  na punjaču i pod istim uslovima, da bi brojevi mogli da se porede.

## Pitanja za proveru razumevanja

1. Šta znači „CUDA Version: 13.3“ u `nvidia-smi` i zašto zbog toga ne instaliramo CUDA Toolkit?
2. Zašto u `requirements-torch.txt` navodimo tačne verzije torch-a i torchvision-a?
3. Šta bi izmerila skripta da nema `torch.cuda.synchronize()`?
4. Zašto je BF16 bolji izbor za trening od FP16 na ovoj kartici?
   (Pomoć: opseg brojeva i GradScaler.)
5. Zašto TF32 na RTX 3060 donosi manje ubrzanja nego na A100?
6. Hipoteza je predvidela da će BF16 biti ~2× brži od FP32, a izmereno je ~4×. Koja dva efekta to objašnjavaju
   i zašto je FP32 radio na nižem taktu iako je limit snage isti?
7. Koliko VRAM-a stvarno imamo za trening i zašto to nije punih 6 GB?

## Sledeći korak

Faza 1: preuzimanje Kaggle skupa podataka, analiza (raspodela rasa, veličine slika) i podela na train i val.
