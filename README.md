# Dog Breed Classifier

Model koji za fotografiju psa prepoznaje rasu i vraća **3 najverovatnije rase** od 120.

Pravljen je da radi na laptopu sa **RTX 3060 (6 GB VRAM)**:
- mali model (< 30M parametara)
- trening u mešovitoj preciznosti BF16
- inference u FP16, uz opcionu INT8 kvantizaciju

Projekat je rađen po fazama. Svaka faza ima svoj PR i dokument u [`docs/`](docs/) sa odlukama i izmerenim rezultatima.

## Status

| Faza | Opis | Status |
|---|---|---|
| 0 | Okruženje, provera GPU-a, struktura repoa ([docs](docs/00-setup.md)) | završena |
| 1 | Podaci: preuzimanje, analiza, podela na train/val/test ([docs](docs/01-podaci.md)) | završena |
| 2 | Polazni model: fine-tuning, top-1 i top-3 tačnost | — |
| 3 | Augmentacije, raspored stope učenja, BF16, praćenje eksperimenata | — |
| 4 | Pretraga hiperparametara (Optuna) | — |
| 5 | Analiza grešaka: koje rase model meša | — |
| 6 | Izvoz u ONNX, FP16/INT8, merenje brzine i VRAM-a | — |
| 7 | API ili demo koji vraća 3 najverovatnije rase | — |
| 8 | Praćenje modela u produkciji | — |

## Brzi početak (Windows, cmd)

Detaljna uputstva i objašnjenja su u [`docs/00-setup.md`](docs/00-setup.md) (okruženje)
i [`docs/01-podaci.md`](docs/01-podaci.md) (podaci).

```bat
git clone https://github.com/dukki98/dog-breed-classifier.git
cd dog-breed-classifier
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-torch.txt
python -m pip install -r requirements.txt
python scripts\check_gpu.py

rem Podaci: prvo prihvatiti pravila takmičenja na Kaggle-u (vidi docs/01-podaci.md)
kaggle auth login
kaggle competitions download -c dog-breed-identification -p data\raw
tar -xf data\raw\dog-breed-identification.zip -C data\raw
python scripts\analyze_data.py
```

## Struktura

```
dog-breed-classifier/
├── CLAUDE.md               uputstva za Claude (način rada, hardver, konvencije)
├── data/                   slike sa Kaggle-a (nije u git-u)
├── docs/                   dokument za svaku fazu: odluke, rezultati, naučeno
│   └── img/                grafikoni za dokumentaciju
├── scripts/
│   ├── check_gpu.py        provera GPU-a i brzine množenja matrica (FP32/TF32/BF16)
│   ├── analyze_data.py     analiza skupa: rase, veličine slika, duplikati
│   ├── image_hash.py       perceptualni hash (dHash) za pronalaženje istih slika
│   └── make_splits.py      podela na train/val/test (70/15/15), bez duplikata
├── splits/                 podela: train.csv, val.csv, test.csv, excluded.csv (u git-u)
├── requirements-torch.txt  PyTorch sa CUDA 13.2
└── requirements.txt        ostale zavisnosti (kaggle, pandas, scikit-learn, matplotlib)
```

## Podaci

Kaggle [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification) (izveden iz Stanford Dogs / ImageNet).
Slike se ne čuvaju u repou, a preuzimanje je opisano u [`docs/01-podaci.md`](docs/01-podaci.md).
U repou je samo podela na train/val/test ([`splits/`](splits/)), da bi svi eksperimenti koristili iste skupove.
Podaci smeju da se koriste samo za nekomercijalno istraživanje i učenje.

## Licenca

Kod: [MIT](LICENSE). Podaci imaju sopstvenu licencu (vidi „Podaci“).
