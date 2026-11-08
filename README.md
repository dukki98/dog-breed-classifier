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
| 0 | Okruženje, provera GPU-a, struktura repoa ([docs](docs/00-setup.md)) | u toku |
| 1 | Podaci: preuzimanje, analiza, podela na train/val | — |
| 2 | Polazni model: fine-tuning, top-1 i top-3 tačnost | — |
| 3 | Augmentacije, raspored stope učenja, BF16, praćenje eksperimenata | — |
| 4 | Pretraga hiperparametara (Optuna) | — |
| 5 | Analiza grešaka: koje rase model meša | — |
| 6 | Izvoz u ONNX, FP16/INT8, merenje brzine i VRAM-a | — |
| 7 | API ili demo koji vraća 3 najverovatnije rase | — |
| 8 | Praćenje modela u produkciji | — |

## Brzi početak (Windows, cmd)

Detaljno uputstvo i objašnjenja su u [`docs/00-setup.md`](docs/00-setup.md).

```bat
git clone https://github.com/dukki98/dog-breed-classifier.git
cd dog-breed-classifier
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-torch.txt
python scripts\check_gpu.py
```

## Struktura

```
dog-breed-classifier/
├── CLAUDE.md               uputstva za Claude (način rada, hardver, konvencije)
├── docs/                   dokument za svaku fazu: odluke, rezultati, naučeno
├── scripts/
│   └── check_gpu.py        provera GPU-a i brzine množenja matrica (FP32/TF32/BF16)
└── requirements-torch.txt  PyTorch sa CUDA 13.2
```

## Podaci

Kaggle [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification) (izveden iz Stanford Dogs / ImageNet).
Slike se ne čuvaju u repou. Preuzimanje se opisuje u fazi 1. Podaci smeju da se koriste samo za nekomercijalno istraživanje i učenje.

## Licenca

Kod: [MIT](LICENSE). Podaci imaju sopstvenu licencu (vidi „Podaci“).
