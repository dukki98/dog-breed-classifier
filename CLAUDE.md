# CLAUDE.md

Uputstva za Claude u ovom repozitorijumu. Odgovaraj na srpskom (latinica).

## O projektu

Klasifikator rasa pasa: za fotografiju psa vraća **3 najverovatnije rase** od 120
(Kaggle „Dog Breed Identification“, ~10.000 slika za trening). Model mora biti mali i brz,
a da radi na laptopu autora (vidi „Hardver“).

Projekat služi i za učenje i kao priprema za razgovor za AI praksu. Autor mora da razume
svaku odluku i da može da je objasni.

## Način rada (po fazama)

- Projekat ide u fazama. Pregled i status su u tabeli u `README.md`, a detalji svake faze u `docs/NN-ime-faze.md`.
- Svaka faza ima svoju granu `faza-N-kratko-ime` i PR ka `main`.
- Claude piše kod, ali **jednu fazu po jednu**:
  1. Na početku faze kratko objasni plan i važne odluke, i sačekaj potvrdu.
  2. Implementiraj, pa objasni šta je urađeno i zašto. Ključne delove koda prođi sa autorom.
  3. Rezultate (merenja, tačnost, brzina) zabeleži u `docs/NN-ime-faze.md`.
  4. **Ne prelazi na sledeću fazu dok autor ne pregleda i ne spoji (merge) PR trenutne faze.**
- U dokumentaciju idu samo stvarno izmereni brojevi. Nikad ne izmišljaj ni ne procenjuj rezultate kao da su izmereni.
- Na kraju faze ažuriraj status u `README.md` i dodaj sekciju „Pitanja za proveru razumevanja“ u dokument faze.

## Hardver (ciljna mašina)

- Windows 10, NVIDIA GeForce RTX 3060 Laptop GPU, **6 GB VRAM** (6144 MiB), do 95 W
- Ampere (GA106), compute capability 8.6. Tensor jezgra rade sa FP16, BF16, TF32 i INT8.
- Drajver 610.88, podržava CUDA do 13.3. CUDA Toolkit nije instaliran i nije potreban, jer PyTorch donosi svoj CUDA runtime.
- Trening: mešovita preciznost **BF16** (`torch.autocast(device_type="cuda", dtype=torch.bfloat16)`), bez GradScaler-a
- Inference: **FP16**, a INT8 kao opcioni napredni korak (faza 6)
- Cilj: model sa < 30M parametara. Kod mora da stane u 6 GB VRAM-a, pa pazi na batch size i rezoluciju.
- Trening pokretati dok je laptop na punjaču.

## Okruženje

- Python 3.13, virtuelno okruženje `.venv` u korenu repoa
- PyTorch se instalira iz `requirements-torch.txt`: `torch==2.14.1` i `torchvision==0.29.1` sa indeksa CUDA 13.2.
  Ostale zavisnosti idu u `requirements.txt` (od faze 1).
- Za TF32 koristi samo `torch.set_float32_matmul_precision(...)`. U PyTorch 2.14 mešanje starog API-ja
  (`torch.backends.cuda.matmul.allow_tf32`) i novog (`...matmul.fp32_precision`) baca RuntimeError.
- Kad Claude sam pokreće Python, koristi `.venv/Scripts/python.exe` direktno. To radi i u cmd-u i u Git Bash-u, bez aktivacije.
- Komande u dokumentaciji za autora pisati za Windows `cmd`.

## Konvencije

- Imena u kodu na engleskom. Komentari, docstring-ovi i dokumentacija na srpskom (latinica).
- `data/`, `checkpoints/` i `runs/` nikad ne idu u git.
- Commit poruke na srpskom, kratke, u imperativu („Dodaj …“, „Ispravi …“).
- Odluke se beleže u dokumentu faze, u sekciji „Odluke“: šta je izabrano, koje su bile alternative i zašto.
