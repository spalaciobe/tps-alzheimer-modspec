# Opciones para correr el LOSO completo más rápido

Resumen del cuello de botella actual:
- **Local**: RTX 3050 (4GB VRAM, capability 8.6, CUDA 12.4). PyTorch 2.6+cu124.
- **Quick mode**: 13 min/método (epochs=10, batch=32, subsample=100/sujeto). Pipeline funcional.
- **Full mode estimado** (epochs=50, batch=64, sin subsample): ~5 h/método ⇒ 10 h total.

## A. Cloud — opciones recomendadas (en orden)

### A.1 Google Colab Free (T4 GPU 16GB) — **mejor opción gratis**
- GPU T4 ≈ **10–15× la RTX 3050 Laptop** en throughput de CNN.
- 12 h de runtime continuo, ~20 GB en `/content`.
- **Plan**: subir solo los `.h5` precomputados (modspecs STFT + CWT, ~11 GB) a Google Drive,
  montar Drive en Colab y correr `03_train_loso.py` desde allí. El preproceso y modspec ya están hechos.
- Estimado en T4: **~20 min STFT full + 20 min CWT full = 40 min total**.
- Cómo:
  1. `pip install -e .` en Colab tras `git clone`.
  2. `from google.colab import drive; drive.mount('/content/drive')`.
  3. Símbolo `data/derivatives/` → ruta en Drive (o `cp -r` al SSD efímero de Colab para más velocidad).
  4. Lanzar el script igual que en local.

### A.2 Kaggle Notebooks (T4 × 2 o P100) — alternativa gratis con más cuota
- 30 h/semana de GPU, 73 GB en `/kaggle/working/` + `/kaggle/input/`.
- Mejor que Colab si Colab te corta por inactividad.
- Subir los `.h5` como **dataset privado**; el repo como gist o notebook.

### A.3 Vercel Sandbox / Hugging Face Spaces / Lambda Cloud — pago por uso
- **Lambda Labs / vast.ai**: ~$0.30/h por RTX 3090 (24GB). Una corrida full LOSO en ~30 min ≈ **$0.15**.
- **AWS SageMaker Studio Lab** (gratis 8h/día con T4) — alternativa.
- **HF Spaces ZeroGPU** — gratis pero limitado a inferencia, no training largo.

### A.4 Lo que NO recomiendo
- **Hostinger / VPS sin GPU**: igual de lento que tu laptop, no aporta.
- **AWS EC2 g4dn.xlarge on-demand**: ~$0.50/h. Más complejo de setup que Colab.

## B. Optimizaciones locales que aún no probamos

| Técnica | Speedup esperado | Esfuerzo | Cómo |
|---|---|---|---|
| **Mixed precision (AMP)** | 1.7–2× en Ampere | Bajo | `torch.amp.autocast('cuda', dtype=torch.float16)` en el train loop + `GradScaler` |
| **batch_size 128 / 256** | 1.3–1.5× | Trivial | Cambiar `cnn.yaml`. RTX 3050 4GB aguanta b=256 con esta CNN |
| **num_workers=4 + persistent_workers** | 1.3× | Bajo | Verificar que no rompa en Windows (a veces ftal con spawn) |
| **modspec en float16** | menos RAM/transfer | Bajo | Storage: float16 en HDF5; dequantize a float32 al cargar |
| **torch.compile** | 1.3–1.7× | Medio | `model = torch.compile(model)` (PyTorch 2.x) |
| **DataLoader con pin_memory + non_blocking transfer** | 1.1× | Trivial | Ya activo |
| **TF32 implícito en Ampere** | gratis | Trivial | Ya activo en torch 2.x |

Combinadas (AMP + batch 128 + num_workers + compile): **~3× speedup** ⇒ full LOSO en ~1.7 h/método localmente.

## C. Recomendación práctica

1. **Para hoy (resultados con quick mode)**: ya están listos. Pipeline validado, SVM da 77% accuracy y AUC 85% para STFT. Cierra el ciclo.
2. **Para el TFM final (resultados publicables)**: subir modspecs a Drive y correr en **Colab T4** — 40 minutos total, gratis, sin cambiar código.
3. **Si Colab te frustra** (limit, desconexión): **Kaggle Notebooks** con dataset privado, mismas 40 minutos.
4. **Si quieres iterar mucho** (varias seeds, varios HP): RTX 3090 alquilada en **vast.ai** por unas horas, ~$1–2.

Lo que **yo haría ahora mismo**:
- Verificar resultados quick en notebook 06.
- Subir HDF5 a Google Drive (~11 GB; tarda 30 min con buena red).
- Crear notebook Colab que clone el repo, monte Drive y corra `03_train_loso.py` sin `--quick`.

## D. Snippets útiles

### D.1 Setup en Colab
```python
!git clone https://github.com/spalaciobe/tps-alzheimer-modspec.git
%cd tps-alzheimer-modspec
!pip install -q -r requirements.txt
!pip install -q -e .

from google.colab import drive
drive.mount('/content/drive')
!ln -s /content/drive/MyDrive/tps-data data
!python scripts/03_train_loso.py --method stft --fs 200 --seed 0
!python scripts/03_train_loso.py --method cwt  --fs 200 --seed 0
```

### D.2 AMP en `src/train.py` (parche local)
```python
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda')
with autocast('cuda', dtype=torch.float16):
    logits = model(x)
    loss = criterion(logits, y)
scaler.scale(loss).backward()
scaler.step(optim)
scaler.update()
```
