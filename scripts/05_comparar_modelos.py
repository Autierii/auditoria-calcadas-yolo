"""
05 - Comparação de configurações (mesmo split do notebook): modelo x resolução de entrada.

Treina cada combinação com os mesmos hiperparâmetros e grava a tabela em model/comparacao.csv.
Usado para escolher a configuração final do notebook de treinamento.

Uso:
    python scripts/05_comparar_modelos.py
"""
import time
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data/dataset/data.yaml"  # criado pelo notebook (split com semente fixa)
CONFIGS = [("yolov8n.pt", 448), ("yolov8n.pt", 640), ("yolo11n.pt", 448), ("yolo11n.pt", 640)]

linhas = []
for pesos, imgsz in CONFIGS:
    nome = f"cmp_{Path(pesos).stem}_{imgsz}"
    t0 = time.time()
    YOLO(pesos).train(data=str(DATA), imgsz=imgsz, epochs=150, patience=40, batch=16, seed=42,
                      project=str(RAIZ / "runs"), name=nome, exist_ok=True, plots=False, verbose=False, workers=0)
    best = YOLO(RAIZ / "runs" / nome / "weights/best.pt")
    v = best.val(data=str(DATA), split="val", imgsz=imgsz, verbose=False, plots=False, workers=0)
    linhas.append(dict(modelo=Path(pesos).stem, imgsz=imgsz, mAP50_val=round(v.box.map50, 3),
                       mAP50_95_val=round(v.box.map, 3), precision=round(v.box.mp, 3), recall=round(v.box.mr, 3),
                       minutos=round((time.time() - t0) / 60, 1)))
    print(linhas[-1], flush=True)

df = pd.DataFrame(linhas)
df.to_csv(RAIZ / "model/comparacao.csv", index=False)
print(df.to_string(index=False))
