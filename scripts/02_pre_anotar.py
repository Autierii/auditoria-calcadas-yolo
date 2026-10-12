"""
02 - Pré-anotação automática (zero-shot) com Grounding DINO.

As caixas geradas aqui são apenas um PONTO DE PARTIDA: depois todas as imagens passam
por revisão visual (script 03) e correção manual (data/revisao.json, script 04)
antes de entrarem no dataset.

Estratégia:
  * um prompt por classe (o Grounding DINO perde precisão com muitas frases juntas);
  * filtros geométricos / de cor por classe para cortar falsos positivos óbvios:
      - piso_tatil: precisa ter pixels amarelos dentro da caixa;
      - poste: caixa alta e fina;
      - buraco / lixo: caixas não podem ser gigantes.

Saída: um .txt por imagem no formato YOLO (classe cx cy w h) + score, em data/pre_labels/.

Uso:
    python scripts/02_pre_anotar.py --imgs data/raw/imagens --out data/pre_labels
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

NOMES = ["buraco", "lixo", "piso_tatil", "poste"]
PROMPTS = {
    0: "pothole. broken sidewalk pavement. missing paving stones. damaged sidewalk.",
    1: "trash bag. garbage. litter. rubbish pile.",
    2: "yellow line on sidewalk. yellow tactile paving.",
    3: "pole. utility pole. lamp post. concrete pole.",
}
LIMIAR = {0: 0.30, 1: 0.30, 2: 0.30, 3: 0.30}


def amarelo(im_np, x1, y1, x2, y2):
    """Fração de pixels amarelados/ocre (piso tátil) dentro da caixa.
    As miniaturas têm cores desbotadas, então o critério é relativo:
    R e G bem acima de B, e G próximo de R (para não pegar tijolo avermelhado)."""
    crop = im_np[int(y1):int(y2), int(x1):int(x2)].astype(int)
    if crop.size == 0:
        return 0.0
    r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
    mask = ((r + g) / 2 - b > 22) & (g > 0.78 * r) & (r > 60)
    return float(mask.mean())


def filtro(c, x1, y1, x2, y2, W, H, im_np):
    w, h = x2 - x1, y2 - y1
    area = (w * h) / (W * H)
    if c == 0:  # buraco
        return area < 0.12 and y1 > 0.25 * H
    if c == 1:  # lixo
        return area < 0.20
    if c == 2:  # piso tátil
        return area < 0.35 and amarelo(im_np, x1, y1, x2, y2) > 0.12
    if c == 3:  # poste
        return h > 2.2 * w and h > 0.15 * H
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", default="data/raw/imagens")
    ap.add_argument("--out", default="data/pre_labels")
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--limite", type=int, default=0, help="processa só N imagens (teste)")
    ap.add_argument("--refazer", action="store_true", help="reprocessa imagens já anotadas")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # as fotos têm só 440x248: não faz sentido o padrão do processor (ampliar para 800x1333)
    proc = AutoProcessor.from_pretrained(args.model, size={"shortest_edge": 496, "longest_edge": 880})
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(dev).eval()

    arquivos = sorted(glob.glob(os.path.join(args.imgs, "*.jpg")))
    if args.limite:
        arquivos = arquivos[:args.limite]
    t0 = time.time()
    for i, path in enumerate(arquivos, 1):
        nome = os.path.splitext(os.path.basename(path))[0]
        destino = os.path.join(args.out, nome + ".txt")
        if os.path.exists(destino) and not args.refazer:
            continue  # permite retomar de onde parou
        im = Image.open(path).convert("RGB")
        W, H = im.size
        im_np = np.asarray(im)
        linhas = []
        for c, prompt in PROMPTS.items():
            inputs = proc(images=im, text=prompt, return_tensors="pt").to(dev)
            with torch.inference_mode():
                out = model(**inputs)
            res = proc.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=LIMIAR[c], text_threshold=0.25, target_sizes=[(H, W)])[0]
            if len(res["boxes"]) == 0:
                continue
            keep = nms(res["boxes"], res["scores"], 0.5)
            for k in keep.tolist():
                x1, y1, x2, y2 = [float(v) for v in res["boxes"][k]]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                if not filtro(c, x1, y1, x2, y2, W, H, im_np):
                    continue
                w, h = x2 - x1, y2 - y1
                s = float(res["scores"][k])
                linhas.append(f"{c} {(x1 + w / 2) / W:.5f} {(y1 + h / 2) / H:.5f} {w / W:.5f} {h / H:.5f} {s:.3f}")
        with open(destino, "w") as fh:
            fh.write("\n".join(linhas))
        if i % 25 == 0 or i == len(arquivos):
            print(f"{i}/{len(arquivos)}  {(time.time() - t0) / i:.2f}s/img", flush=True)


if __name__ == "__main__":
    main()
