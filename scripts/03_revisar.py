"""
03 - Revisão visual das anotações.

Gera "folhas de revisão" (grades de imagens com as caixas desenhadas e numeradas)
para conferir rótulos imagem a imagem. Usado tanto na revisão da pré-anotação
quanto para conferir o dataset final.

Uso:
    python scripts/03_revisar.py --imgs data/raw/imagens --labels data/pre_labels --out review --por-folha 12
"""
import argparse
import glob
import os

from PIL import Image, ImageDraw, ImageFont

NOMES = ["buraco", "lixo", "piso_tatil", "poste"]
CORES = [(255, 23, 68), (255, 145, 0), (255, 214, 0), (0, 176, 255)]


def ler_labels(path):
    if not os.path.exists(path):
        return []
    caixas = []
    for linha in open(path).read().split("\n"):
        p = linha.split()
        if len(p) >= 5:
            caixas.append((int(p[0]), *map(float, p[1:5])))
    return caixas


def desenhar(img_path, lbl_path, escala=2, grade=0):
    im = Image.open(img_path).convert("RGB")
    w0, h0 = im.size
    im = im.resize((im.width * escala, im.height * escala))
    W, H = im.size
    d = ImageDraw.Draw(im)
    try:
        fonte = ImageFont.truetype("arial.ttf", 18)
        fonte_p = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        fonte = fonte_p = ImageFont.load_default()
    if grade:  # grade em coordenadas da imagem ORIGINAL (px), para anotar caixas novas
        for x in range(grade, w0, grade):
            d.line([(x * escala, 0), (x * escala, H)], fill=(255, 255, 255), width=1)
            d.text((x * escala + 2, 2), str(x), fill=(255, 255, 0), font=fonte_p)
        for y in range(grade, h0, grade):
            d.line([(0, y * escala), (W, y * escala)], fill=(255, 255, 255), width=1)
            d.text((2, y * escala + 2), str(y), fill=(255, 255, 0), font=fonte_p)
    for k, (c, cx, cy, w, h) in enumerate(ler_labels(lbl_path)):
        x1, y1, x2, y2 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
        d.rectangle([x1, y1, x2, y2], outline=CORES[c], width=3)
        txt = f"{k}:{NOMES[c]}"
        d.rectangle([x1, max(0, y1 - 20), x1 + 9 * len(txt) + 6, max(20, y1)], fill=CORES[c])
        d.text((x1 + 3, max(0, y1 - 20)), txt, fill=(0, 0, 0), font=fonte)
    nome = os.path.splitext(os.path.basename(img_path))[0]
    d.rectangle([0, H - 24, 12 * len(nome) + 8, H], fill=(0, 0, 0))
    d.text((4, H - 22), nome, fill=(255, 255, 255), font=fonte)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="review")
    ap.add_argument("--por-folha", type=int, default=12)
    ap.add_argument("--colunas", type=int, default=3)
    ap.add_argument("--lista", help="arquivo com nomes (sem extensão) a revisar, um por linha")
    ap.add_argument("--grade", type=int, default=0, help="desenha grade a cada N px (coordenadas originais)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.lista:
        nomes = [n.strip() for n in open(args.lista) if n.strip()]
        imgs = [os.path.join(args.imgs, n + ".jpg") for n in nomes]
    else:
        imgs = sorted(glob.glob(os.path.join(args.imgs, "*.jpg")))

    for f0 in range(0, len(imgs), args.por_folha):
        tiles = [desenhar(p, os.path.join(args.labels, os.path.splitext(os.path.basename(p))[0] + ".txt"),
                          grade=args.grade)
                 for p in imgs[f0:f0 + args.por_folha]]
        tw, th = tiles[0].size
        linhas = (len(tiles) + args.colunas - 1) // args.colunas
        folha = Image.new("RGB", (tw * args.colunas, th * linhas), (40, 40, 40))
        for i, t in enumerate(tiles):
            folha.paste(t.resize((tw, th)), ((i % args.colunas) * tw, (i // args.colunas) * th))
        folha.save(os.path.join(args.out, f"folha_{f0 // args.por_folha:03d}.jpg"), quality=85)
    print(f"{(len(imgs) + args.por_folha - 1) // args.por_folha} folhas em {args.out}")


if __name__ == "__main__":
    main()
