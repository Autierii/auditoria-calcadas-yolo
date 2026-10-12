"""
04 - Aplica a revisão manual sobre a pré-anotação e gera os rótulos finais.

Arquivo de revisão (data/revisao.txt), uma linha por imagem revisada:

    <id> ok                         -> aceita a pré-anotação como está
    <id> rej                        -> imagem descartada do dataset
    <id> d=0,2 c=1:poste a=lixo:10,150,60,200;poste:300,0,320,180
         d=  índices (da folha de revisão) das caixas a remover
         c=  troca de classe da caixa <idx>:<classe>
         m=  move/redimensiona a caixa <idx>:x1,y1,x2,y2
         a=  caixas novas <classe>:x1,y1,x2,y2 (pixels da imagem original 440x248)
    linhas começando com # são comentários

Imagens não listadas no arquivo NÃO entram no dataset (só entra o que foi revisado).

Uso:
    python scripts/04_aplicar_revisao.py --pre data/pre_labels --rev data/revisao.txt --out data/labels
"""
import argparse
import os

from PIL import Image

NOMES = ["buraco", "lixo", "piso_tatil", "poste"]


def ler_pre(path):
    caixas = []
    if os.path.exists(path):
        for linha in open(path).read().split("\n"):
            p = linha.split()
            if len(p) >= 5:
                caixas.append([int(p[0])] + [float(v) for v in p[1:5]])
    return caixas


def xyxy_para_yolo(x1, y1, x2, y2, W, H):
    x1, x2 = sorted((max(0, x1), min(W, x2)))
    y1, y2 = sorted((max(0, y1), min(H, y2)))
    return [(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", default="data/raw/imagens")
    ap.add_argument("--pre", default="data/pre_labels")
    ap.add_argument("--rev", default="data/revisao.txt")
    ap.add_argument("--out", default="data/labels")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    aceitas, rejeitadas = 0, 0
    for n_linha, linha in enumerate(open(args.rev, encoding="utf-8"), 1):
        linha = linha.split("#")[0].strip()
        if not linha:
            continue
        partes = linha.split()
        nome, cmds = partes[0], partes[1:]
        if cmds == ["rej"]:
            rejeitadas += 1
            continue
        W, H = Image.open(os.path.join(args.imgs, nome + ".jpg")).size
        caixas = ler_pre(os.path.join(args.pre, nome + ".txt"))
        remover, novas = set(), []
        for cmd in cmds:
            if cmd == "ok":
                continue
            chave, valor = cmd.split("=", 1)
            if chave == "d":
                remover |= {int(i) for i in valor.split(",")}
            elif chave == "c":
                for item in valor.split(";"):
                    i, cls = item.split(":")
                    caixas[int(i)][0] = NOMES.index(cls)
            elif chave == "m":
                for item in valor.split(";"):
                    i, coords = item.split(":")
                    caixas[int(i)][1:] = xyxy_para_yolo(*map(float, coords.split(",")), W, H)
            elif chave == "a":
                for item in valor.split(";"):
                    cls, coords = item.split(":")
                    novas.append([NOMES.index(cls)] + xyxy_para_yolo(*map(float, coords.split(",")), W, H))
            else:
                raise ValueError(f"linha {n_linha}: comando desconhecido '{cmd}'")
        finais = [c for i, c in enumerate(caixas) if i not in remover] + novas
        with open(os.path.join(args.out, nome + ".txt"), "w") as fh:
            fh.write("".join(f"{c[0]} {c[1]:.5f} {c[2]:.5f} {c[3]:.5f} {c[4]:.5f}\n" for c in finais))
        aceitas += 1
    print(f"{aceitas} imagens rotuladas, {rejeitadas} descartadas -> {args.out}")


if __name__ == "__main__":
    main()
