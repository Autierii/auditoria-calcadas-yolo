"""
01 - Coleta de imagens da Auditoria Cidadã de Calçadas (Brás / SP).

Adaptado do notebook de apoio da disciplina (download via API do Mapillary).

Duas fontes possíveis:
  * "mapillary": usa a Graph API do Mapillary (precisa de token em MAPILLARY_TOKEN),
    exatamente como no notebook de apoio (usuário gabinete_falzoni, 08-17/nov/2025, bbox do Brás).
  * "falzoni" (padrão): usa a camada "Fotos Auditoria" do mapa público do projeto
    (https://falzoni.com.br/auditoria_calcadas/mapa/), que referencia AS MESMAS fotos do
    Mapillary (mesmos IDs), hospedadas no site do projeto. Não exige token.

Uso:
    python scripts/01_coletar_imagens.py --n 400 --seed 42 --out data/raw
"""
import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

FALZONI_LAYER = "https://falzoni.com.br/auditoria_calcadas/mapa/data/FotosAuditoria_6.js"
CREATOR_NAME = "gabinete_falzoni"
START_DATE = "2025-11-08T00:00:00Z"
END_DATE = "2025-11-17T23:59:59Z"
BBOX = (-46.628036, -23.549392, -46.605377, -23.529917)  # Brás (xmin, ymin, xmax, ymax)

session = requests.Session()
session.headers.update({"User-Agent": "auditoria-calcadas-downloader/1.0"})


def listar_falzoni():
    txt = session.get(FALZONI_LAYER, timeout=60).text
    geojson = json.loads(txt[txt.index("{"):].rstrip().rstrip(";"))
    linhas = []
    for f in geojson["features"]:
        p = f["properties"]
        url = re.search(r"src='([^']+)'", p["foto_url"]).group(1)
        lon, lat = f["geometry"]["coordinates"][:2]
        linhas.append({"id": p["id"], "captured_datetime": p["captured_datetime"],
                       "latitude": lat, "longitude": lon, "url": url})
    return pd.DataFrame(linhas)


def listar_mapillary(token, nx=20, ny=20):
    xmin, ymin, xmax, ymax = BBOX
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    todas = {}
    for i in range(nx):
        for j in range(ny):
            tile = (xmin + i * dx, ymin + j * dy, xmin + (i + 1) * dx, ymin + (j + 1) * dy)
            url = "https://graph.mapillary.com/images"
            params = {"fields": "id,computed_geometry,thumb_original_url,captured_at",
                      "bbox": ",".join(map(str, tile)), "creator_username": CREATOR_NAME,
                      "start_captured_at": START_DATE, "end_captured_at": END_DATE,
                      "limit": 1000, "access_token": token}
            while url:
                data = session.get(url, params=params, timeout=30).json()
                for im in data.get("data", []):
                    todas[str(im["id"])] = im
                url, params = data.get("paging", {}).get("next"), None
    linhas = []
    for im in todas.values():
        lon, lat = (im.get("computed_geometry") or {}).get("coordinates", [None, None])[:2]
        linhas.append({"id": str(im["id"]),
                       "captured_datetime": pd.to_datetime(im.get("captured_at"), unit="ms", utc=True),
                       "latitude": lat, "longitude": lon, "url": im.get("thumb_original_url")})
    return pd.DataFrame(linhas)


def baixar(row, pasta):
    destino = os.path.join(pasta, f"{row['id']}.jpg")
    if os.path.exists(destino):
        return "existe"
    for tentativa in range(3):
        try:
            r = session.get(row["url"], timeout=30)
            r.raise_for_status()
            if not r.content:
                raise ValueError("arquivo vazio")
            with open(destino, "wb") as fh:
                fh.write(r.content)
            return "ok"
        except Exception:
            time.sleep(1 + tentativa)
    return "erro"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonte", choices=["falzoni", "mapillary"], default="falzoni")
    ap.add_argument("--n", type=int, default=400, help="quantas imagens amostrar (0 = todas)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    if args.fonte == "mapillary":
        df = listar_mapillary(os.environ["MAPILLARY_TOKEN"])
    else:
        df = listar_falzoni()
    df = df.drop_duplicates("id").sort_values("captured_datetime").reset_index(drop=True)
    print(f"{len(df)} imagens disponíveis na fonte '{args.fonte}'")

    if args.n and args.n < len(df):
        random.seed(args.seed)
        df = df.loc[sorted(random.sample(range(len(df)), args.n))].reset_index(drop=True)

    pasta = os.path.join(args.out, "imagens")
    os.makedirs(pasta, exist_ok=True)
    df.to_csv(os.path.join(args.out, "fotos.csv"), index=False)

    status = {"ok": 0, "existe": 0, "erro": 0}
    with ThreadPoolExecutor(max_workers=10) as ex:
        for fut in as_completed([ex.submit(baixar, r, pasta) for _, r in df.iterrows()]):
            status[fut.result()] += 1
    print(f"Download concluído: {status}")


if __name__ == "__main__":
    main()
