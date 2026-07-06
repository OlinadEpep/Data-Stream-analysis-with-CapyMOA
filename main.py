#!/usr/bin/env python3
"""
main.py
Generazione chunk + integrazione con Algoritmo/code/process_chunks.py

Posizione prevista:
- main.py                 -> C:/.../Progetto tesi/main.py
- process_chunks.py       -> C:/.../Progetto tesi/Algoritmo/code/process_chunks.py
"""

from pathlib import Path
import sys
import math
import random
from collections import defaultdict
import subprocess

# pacchetti esterni
import pandas as pd
import numpy as np

# --------------------------
# Configurazione principale
# --------------------------
BASE_DIR = Path(__file__).resolve().parent

CARTELLA = BASE_DIR / "dati"
OUTPUT_CSV_COMPLETO = BASE_DIR / "pazienti.csv"
OUTPUT_CSV_FEATURES = BASE_DIR / "pazienti_feature.csv"
COLONNA_ID = "ID_Paziente"
COLONNA_CLASSE = "Classe"

CHUNKS_DIR = BASE_DIR / "chunks"
CHUNKS_STRAT_DIR = BASE_DIR / "chunks_scenario_1"
CHUNKS_SCENARIO2_DIR = BASE_DIR / "chunks_scenario_2"
CHUNKS_SCENARIO3_DIR = BASE_DIR / "chunks_scenario_3"
CHUNKS_SCENARIO4_DIR = BASE_DIR / "chunks_scenario_4"

NUM_CHUNK = 10
NUM_CHUNK_5 = 5
SHUFFLE = True
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DESIRED_COLUMNS = [
    "original_shape2D_Elongation",
    "original_shape2D_MajorAxisLength",
    "original_shape2D_MinorAxisLength",
    "original_shape2D_Perimeter",
    "original_shape2D_MaximumDiameter",
    "original_shape2D_Sphericity",
    "original_firstorder_Mean",
    "original_firstorder_Median",
    "original_firstorder_Minimum",
    "original_firstorder_Maximum",
    "original_firstorder_Range",
    "original_firstorder_Uniformity",
    "Centroid_X",
    "Centroid_Y",
]

# --------------------------
# Funzioni ausiliarie
# --------------------------

def salva_distribuzione_chunk(chunk_dir: Path, num_chunk: int, nome_file: str):
    distribuzione_chunk = []
    for i in range(num_chunk):
        path = chunk_dir / f"chunk_{i+1}.csv"
        if not path.exists():
            continue
        df_chunk = pd.read_csv(path)
        if COLONNA_CLASSE not in df_chunk.columns:
            continue
        dist = df_chunk[COLONNA_CLASSE].value_counts().reset_index()
        dist.columns = ["Classe", "Conteggio"]
        dist["Chunk"] = i + 1
        distribuzione_chunk.append(dist)
    if distribuzione_chunk:
        distribuzione_chunk_df = pd.concat(distribuzione_chunk, ignore_index=True)
        try:
            distribuzione_chunk_df["Classe"] = distribuzione_chunk_df["Classe"].astype(int)
        except Exception:
            pass
        distribuzione_chunk_df = (
            distribuzione_chunk_df.pivot_table(index="Classe", columns="Chunk", values="Conteggio", fill_value=0)
            .reset_index()
            .sort_values(by="Classe")
        )
        out_path = chunk_dir / nome_file
        distribuzione_chunk_df.to_csv(out_path, index=False)
        print(f"📊 Distribuzione classi salvata in {out_path}")

def genera_versioni_etichettate(chunk_path: Path, num_chunk: int):
    df_chunk = pd.read_csv(chunk_path)
    file_name = chunk_path.name
    scenario_dir = chunk_path.parent

    schemi = {
        "100": [True],
        "75": [True, True, True, False],
        "50": [True, True, False, False],
        "25": [True, False, False, False],
    }

    for perc, pattern in schemi.items():
        output_dir = scenario_dir / perc
        output_dir.mkdir(parents=True, exist_ok=True)
        mask = np.tile(pattern, int(np.ceil(len(df_chunk)/len(pattern))))[:len(df_chunk)]

        df_mod = df_chunk.copy()
        df_mod.loc[~mask, COLONNA_CLASSE] = -1

        cols_finali = [COLONNA_ID, "ID"] + [c for c in DESIRED_COLUMNS if c in df_mod.columns] + [COLONNA_CLASSE]
        cols_finali = [c for c in cols_finali if c in df_mod.columns]
        df_mod = df_mod[cols_finali]
        if "ID_Paziente" in df_mod.columns and "ID" in df_mod.columns:
            df_mod = df_mod.sort_values(by=[COLONNA_ID, "ID"])
        elif COLONNA_ID in df_mod.columns:
            df_mod = df_mod.sort_values(by=[COLONNA_ID])

        output_file = output_dir / file_name
        df_mod.to_csv(output_file, index=False)

        salva_distribuzione_chunk(output_dir, num_chunk, f"distribuzione_{perc}.csv")
        print(f"✅ Versione {perc}% salvata in {output_file}")

def crea_chunk_scenario(df_all: pd.DataFrame, soggetti_per_classe: dict, num_chunk: int, chunk_dir: Path, crea_versioni: bool=True, assente: bool=False):
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_soggetti = [[] for _ in range(num_chunk)]

    if assente:
        salti_per_classe = {classe: random.randint(0, num_chunk - 1) for classe in soggetti_per_classe}
        for classe, ids in soggetti_per_classe.items():
            ids_copy = ids.copy()
            if SHUFFLE:
                random.shuffle(ids_copy)
            skip_chunk = salti_per_classe[classe]
            idx = 0
            for pid in ids_copy:
                while True:
                    target = idx % num_chunk
                    idx += 1
                    if target != skip_chunk:
                        chunk_soggetti[target].append(pid)
                        break
    else:
        for classe, ids in soggetti_per_classe.items():
            ids_copy = ids.copy()
            if SHUFFLE:
                random.shuffle(ids_copy)
            quota = math.floor(len(ids_copy) / num_chunk) if num_chunk > 0 else 0
            for i in range(num_chunk):
                start, end = i * quota, i * quota + quota
                chunk_soggetti[i].extend(ids_copy[start:end])
            resti = ids_copy[num_chunk * quota:]
            for idx, pid in enumerate(resti):
                chunk_soggetti[idx % num_chunk].append(pid)

    for i in range(num_chunk):
        pids = chunk_soggetti[i]
        if not pids:
            (chunk_dir / f"chunk_{i+1}.csv").write_text("")
            print(f"⚠ chunk_{i+1}.csv vuoto (nessun soggetto assegnato).")
            continue

        chunk_df = df_all[df_all[COLONNA_ID].isin(pids)]
        cols_chunk = [COLONNA_ID, "ID"] + [c for c in DESIRED_COLUMNS if c in df_all.columns] + [COLONNA_CLASSE]
        cols_chunk = [c for c in cols_chunk if c in df_all.columns]
        chunk_df = chunk_df[cols_chunk]
        if COLONNA_ID in chunk_df.columns and "ID" in chunk_df.columns:
            chunk_df = chunk_df.sort_values(by=[COLONNA_ID, "ID"])
        elif COLONNA_ID in chunk_df.columns:
            chunk_df = chunk_df.sort_values(by=[COLONNA_ID])

        path = chunk_dir / f"chunk_{i+1}.csv"
        chunk_df.to_csv(path, index=False)
        print(f"📁 Salvato: {path}")

        if crea_versioni:
            genera_versioni_etichettate(path, num_chunk)

    salva_distribuzione_chunk(chunk_dir, num_chunk, f"distribuzione_classi_{chunk_dir.name}.csv")

# --------------------------
# Flusso principale
# --------------------------

def run_chunk_generation():
    if not CARTELLA.exists():
        raise SystemExit(f"Cartella '{CARTELLA}' non trovata. Metti i file nella cartella corretta.")

    dfs = []
    for f in sorted(CARTELLA.iterdir()):
        if not f.is_file():
            continue
        try:
            if f.suffix.lower() == ".csv":
                dfs.append(pd.read_csv(f))
            elif f.suffix.lower() in (".xls", ".xlsx"):
                dfs.append(pd.read_excel(f))
        except Exception as e:
            print(f"⚠ Errore leggendo '{f.name}': {e}. File ignorato.")

    if not dfs:
        raise SystemExit("❌ Nessun file valido trovato nella cartella dati.")

    prime_colonne = dfs[0].columns.tolist()
    if not all(list(df.columns) == prime_colonne for df in dfs):
        raise SystemExit("❌ Le colonne nei file non coincidono. Assicurati che tutti i file abbiano le stesse colonne.")

    df_all = pd.concat(dfs, ignore_index=True)

    if COLONNA_ID not in df_all.columns or COLONNA_CLASSE not in df_all.columns:
        raise SystemExit(f"❌ Colonne '{COLONNA_ID}' o '{COLONNA_CLASSE}' non trovate nel dataset.")

    sort_cols = [COLONNA_ID, "ID", COLONNA_CLASSE]
    sort_cols = [c for c in sort_cols if c in df_all.columns]
    if sort_cols:
        df_all = df_all.sort_values(by=sort_cols)

    OUTPUT_CSV_COMPLETO.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUTPUT_CSV_COMPLETO, index=False)
    print(f"✅ Dataset completo salvato: {OUTPUT_CSV_COMPLETO}")

    colonne_presenti = [COLONNA_ID, "ID", COLONNA_CLASSE] + [c for c in DESIRED_COLUMNS if c in df_all.columns]
    colonne_presenti = [c for c in colonne_presenti if c in df_all.columns]
    df_features = df_all[colonne_presenti]
    df_features.to_csv(OUTPUT_CSV_FEATURES, index=False)
    print(f"✅ Dataset con feature salvato: {OUTPUT_CSV_FEATURES}")

    dist_globale = df_all[COLONNA_CLASSE].value_counts().reset_index()
    dist_globale.columns = ["Classe", "Totale"]
    dist_globale["Percentuale"] = (dist_globale["Totale"] / dist_globale["Totale"].sum() * 100).round(2)
    try:
        dist_globale["Classe"] = dist_globale["Classe"].astype(int)
    except Exception:
        pass
    dist_globale = dist_globale.sort_values(by="Classe").reset_index(drop=True)
    dist_globale.to_csv(BASE_DIR / "distribuzione_classi_completa.csv", index=False)
    print("📊 Distribuzione classi globale salvata: distribuzione_classi_completa.csv")

    df_soggetti = df_all.groupby(COLONNA_ID)[COLONNA_CLASSE].agg(lambda x: x.mode()[0]).reset_index()
    soggetti_per_classe = defaultdict(list)
    for _, row in df_soggetti.iterrows():
        soggetti_per_classe[row[COLONNA_CLASSE]].append(row[COLONNA_ID])

    print("\n📦 Creazione chunks base...")
    crea_chunk_scenario(df_all, soggetti_per_classe, NUM_CHUNK, CHUNKS_DIR, crea_versioni=False)

    print("\n📦 Scenario 1: Stratificato...")
    crea_chunk_scenario(df_all, soggetti_per_classe, NUM_CHUNK, CHUNKS_STRAT_DIR, crea_versioni=True, assente=False)

    print("\n🚫 Scenario 2: Classi assenti...")
    crea_chunk_scenario(df_all, soggetti_per_classe, NUM_CHUNK, CHUNKS_SCENARIO2_DIR, crea_versioni=True, assente=True)

    print("\n📦 Scenario 3: 5 chunk stratificati...")
    crea_chunk_scenario(df_all, soggetti_per_classe, NUM_CHUNK_5, CHUNKS_SCENARIO3_DIR, crea_versioni=True, assente=False)

    print("\n🚫 Scenario 4: 5 chunk con classi assenti...")
    crea_chunk_scenario(df_all, soggetti_per_classe, NUM_CHUNK_5, CHUNKS_SCENARIO4_DIR, crea_versioni=True, assente=True)

    print("\n🎯 Tutti i chunk e le versioni etichettate sono stati generati correttamente!")

    return {
        "base_dir": str(BASE_DIR),
        "chunks_dirs": {
            "base": str(CHUNKS_DIR),
            "strat": str(CHUNKS_STRAT_DIR),
            "scenario2": str(CHUNKS_SCENARIO2_DIR),
            "scenario3": str(CHUNKS_SCENARIO3_DIR),
            "scenario4": str(CHUNKS_SCENARIO4_DIR),
        }
    }

# Esecuzione process_chunks.py come script esterno


def run_process_chunks_script():
    code_dir = BASE_DIR / "Algoritmo" / "code"
    script_path = code_dir / "process_chunks.py"

    if not script_path.exists():
        print(f"❌ File '{script_path}' non trovato. Verifica il percorso.")
        return

    print("\n▶ Avvio process_chunks.py come script esterno ...")
    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
        print("✅ process_chunks.py eseguito correttamente.")
    except subprocess.CalledProcessError as e:
        print(f"❌ process_chunks.py terminato con errore: {e}")
    except Exception as e:
        print(f"❌ Errore durante l'esecuzione di process_chunks.py: {e}")

def run_validate_results_script():
    code_dir = BASE_DIR / "Algoritmo" / "code"
    script_path = code_dir / "validate_results.py"

    if not script_path.exists():
        print(f"❌ File '{script_path}' non trovato. Verifica il percorso.")
        return

    print("\n▶ Avvio validate_results.py come script esterno ...")
    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
        print("✅ validate_results.py eseguito correttamente.")
    except subprocess.CalledProcessError as e:
        print(f"❌ validate_results.py terminato con errore: {e}")
    except Exception as e:
        print(f"❌ Errore durante l'esecuzione di validate_results.py: {e}")


# --------------------------
# Punto di ingresso
# --------------------------

if __name__ == "__main__":
    info = run_chunk_generation()
    run_process_chunks_script()
    run_validate_results_script()
