# File di avvio 
import os
import pandas as pd
import numpy as np
import sys

# BASE_DIR punta alla cartella code
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from srp_classifier import run_srp
from osnn_classifier import run_osnn
from hat_classifier import run_hat
from ozaboost_adwin_classifier import run_ozaboost_adwin
from sleade_classifier import run_sleade

# Modifica: Punti alla directory radice (Progetto_Tesi-main), salendo di 2 livelli da 'code'
INPUT_PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
SCENARIO_PREFIX = "chunks_scenario"
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results_capymoa")
os.makedirs(RESULTS_DIR, exist_ok=True)


def process_chunks_for_scenario(scenario_dir, scenario_name):
    """
    Elabora tutti i chunk in una cartella scenario (100,75,50,25)
    """
    chunk_files = sorted(
        [f for f in os.listdir(scenario_dir) if f.startswith("chunk_") and f.endswith(".csv")],
        key=lambda x: int(x.split("_")[-1].split(".")[0]) if "_" in x else 0
    )

    print(f"\n📦 {scenario_name}: trovati {len(chunk_files)} chunk in {scenario_dir}")

    chunk_paths = []
    chunk_ids = []

    subfolder_name = os.path.basename(scenario_dir)  # es. "100"

    for i, file_name in enumerate(chunk_files, start=1):
        file_path = os.path.join(scenario_dir, file_name)

        try:
            df_check = pd.read_csv(file_path)
            if df_check.empty:
                print(f"⚠️ Chunk {file_name} vuoto, saltato\n")
                continue
        except Exception as e:
            print(f"❌ Errore nel caricamento del chunk {file_name}: {e}\n")
            continue
        
        chunk_paths.append(file_path)
        chunk_ids.append(i)

    if not chunk_paths:
        print("⚠️ Nessun chunk valido trovato da elaborare.\n")
        return

    try:
        if subfolder_name == "100":
            algos = ["SRP", "HAT", "OzaBoostADWIN"]
        else:
            algos = ["OSNN", "SLEADE"]
            
        for alg in algos:
            alg_result_dirs = []
            for i in chunk_ids:
                alg_result_dirs.append(os.path.join(RESULTS_DIR, scenario_name, subfolder_name, f"{alg}_chunk_{i}"))
                
            if alg == "SRP":
                run_srp(chunk_paths=chunk_paths, chunk_ids=chunk_ids, result_dirs=alg_result_dirs)
            elif alg == "HAT":
                run_hat(chunk_paths=chunk_paths, chunk_ids=chunk_ids, result_dirs=alg_result_dirs)
            elif alg == "OzaBoostADWIN":
                run_ozaboost_adwin(chunk_paths=chunk_paths, chunk_ids=chunk_ids, result_dirs=alg_result_dirs)
            elif alg == "OSNN":
                run_osnn(chunk_paths=chunk_paths, chunk_ids=chunk_ids, result_dirs=alg_result_dirs)
            elif alg == "SLEADE":
                run_sleade(chunk_paths=chunk_paths, chunk_ids=chunk_ids, result_dirs=alg_result_dirs)
            
        print(f"🎯 Elaborazione continua completata per la cartella {scenario_name}/{subfolder_name}\n")

    except Exception as e:
        print(f"❌ Errore durante l'elaborazione continua in {scenario_dir}: {e}\n")


def process_all_scenarios():
    # Trova tutte le cartelle chunks_scenario_*
    scenarios = [d for d in os.listdir(INPUT_PARENT_DIR)
                 if d.startswith(SCENARIO_PREFIX) and os.path.isdir(os.path.join(INPUT_PARENT_DIR, d))]

    for scenario_name in sorted(scenarios):
        scenario_dir = os.path.join(INPUT_PARENT_DIR, scenario_name)
        # Trova sotto-cartelle (100,75,50,25)
        subfolders = [f for f in os.listdir(scenario_dir) if os.path.isdir(os.path.join(scenario_dir, f))]
        for subfolder in sorted(subfolders):
            subfolder_dir = os.path.join(scenario_dir, subfolder)
            process_chunks_for_scenario(subfolder_dir, scenario_name)

    print("🎯 Tutti gli scenari e sottocartelle sono stati elaborati.")


if __name__ == "__main__":
    process_all_scenarios()
