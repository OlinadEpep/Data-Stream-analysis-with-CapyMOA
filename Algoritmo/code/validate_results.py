import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json, time
import platform

# Usa percorsi dinamici relativi alla posizione dello script
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results_capymoa")
OUTPUT = os.path.join(RESULTS_DIR, "results_summary.csv")

summary = []

print("\n🔍 Validazione risultati CapyMOA (SRP10 & OSNN)...")

if not os.path.exists(RESULTS_DIR):
    print(f"❌ Cartella dei risultati non trovata: {RESULTS_DIR}")
    print("Assicurati di aver eseguito prima gli algoritmi.")
    exit()

for scenario in sorted(os.listdir(RESULTS_DIR)):
    spath = os.path.join(RESULTS_DIR, scenario)
    if not os.path.isdir(spath): continue

    for perc in sorted(os.listdir(spath)):
        ppath = os.path.join(spath, perc)
        if not os.path.isdir(ppath): continue

        for chunkf in sorted(os.listdir(ppath)):
            cpath = os.path.join(ppath, chunkf)
            if not os.path.isdir(cpath): continue

            try:
                chunk_id = int(chunkf.split("_")[-1])
                if "_chunk_" in chunkf:
                    alg_name = chunkf.split("_chunk_")[0]
                else:
                    alg_name = "SRP" if perc == "100" else "OSNN"
            except ValueError:
                continue

            exec_file = os.path.join(cpath, "execution_info.csv")
            pred_file = os.path.join(cpath, "predictions.csv")

            if not (os.path.exists(exec_file) and os.path.exists(pred_file)):
                continue

            # Leggi e controlla le metriche di esecuzione
            try:
                exec_df = pd.read_csv(exec_file)
                acc = exec_df["Accuracy"].iloc[0]
                time_s = exec_df["Time_s"].iloc[0]
                kappa = exec_df["Kappa"].iloc[0] if "Kappa" in exec_df.columns else None
                f1_macro = exec_df["F1_Macro"].iloc[0] if "F1_Macro" in exec_df.columns else None
                precision = exec_df["Precision"].iloc[0] if "Precision" in exec_df.columns else None
                recall = exec_df["Recall"].iloc[0] if "Recall" in exec_df.columns else None
            except Exception:
                acc, time_s, kappa, f1_macro, precision, recall = None, None, None, None, None, None

            # Leggi e valida il file delle predizioni
            try:
                pred_df = pd.read_csv(pred_file)
                predictions_ok = not pred_df.empty and "Predicted_Class" in pred_df.columns
            except Exception:
                predictions_ok = False

            summary.append({
                "Algorithm": alg_name,
                "Scenario": scenario,
                "Perc": perc,
                "Chunk": chunk_id,
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1_Macro": f1_macro,
                "Kappa": kappa,
                "Time_s": time_s,
                "Predictions_OK": predictions_ok
            })

if summary:
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT, index=False)
    print(f"\n✅ Validazione completata. Analizzati {len(summary)} chunk.")
    print(f"📁 Report generale salvato in: {OUTPUT}")

    # Aggregazione Media ± Deviazione Standard 
    print("\n📊 Calcolo aggregazioni per la tesi (Media ± Std)...")
    # Definiamo le metriche numeriche da aggregare
    metrics_to_agg = ["Accuracy", "Precision", "Recall", "F1_Macro", "Kappa"]
    
    # Raggruppiamo e calcoliamo media e deviazione standard
    agg_df = summary_df.groupby(["Algorithm", "Scenario", "Perc"])[metrics_to_agg].agg(['mean', 'std']).reset_index()
    
    # Appiattiamo i nomi delle multi-colonne (es. 'Accuracy', 'mean' -> 'Accuracy_mean')
    agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]
    
    
    thesis_table = pd.DataFrame()
    thesis_table["Algorithm"] = agg_df["Algorithm"]
    thesis_table["Scenario"] = agg_df["Scenario"]
    thesis_table["Perc"] = agg_df["Perc"]
    
    for m in metrics_to_agg:
        thesis_table[m] = agg_df.apply(lambda row: f"{row[m + '_mean']:.4f} ± {row[m + '_std']:.4f}", axis=1)
        
    thesis_out_path = os.path.join(RESULTS_DIR, "aggregated_thesis_table.csv")
    thesis_table.to_csv(thesis_out_path, index=False)
    print(f"📁 Tabella aggregata per la tesi salvata in: {thesis_out_path}")

    print("\n📊 Generazione dei grafici di analisi in corso...")

    # #region agent log
    def _dbg(hypothesisId, location, message, data):
        try:
            repo_root = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
            log_path = os.path.join(repo_root, "debug-4a5394.log")
            payload = {
                "sessionId": "4a5394",
                "runId": "pre-fix",
                "hypothesisId": hypothesisId,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _tail_path(p: str, keep: int = 4) -> str:
        try:
            parts = os.path.normpath(p).split(os.sep)
            return os.sep.join(parts[-keep:])
        except Exception:
            return "<unavailable>"

    def _win_ext_path(p: str) -> str:
        # Win32 extended-length path prefix to bypass some path quirks.
        try:
            if os.name == "nt":
                ap = os.path.abspath(p)
                if ap.startswith("\\\\?\\"):
                    return ap
                if ap.startswith("\\\\"):
                    # UNC path
                    return "\\\\?\\UNC\\" + ap.lstrip("\\")
                return "\\\\?\\" + ap
        except Exception:
            pass
        return p
    # #endregion agent log

    # #region agent log
    # Snapshot ambiente: utile per confrontare VSCode vs Cursor.
    try:
        pil_ver = None
        try:
            from PIL import __version__ as pil_ver  # type: ignore
        except Exception:
            pil_ver = None

        _dbg(
            "H_env",
            "validate_results.py:startup",
            "Runtime environment snapshot",
            {
                "cwd_tail": _tail_path(os.getcwd(), keep=6),
                "sys_executable_tail": _tail_path(sys.executable, keep=6),
                "sys_version": sys.version.split()[0],
                "platform": platform.platform(),
                "matplotlib_version": getattr(matplotlib, "__version__", None),
                "pillow_version": pil_ver,
                "results_dir_tail": _tail_path(RESULTS_DIR, keep=6),
            },
        )
    except Exception:
        pass
    # #endregion agent log
    
    # 1. Learning Curve (Prequential Accuracy Plot) - Cartella etichettature
    for (scenario, perc), group in summary_df.groupby(["Scenario", "Perc"]):
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=group, x="Chunk", y="Kappa", hue="Algorithm", marker="o")
        plt.title(f"Learning Curve (Kappa) - {scenario} ({perc}%)")
        plt.ylim(-0.1, 1.05)
        plt.grid(True)
        out_path = os.path.join(RESULTS_DIR, scenario, perc, "learning_curve.png")
        _dbg(
            "H_path",
            "validate_results.py:savefig_learning_curve",
            "About to savefig",
            {
                "scenario": str(scenario),
                "perc_type": str(type(perc)),
                "perc": str(perc),
                "out_path": str(out_path),
                "out_path_repr": repr(out_path),
                "out_path_len": len(str(out_path)),
                "dir_exists": os.path.isdir(os.path.dirname(out_path)),
                "dir": os.path.dirname(out_path),
                "dir_repr": repr(os.path.dirname(out_path)),
                "dir_len": len(os.path.dirname(out_path)),
            },
        )
        # #region agent log
        # Prova minimale: Windows riesce ad aprire/creare il file con open()?
        try:
            with open(out_path, "wb") as _f:
                _f.write(b"")
            _dbg(
                "H_fs",
                "validate_results.py:preopen_learning_curve",
                "open(out_path,'wb') ok",
                {"out_path": str(out_path)},
            )
        except Exception as e:
            # Se fallisce, proviamo il path esteso Win32 e logghiamo dettagli extra.
            ext = _win_ext_path(out_path)
            def _safe_stat(p):
                try:
                    st = os.stat(p)
                    return {"st_mode": int(st.st_mode), "st_size": int(st.st_size)}
                except Exception as se:
                    return {"exc_type": type(se).__name__, "exc_str": str(se), "errno": getattr(se, "errno", None), "winerror": getattr(se, "winerror", None)}
            _dbg(
                "H_fs",
                "validate_results.py:preopen_learning_curve",
                "open(out_path,'wb') failed",
                {
                    "exc_type": type(e).__name__,
                    "exc_str": str(e),
                    "errno": getattr(e, "errno", None),
                    "winerror": getattr(e, "winerror", None),
                    "filename": getattr(e, "filename", None),
                    "out_path_norm": os.path.normpath(out_path),
                    "out_path_chars": [ord(ch) for ch in str(out_path)[-40:]],
                    "is_file": os.path.isfile(out_path),
                    "is_dir": os.path.isdir(out_path),
                    "exists": os.path.exists(out_path),
                    "stat": _safe_stat(out_path),
                    "dir_stat": _safe_stat(os.path.dirname(out_path)),
                    "dir_list_sample": [repr(n) for n in os.listdir(os.path.dirname(out_path))[:20]] if os.path.isdir(os.path.dirname(out_path)) else None,
                },
            )
            try:
                with open(ext, "wb") as _f:
                    _f.write(b"")
                _dbg(
                    "H_fs",
                    "validate_results.py:preopen_learning_curve",
                    "open(ext_path,'wb') ok",
                    {"ext_path": ext, "ext_len": len(ext)},
                )
            except Exception as e2:
                _dbg(
                    "H_fs",
                    "validate_results.py:preopen_learning_curve",
                    "open(ext_path,'wb') failed",
                    {
                        "ext_path": ext,
                        "ext_len": len(ext),
                        "exc_type": type(e2).__name__,
                        "exc_str": str(e2),
                        "errno": getattr(e2, "errno", None),
                        "winerror": getattr(e2, "winerror", None),
                        "filename": getattr(e2, "filename", None),
                        "ext_is_file": os.path.isfile(ext),
                        "ext_is_dir": os.path.isdir(ext),
                        "ext_exists": os.path.exists(ext),
                        "ext_stat": _safe_stat(ext),
                    },
                )
        # #endregion agent log
        try:
            plt.savefig(_win_ext_path(out_path))
        except Exception as e:
            _dbg(
                "H_path",
                "validate_results.py:savefig_learning_curve",
                "savefig failed",
                {
                    "exc_type": type(e).__name__,
                    "exc_str": str(e),
                    "errno": getattr(e, "errno", None),
                    "winerror": getattr(e, "winerror", None),
                    "filename": getattr(e, "filename", None),
                },
            )
            raise
        plt.close()

    # 2. Degradation Analysis - Cartella Scenario
    for scenario, group in summary_df.groupby("Scenario"):
        for algorithm, alg_group in group.groupby("Algorithm"):
            plt.figure(figsize=(8, 5))
            sns.lineplot(data=alg_group, x="Chunk", y="Kappa", hue="Perc", marker="o")
            plt.title(f"Degradation Analysis (Kappa) - {scenario} ({algorithm})")
            plt.ylim(-0.1, 1.05)
            plt.grid(True)
            out_path = os.path.join(RESULTS_DIR, scenario, f"degradation_plot_{algorithm}.png")
            plt.savefig(out_path)
            plt.close()

    # 3. Concept Drift Resilience - Cartella results_capymoa
    if not summary_df.empty:
        for algorithm, alg_group in summary_df.groupby("Algorithm"):
            summary_df_copy = alg_group.copy()
            summary_df_copy["Scenario_Perc"] = summary_df_copy["Scenario"] + " (" + summary_df_copy["Perc"].astype(str) + "%)"
            
            custom_palette = {}
            # Sfumature: dal più scuro (100%) al più chiaro (25%)
            yellows = ['#B8860B', '#DAA520', '#FFD700', '#F0E68C'] # Scenario 1 (Giallo/Oro)
            greens = ['#006400', '#228B22', '#32CD32', '#90EE90']   # Scenario 2 (Verde)
            blues = ['#00008B', '#0000CD', '#4169E1', '#87CEFA']    # Scenario 3 (Blu)
            reds = ['#8B0000', '#DC143C', '#FF0000', '#FA8072']     # Scenario 4 (Rosso)

            for i, row in summary_df_copy[["Scenario", "Perc"]].drop_duplicates().iterrows():
                scen = row["Scenario"]
                perc = str(row["Perc"])
                name = f"{scen} ({perc}%)"
                
                try:
                    idx = ['100', '75', '50', '25'].index(perc)
                except ValueError:
                    idx = 0
                    
                if 'scenario_1' in scen:
                    custom_palette[name] = yellows[idx]
                elif 'scenario_2' in scen:
                    custom_palette[name] = greens[idx]
                elif 'scenario_3' in scen:
                    custom_palette[name] = blues[idx]
                elif 'scenario_4' in scen:
                    custom_palette[name] = reds[idx]
                else:
                    custom_palette[name] = '#000000'

            plt.figure(figsize=(12, 7))
            sns.lineplot(data=summary_df_copy, x="Chunk", y="Kappa", hue="Scenario_Perc", palette=custom_palette, marker="o")
            plt.title(f"Concept Drift Comparison (Kappa) - {algorithm}")
            plt.ylim(-0.1, 1.05)
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            out_path = os.path.join(RESULTS_DIR, f"concept_drift_comparison_{algorithm}.png")
            plt.savefig(out_path)
            plt.close()

    # 4. Overall Degradation by Labeling Percentage per Algorithm - Cartella results_capymoa
    if not summary_df.empty:
        for algorithm, alg_group in summary_df.groupby("Algorithm"):
            plt.figure(figsize=(8, 5))
            # Ordinamento delle percentuali in modo decrescente (es. 100, 75, 50, 25)
            perc_order = sorted(alg_group["Perc"].astype(str).unique(), key=lambda x: int(x) if x.isdigit() else 0, reverse=True)
            sns.lineplot(data=alg_group, x="Chunk", y="Kappa", hue="Perc", hue_order=perc_order, marker="o", errorbar=None)
            plt.title(f"Overall Degradation by Labeling Percentage (Kappa) - {algorithm}")
            plt.ylim(-0.1, 1.05)
            plt.grid(True)
            out_path = os.path.join(RESULTS_DIR, f"overall_degradation_{algorithm}.png")
            plt.savefig(out_path)
            plt.close()

    # 5. Efficiency Analysis (Tempi di Esecuzione medi)
    time_summary = summary_df.groupby(["Algorithm", "Scenario", "Perc"])["Time_s"].mean().reset_index()
    time_summary.rename(columns={"Time_s": "Time_s"}, inplace=True)
    time_out = os.path.join(RESULTS_DIR, "execution_times_summary.csv")
    time_summary.to_csv(time_out, index=False)
     
    print(f"✅ Grafici e analisi dell'efficienza generati con successo!")
    print(f"📁 Report tempi medi: {time_out}")
else:
    print("\n⚠️ Nessun risultato valido trovato per la validazione.")
