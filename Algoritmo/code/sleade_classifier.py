import os
import time
import numpy as np
import pandas as pd
from capymoa.stream import NumpyStream
from capymoa.ssl import SLEADE
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def run_sleade(chunk_paths, chunk_ids, result_dirs):
    print(f"\n🚀 Esecuzione SLEADE continua su {len(chunk_paths)} chunk...")

    # 1. Caricamento e preparazione dati
    df_list = []
    chunk_sizes = []
    for p in chunk_paths:
        df = pd.read_csv(p)
        df_list.append(df)
        chunk_sizes.append(len(df))
        
    df_all = pd.concat(df_list, ignore_index=True)
    
    df_ids = df_all[["ID_Paziente", "ID"]] if "ID_Paziente" in df_all.columns else None
    cols_to_drop = ["ID_Paziente", "ID", "Augmented"] 
    df_train = df_all.drop(columns=[c for c in cols_to_drop if c in df_all.columns])

    # 2. Estrazione Feature e Scaling
    feature_cols = [c for c in df_train.columns if c != "Classe" and not c.startswith("diagnostics_")]
    X_raw = df_train[feature_cols].to_numpy(dtype=float)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw) 
    
    y_raw = df_train["Classe"].to_numpy(dtype=int)
    
    classes = np.unique(y_raw[y_raw != -1])
    y = np.full(y_raw.shape, np.nan, dtype=float)
    
    for i, c in enumerate(classes):
        y[y_raw == c] = float(i)

    global_pca = PCA(n_components=2)
    global_pca.fit(X)

    model_pca = PCA(n_components=0.95) 
    X_per_model = model_pca.fit_transform(X)

    nuovi_nomi_feature = [f"PC{i+1}" for i in range(X_per_model.shape[1])]
    stream = NumpyStream(
        X=X_per_model, 
        y=y, 
        dataset_name="stream_continuo_sleade",
        feature_names=nuovi_nomi_feature,
        target_name="Classe"
    )

    try:
        model = SLEADE(schema=stream.get_schema())
    except Exception as e:
        print(f"⚠️ Errore nell'inizializzazione di SLEADE con schema: {e}. Tento senza parametri.")
        model = SLEADE()

    predictions = []
    true_labels = []
    probabilities = []
    correct = 0
    total = 0

    start_time = time.time()
    current_chunk_idx = 0
    instances_processed_in_chunk = 0

    while stream.has_more_instances():
        instance = stream.next_instance()
        
        raw_y_idx = getattr(instance, 'y_index', -1)
        y_true_idx = int(raw_y_idx) if raw_y_idx is not None and not np.isnan(raw_y_idx) else -1
        
        if 0 <= y_true_idx < len(classes) and y_raw[len(predictions)] != -1:
            y_true = classes[y_true_idx]
        else:
            y_true = -1

        y_pred_idx = model.predict(instance)
        
        try:
            prob_dict = model.predict_proba(instance)
            prob_array = np.zeros(len(classes))
            if prob_dict is not None:
                if isinstance(prob_dict, dict):
                    for k, v in prob_dict.items():
                        if 0 <= int(k) < len(classes):
                            prob_array[int(k)] = float(v)
                else:
                    for i in range(min(len(prob_dict), len(classes))):
                        prob_array[i] = float(prob_dict[i])
                
                sum_prob = np.sum(prob_array)
                if sum_prob > 0:
                    prob_array = prob_array / sum_prob
            probabilities.append(prob_array)
        except Exception as e:
            probabilities.append(np.zeros(len(classes)))

        if y_pred_idx is None:
            y_pred = classes[0] if len(classes) > 0 else 0
        else:
            pred_idx = int(y_pred_idx)
            if 0 <= pred_idx < len(classes):
                y_pred = classes[pred_idx]
            else:
                y_pred = classes[0] if len(classes) > 0 else 0

        predictions.append(y_pred)
        true_labels.append(y_true)

        if y_true != -1:
            if y_pred == y_true:
                correct += 1
            total += 1

        if y_true == -1:
            if hasattr(model, 'train_on_unlabeled'):
                model.train_on_unlabeled(instance) 
        else:
            model.train(instance)
        
        instances_processed_in_chunk += 1

        if instances_processed_in_chunk == chunk_sizes[current_chunk_idx]:
            end_time = time.time()
            chunk_id = chunk_ids[current_chunk_idx]
            path_result = result_dirs[current_chunk_idx]
            os.makedirs(path_result, exist_ok=True)
            
            acc = correct / total if total > 0 else 0
            start_idx = sum(chunk_sizes[:current_chunk_idx])
            end_idx = start_idx + chunk_sizes[current_chunk_idx]
            
            chunk_preds = predictions[start_idx:end_idx]
            chunk_trues = true_labels[start_idx:end_idx]
            
            cum_preds = predictions[:end_idx]
            cum_trues = true_labels[:end_idx]

            valid_trues = [t for t in cum_trues if t != -1]
            valid_preds = [p for p, t in zip(cum_preds, cum_trues) if t != -1]

            if len(valid_trues) > 0:
                kappa = cohen_kappa_score(valid_trues, valid_preds)
                f1_macro = f1_score(valid_trues, valid_preds, average="macro", zero_division=0)
                prec_macro = precision_score(valid_trues, valid_preds, average="macro", zero_division=0)
                rec_macro = recall_score(valid_trues, valid_preds, average="macro", zero_division=0)
            else:
                kappa, f1_macro, prec_macro, rec_macro = 0.0, 0.0, 0.0, 0.0

            print(f"✅ Chunk {chunk_id} Completato! Acc: {acc:.4f} | Prec: {prec_macro:.4f} | Rec: {rec_macro:.4f} | F1: {f1_macro:.4f} | Kappa: {kappa:.4f}")

            pd.DataFrame({
                "Chunk": [chunk_id], "Accuracy": [acc], "Precision": [prec_macro],
                "Recall": [rec_macro], "F1_Macro": [f1_macro], "Kappa": [kappa], 
                "Time_s": [end_time - start_time]
            }).to_csv(os.path.join(path_result, "execution_info.csv"), index=False)
            
            res_df = pd.DataFrame({"Real_Class": chunk_trues, "Predicted_Class": chunk_preds})
            if df_ids is not None:
                chunk_ids_df = df_ids.iloc[start_idx:end_idx].reset_index(drop=True)
                res_df = pd.concat([chunk_ids_df, res_df], axis=1)
            res_df.to_csv(os.path.join(path_result, "predictions.csv"), index=False)
            
            X_chunk_original = X[start_idx:end_idx]
            X_pca_chunk = global_pca.transform(X_chunk_original)
            df_pca = pd.DataFrame(X_pca_chunk, columns=["PCA1", "PCA2"])
            df_pca["Classe_Reale"] = np.array(chunk_trues).astype(int)
            df_pca["Classe_Predetta"] = np.array(chunk_preds).astype(int)
            df_pca = df_pca.sort_values(by="Classe_Reale")
            
            standard_colors = sns.color_palette("tab10", len(classes))
            palette_reale = {c: standard_colors[i % len(standard_colors)] for i, c in enumerate(classes)}
            palette_reale[-1] = "lightgray" 
            palette_predetta = {c: standard_colors[i % len(standard_colors)] for i, c in enumerate(classes)}
            palette_predetta[-1] = "black"  

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="Classe_Reale", palette=palette_reale, legend="full", alpha=0.8, ax=axes[0])
            axes[0].set_title(f"Classi Reali - Chunk {chunk_id}")
            sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="Classe_Predetta", palette=palette_predetta, legend="full", alpha=0.8, ax=axes[1])
            axes[1].set_title(f"Predizioni Modello SLEADE - Chunk {chunk_id}")
            
            X_pca_global = global_pca.transform(X)
            x_min, x_max = X_pca_global[:, 0].min() - 1, X_pca_global[:, 0].max() + 1
            y_min, y_max = X_pca_global[:, 1].min() - 1, X_pca_global[:, 1].max() + 1
            for ax in axes:
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(path_result, "pca_plot_comparison.png"))
            plt.close()
            current_chunk_idx += 1
            instances_processed_in_chunk = 0