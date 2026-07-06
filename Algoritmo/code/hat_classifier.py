import os
import time
import numpy as np
import pandas as pd
from capymoa.stream import NumpyStream
from capymoa.classifier import HoeffdingAdaptiveTree
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def run_hat(chunk_paths, chunk_ids, result_dirs):
    print(f"\n🚀 Esecuzione HAT continua su {len(chunk_paths)} chunk...")
    df_list = []
    chunk_sizes = []
    for p in chunk_paths:
        df = pd.read_csv(p)
        df_list.append(df)
        chunk_sizes.append(len(df))
        
    df_all = pd.concat(df_list, ignore_index=True)
    
    df_ids = df_all[["ID_Paziente", "ID"]] if "ID_Paziente" in df_all.columns else None
    cols_to_drop = ["ID_Paziente", "ID"]
    df_train = df_all.drop(columns=[c for c in cols_to_drop if c in df_all.columns])

    feature_cols = [c for c in df_train.columns if c != "Classe"]
    X = df_train[feature_cols].to_numpy(dtype=float)
    y_raw = df_train["Classe"].to_numpy(dtype=int)
    
    classes = np.unique(y_raw[y_raw != -1])
    y = np.copy(y_raw)
    for i, c in enumerate(classes):
        y[y_raw == c] = i

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    global_pca = PCA(n_components=2)
    global_pca.fit(X_scaled)
    X_pca_global = global_pca.transform(X_scaled)

    stream = NumpyStream(
        X=X, 
        y=y, 
        dataset_name="stream_continuo_hat",
        feature_names=feature_cols,
        target_name="Classe"
    )

    model = HoeffdingAdaptiveTree(schema=stream.get_schema())

    predictions = []
    true_labels = []
    correct = 0
    total = 0

    start_time = time.time()
    current_chunk_idx = 0
    instances_processed_in_chunk = 0

    while stream.has_more_instances():
        instance = stream.next_instance()
        
        raw_y_idx = getattr(instance, 'y_index', -1)
        y_true_idx = int(raw_y_idx) if raw_y_idx is not None and not np.isnan(raw_y_idx) else -1
        
        if 0 <= y_true_idx < len(classes):
            y_true = classes[y_true_idx]
            is_labeled = True
        else:
            y_true = -1
            is_labeled = False

        y_pred_idx = model.predict(instance)
        
        if y_pred_idx is None:
            y_pred = classes[0] if len(classes) > 0 else 0
        else:
            pred_idx = int(y_pred_idx)
            y_pred = classes[pred_idx] if 0 <= pred_idx < len(classes) else classes[0]

        predictions.append(y_pred)
        true_labels.append(y_true)

        if is_labeled:
            if y_pred == y_true:
                correct += 1
            total += 1
            model.train(instance)
        else:
            if hasattr(model, 'train_on_unlabeled'):
                model.train_on_unlabeled(instance)
        
        instances_processed_in_chunk += 1

        if instances_processed_in_chunk == chunk_sizes[current_chunk_idx]:
            end_time = time.time()
            chunk_id = chunk_ids[current_chunk_idx]
            path_result = result_dirs[current_chunk_idx]
            os.makedirs(path_result, exist_ok=True)
            
            acc = correct / total if total > 0 else 0
            start_idx = sum(chunk_sizes[:current_chunk_idx])
            end_idx = start_idx + chunk_sizes[current_chunk_idx]
            
            cum_preds = predictions[:end_idx]
            cum_trues = true_labels[:end_idx]
            filtered_trues = [t for t in cum_trues if t != -1]
            filtered_preds = [p for p, t in zip(cum_preds, cum_trues) if t != -1]
            
            if len(filtered_trues) > 0:
                kappa = cohen_kappa_score(filtered_trues, filtered_preds)
                f1_macro = f1_score(filtered_trues, filtered_preds, average="macro", zero_division=0)
                prec_macro = precision_score(filtered_trues, filtered_preds, average="macro", zero_division=0)
                rec_macro = recall_score(filtered_trues, filtered_preds, average="macro", zero_division=0)
            else:
                kappa, f1_macro, prec_macro, rec_macro = 0.0, 0.0, 0.0, 0.0

            print(f"✅ Chunk {chunk_id} Completato! Acc: {acc:.4f} | Prec: {prec_macro:.4f} | Rec: {rec_macro:.4f} | F1: {f1_macro:.4f} | Kappa: {kappa:.4f}")

            pd.DataFrame({
                "Chunk": [chunk_id], "Accuracy": [acc], "Precision": [prec_macro],
                "Recall": [rec_macro], "F1_Macro": [f1_macro], "Kappa": [kappa], 
                "Time_s": [end_time - start_time]
            }).to_csv(os.path.join(path_result, "execution_info.csv"), index=False)
            
            chunk_preds = predictions[start_idx:end_idx]
            chunk_trues = true_labels[start_idx:end_idx]
            
            res_df = pd.DataFrame({"Real_Class": chunk_trues, "Predicted_Class": chunk_preds})
            if df_ids is not None:
                chunk_ids_df = df_ids.iloc[start_idx:end_idx].reset_index(drop=True)
                res_df = pd.concat([chunk_ids_df, res_df], axis=1)
            res_df.to_csv(os.path.join(path_result, "predictions.csv"), index=False)
            
            X_chunk_scaled = X_scaled[start_idx:end_idx]
            X_pca_chunk = global_pca.transform(X_chunk_scaled)
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
            axes[1].set_title(f"Predizioni Modello HAT - Chunk {chunk_id}")
            
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