# dissfcm_educational_data_mining.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from scipy.spatial.distance import cdist
import json
from CFCM.cfcm import cfcm
from CFCM.cluster_splitting import split_cluster, select_cluster, reconstruction_error
from generate_membership_f import generate_equidistant_gaussians
from fuzzy_rules import generate_rules_for_prototypes, save_rules


def _cluster_to_class_mapping(labels_csv, chunk_df):
    """
    Restituisce dict cluster->classe usando la maggioranza dei pazienti etichettati (Classe != -1)
    assegnati a ciascun cluster nel file labels.csv.
    """
    lbl_df = pd.read_csv(labels_csv)
    if "Cluster" not in lbl_df.columns:
        raise ValueError("labels.csv deve avere colonna 'Cluster'.")

    clusters = lbl_df["Cluster"].to_numpy()
    y = chunk_df["Classe"].to_numpy()

    mapping = {}
    for c in np.unique(clusters):
        m = clusters == c
        vals = y[m]
        vals = vals[vals != -1]
        mapping[int(c)] = int(np.bincount(vals.astype(int)).argmax()) if len(vals) else -1
    return mapping


def test_with_map(X, y, prototypes):
    """
    Assegna ogni campione al prototipo più vicino (distanza euclidea) e
    calcola le metriche usando solo i campioni con etichetta != -1.

    Parametri
    ---------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    prototypes : np.ndarray, shape (n_prototypes, n_features)  # SOLO feature! niente ID in coda

    Ritorna
    -------
    preds : np.ndarray (n_samples,)    # classe predetta via majority per cluster
    cluster_id : np.ndarray (n_samples,)  # indice prototipo più vicino
    metrics_tuple : (acc, prec, rec, f1)
    """
    from scipy.spatial.distance import cdist

    # distanza su tutte le feature: N x d  vs  K x d
    dist = cdist(X, prototypes)
    cluster_id = np.argmin(dist, axis=1)

    # mappa cluster -> classe dominante sui soli y != -1
    mapping = {}
    for c in np.unique(cluster_id):
        vals = y[cluster_id == c]
        vals = vals[vals != -1]
        mapping[c] = np.bincount(vals.astype(int)).argmax() if len(vals) > 0 else -1

    preds = np.array([mapping[c] for c in cluster_id])

    mask = y != -1
    if mask.sum() > 0:
        acc = accuracy_score(y[mask], preds[mask])
        prec, rec, f1, _ = precision_recall_fscore_support(
            y[mask], preds[mask], average="macro", zero_division=0
        )
    else:
        acc = prec = rec = f1 = "N/A"

    return preds, cluster_id, (acc, prec, rec, f1)


def main(chunk_path, chunk_id=0, path_result=None):
    print(f"\n🚀 DSSFCM sul chunk {chunk_id}: {chunk_path}")

    try:
        # 1) Carica chunk
        df = pd.read_csv(chunk_path)
        if "Classe" not in df.columns:
            raise ValueError("Colonna 'Classe' mancante nel chunk.")

        feature_cols = [c for c in df.columns if c not in ["ID_Paziente", "ID", "Classe"]]
        X = df[feature_cols].to_numpy(dtype=float)
        y = df["Classe"].to_numpy()

        # 2) # cluster dinamico = # classi presenti (escludendo -1)
        classi_presenti = np.unique(y[y != -1])
        num_clusters = max(1, len(classi_presenti))
        print(f"📌 Numero cluster usato: {num_clusters}")

        # 3) CFCM iniziale
        U, v, _ = cfcm(
            X=X,
            context=np.ones(len(X)),
            number_of_clusters=num_clusters,
            fuzziness_coefficient=2
        )

        # 4) Cartella risultati
        if path_result is None:
            path_result = os.path.join("results", f"chunk_{chunk_id}")
        os.makedirs(path_result, exist_ok=True)

        # 5) split cluster sicuro (solo se disponibile l'ID nell'ultima colonna di v)
        new_U, new_v = None, None
        if num_clusters > 1:
            try:
                _, cluster_to_split = select_cluster(
                    X, U, v, reconstruction_error, True, chunk_id, path_result
                )
                if v.ndim == 2 and v.shape[1] > 1:
                    prot_ids = list(v[:, -1])
                    if cluster_to_split in prot_ids:
                        idx = prot_ids.index(cluster_to_split)
                        new_U, new_v = split_cluster(X, U[:, idx], v.shape[0], num_clusters)
                    else:
                        print(f"⚠️ Cluster selezionato ({cluster_to_split}) NON presente → salto split")
                else:
                    print("⚠️ Prototipi senza colonna ID → salto split")
            except Exception:
                # qualunque errore di selezione/scrittura (es. percorsi) -> prosegui senza split
                print("⚠️ Split non eseguito (selezione cluster fallita).")

        # 6) Scegli rappresentazione finale
        membership_matrix = new_U if new_U is not None else U
        prototypes = new_v if new_v is not None else v
        labels = np.argmax(membership_matrix, axis=1)

        # *** Protezione: se i prototipi hanno l'ID in ultima colonna, taglialo per i calcoli successivi/visualizzazioni.
        if prototypes.ndim == 2 and prototypes.shape[1] > len(feature_cols):
            prototypes_noid = prototypes[:, :len(feature_cols)]
        else:
            prototypes_noid = prototypes

        # 7) Salva numerici
        pd.DataFrame(membership_matrix).to_csv(os.path.join(path_result, "membership_matrix.csv"), index=False)
        pd.DataFrame(prototypes).to_csv(os.path.join(path_result, "prototypes.csv"), index=False)
        pd.DataFrame(labels, columns=["Cluster"]).to_csv(os.path.join(path_result, "labels.csv"), index=False)

        # 8) Metriche
        preds, _, (acc, prec, rec, f1) = test_with_map(X, y, prototypes_noid)

        if len(np.unique(labels)) > 1:
            sil = silhouette_score(X, labels)
            dav = davies_bouldin_score(X, labels)
            calH = calinski_harabasz_score(X, labels)
        else:
            sil = dav = calH = "N/A"

        pd.DataFrame({
            "Chunk #": [chunk_id],
            "Clusters used": [num_clusters],
            "Accuracy": [acc],
            "Precision": [prec],
            "Recall": [rec],
            "F-Score": [f1],
            "Silhouette": [sil],
            "Davies": [dav],
            "Calinski Harabasz Score": [calH]
        }).to_csv(os.path.join(path_result, "execution_info.csv"), index=False)

        # 9) PCA (visual)
        pca = PCA(n_components=2, random_state=42)
        red_X = pca.fit_transform(X)
        red_P = pca.transform(prototypes_noid)

        df_pca = pd.DataFrame(red_X, columns=["Component1", "Component2"])
        df_pca["Cluster"] = labels
        df_pca["Classe"] = y
        df_pca["Similarity"] = minmax_scale(np.max(membership_matrix, axis=1))
        df_pca.to_csv(os.path.join(path_result, "pca_data.csv"), index=False)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_pca, x="Component1", y="Component2", hue="Cluster", palette="viridis")
        plt.scatter(red_P[:, 0], red_P[:, 1], c="red", s=140, marker="X")
        plt.title(f"PCA Cluster - Chunk {chunk_id}")
        plt.savefig(os.path.join(path_result, "pca_clusters.png"))
        plt.close()

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_pca, x="Component1", y="Component2", hue="Classe", palette="coolwarm")
        plt.scatter(red_P[:, 0], red_P[:, 1], c="black", s=140, marker="X")
        plt.title(f"PCA Classi Reali - Chunk {chunk_id}")
        plt.savefig(os.path.join(path_result, "pca_classes.png"))
        plt.close()

        # 10) Fuzzy explainability: set gaussiani + regole IF–THEN
        fuzzy_out = os.path.join(path_result, "fuzzy")
        os.makedirs(fuzzy_out, exist_ok=True)

        # 10a) Genera 3 gaussiane LOW/MEDIUM/HIGH per ogni feature (range del chunk)
        fuzzy_sets = generate_equidistant_gaussians(
            colum_list=feature_cols,
            df=df[feature_cols],
            output_folder=fuzzy_out,
            n=3,
            term_labels=["LOW", "MEDIUM", "HIGH"]
        )

        # 10b) Allinea i prototipi alle feature
        proto_feature_width = min(prototypes_noid.shape[1], len(feature_cols))
        P = prototypes_noid[:, :proto_feature_width].astype(float)
        feature_cols_for_proto = feature_cols[:proto_feature_width]

        # 10c) Cluster -> THEN class (maggioranza)
        labels_csv = os.path.join(path_result, "labels.csv")
        c2y = _cluster_to_class_mapping(labels_csv, df)

        # 10d) Regole testuali (una per ogni prototipo/cluster)
        rules_txt, rules_struct = generate_rules_for_prototypes(
            prototypes_matrix=P,
            feature_names=feature_cols_for_proto,
            fuzzy_sets=fuzzy_sets,
            class_labels=[c2y.get(k, None) for k in range(P.shape[0])],
            top_k=1,
            min_degree=0.10,
            include_degrees=True
        )

        # 10e) Salva regole + JSON
        with open(os.path.join(fuzzy_out, "fuzzy_rules.txt"), "w", encoding="utf-8") as f:
            for r in rules_txt:
                f.write(r + "\n")

        pd.DataFrame({
            "cluster": list(range(len(rules_txt))),
            "rule": rules_txt,
            "then_class": [c2y.get(k, -1) for k in range(len(rules_txt))]
        }).to_csv(os.path.join(fuzzy_out, "fuzzy_rules.csv"), index=False)

        payload = {
            "feature_order": feature_cols_for_proto,
            "cluster_to_class": c2y,
            "rules": rules_txt
        }
        with open(os.path.join(fuzzy_out, "fuzzy_rules.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"🧠 Regole fuzzy generate in: {fuzzy_out}")
        print(f"✅ Chunk {chunk_id} COMPLETATO ✅\n📁 Risultati: {path_result}")

    except Exception as e:
        print(f"❌ Errore nel chunk {chunk_id}: {e}")
