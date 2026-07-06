import os
import json
import numpy as np
import pandas as pd
import skfuzzy as fuzz


def _interp_memberships_for_vector(vector, feature_names, fuzzy_sets):
    """
    Calcola, per ogni feature, il grado di appartenenza del valore del vettore
    rispetto a tutti i termini fuzzy definiti per quella feature.

    Parameters
    ----------
    vector : 1D np.ndarray
        Valori del prototipo (o di un'istanza) nelle stesse colonne di feature_names.
    feature_names : list[str]
        Nomi/etichette delle feature (stesso ordine di 'vector').
    fuzzy_sets : dict
        Dizionario come restituito dai generatori (x, terms) per ciascuna feature:
        fuzzy_sets[feature] = { "x": <np.ndarray>, "terms": { "<label>": <np.ndarray mf> } }

    Returns
    -------
    dict : { feature -> {term -> degree} }
    """
    results = {}
    for i, feat in enumerate(feature_names):
        if feat not in fuzzy_sets:
            # se i nomi nel df e nel dizionario non coincidono, si prova una fallback con indice
            key = f"Feature_{feat}"
            if key not in fuzzy_sets:
                # nessun set disponibile per questa feature: skip
                continue
            feat_key = key
        else:
            feat_key = feat

        xgrid = fuzzy_sets[feat_key]["x"]
        terms = fuzzy_sets[feat_key]["terms"]
        val = vector[i]

        term_degrees = {}
        for term_label, mf_vals in terms.items():
            deg = float(fuzz.interp_membership(xgrid, mf_vals, val))
            term_degrees[term_label] = deg

        results[feat] = term_degrees
    return results


def _pick_terms(term_degrees_by_feature, top_k=1, min_degree=0.0):
    """
    Seleziona per ogni feature i migliori termini (top_k) con grado >= min_degree.

    Returns
    -------
    dict : { feature -> list[(term_label, degree)] }  (ordinati per degree desc)
    """
    picked = {}
    for feat, term_degs in term_degrees_by_feature.items():
        items = sorted(term_degs.items(), key=lambda x: x[1], reverse=True)
        items = [t for t in items if t[1] >= min_degree]
        picked[feat] = items[:top_k] if top_k > 0 else items
    return picked


def build_rule_for_prototype(
    prototype_vec,
    feature_names,
    fuzzy_sets,
    class_label=None,
    top_k=1,
    min_degree=0.1,
    include_degrees=True,
):
    """
    Costruisce UNA regola IF-THEN a partire da un singolo vettore prototipo.

    Parameters
    ----------
    prototype_vec : 1D np.ndarray
    feature_names : list[str]
    fuzzy_sets : dict
    class_label : int|str|None
        Classe associata al prototipo (se disponibile).
    top_k : int
        Quanti termini per feature includere (1 = quello dominante).
    min_degree : float
        Soglia minima del grado per includere un termine.
    include_degrees : bool
        Se True, scrive tra parentesi il grado (es. High (0.82)).

    Returns
    -------
    rule_text : str
    rule_struct : dict  (comodo per CSV/JSON)
    """
    term_degrees = _interp_memberships_for_vector(prototype_vec, feature_names, fuzzy_sets)
    picked = _pick_terms(term_degrees, top_k=top_k, min_degree=min_degree)

    cond_parts = []
    cond_struct = []
    for feat in feature_names:
        if feat not in picked or len(picked[feat]) == 0:
            continue
        # se top_k=1 mostriamo solo il migliore, altrimenti concateno con OR
        if top_k == 1:
            term, deg = picked[feat][0]
            label = f"{term} ({deg:.2f})" if include_degrees else term
            cond_parts.append(f"{feat} IS {label}")
            cond_struct.append({"feature": feat, "term": term, "degree": float(deg)})
        else:
            alts = []
            alts_struct = []
            for term, deg in picked[feat]:
                label = f"{term} ({deg:.2f})" if include_degrees else term
                alts.append(label)
                alts_struct.append({"feature": feat, "term": term, "degree": float(deg)})
            cond_parts.append(f"{feat} IS ({' OR '.join(alts)})")
            cond_struct.extend(alts_struct)

    right_part = f"class = {class_label}" if class_label is not None else "class = ?"
    rule_text = f"IF " + " AND ".join(cond_parts) + f" THEN {right_part}"

    rule_struct = {
        "if": cond_struct,
        "then": {"class": class_label},
    }
    return rule_text, rule_struct


def generate_rules_for_prototypes(
    prototypes_matrix,
    feature_names,
    fuzzy_sets,
    class_labels=None,
    top_k=1,
    min_degree=0.1,
    include_degrees=True,
):
    """
    Genera regole per una matrice di prototipi.

    Parameters
    ----------
    prototypes_matrix : 2D np.ndarray (n_protos x n_features)
    feature_names : list[str]
    fuzzy_sets : dict
    class_labels : list|np.ndarray|None
        classe per ciascun prototipo; se None, usa None per tutti.
    Returns
    -------
    rules_text : list[str]
    rules_struct : list[dict]
    """
    n_protos = prototypes_matrix.shape[0]
    if class_labels is None:
        class_labels = [None] * n_protos

    rules_text, rules_struct = [], []
    for i in range(n_protos):
        rtxt, rstruct = build_rule_for_prototype(
            prototype_vec=prototypes_matrix[i],
            feature_names=feature_names,
            fuzzy_sets=fuzzy_sets,
            class_label=class_labels[i],
            top_k=top_k,
            min_degree=min_degree,
            include_degrees=include_degrees,
        )
        # aggiungo id_prototipo per tracciabilità
        rstruct["prototype_index"] = i
        rules_text.append(rtxt)
        rules_struct.append(rstruct)
    return rules_text, rules_struct


def save_rules(output_dir, rules_text, rules_struct, file_stub="fuzzy_rules"):
    """
    Salva regole in TXT/CSV/JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    # TXT
    txt_path = os.path.join(output_dir, f"{file_stub}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, rt in enumerate(rules_text, 1):
            f.write(f"Rule {i}: {rt}\n")
    # JSON
    json_path = os.path.join(output_dir, f"{file_stub}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rules_struct, f, indent=2, ensure_ascii=False)
    # CSV “flattened”
    rows = []
    for r in rules_struct:
        class_lbl = r["then"]["class"]
        proto_id = r.get("prototype_index", None)
        # una riga per ciascuna condizione (feature-term)
        for cond in r["if"]:
            rows.append({
                "prototype_index": proto_id,
                "feature": cond["feature"],
                "term": cond["term"],
                "degree": cond["degree"],
                "class": class_lbl
            })
    csv_path = os.path.join(output_dir, f"{file_stub}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return {"txt": txt_path, "json": json_path, "csv": csv_path}
