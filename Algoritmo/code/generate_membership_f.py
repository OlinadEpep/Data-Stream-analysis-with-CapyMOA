import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import os
import json




# Costruisce la membership function su una griglia x arbitraria
def membership_from_shape(x, shape, params):
    """
    Costruisce una funzione di appartenenza fuzzy basata sulla forma specificata.

    Parametri:
    - x: array-like
        Griglia di valori su cui calcolare la funzione di appartenenza.
    - shape: str
        Tipo di funzione di appartenenza da generare. Può essere:
        - "trimf": Triangolare
        - "trapmf": Trapezoidale
        - "trapmf_left": Trapezoidale aperta a sinistra
        - "trapmf_right": Trapezoidale aperta a destra
        - "gaussmf": Gaussiana
        - "sigmf": Sigmoide
        - "smf": Forma a S
        - "zmf": Forma a Z
    - params: list
        Parametri specifici per la funzione di appartenenza:
        - Per "trimf": [a, b, c] (vertici del triangolo)
        - Per "trapmf": [a, b, c, d] (vertici del trapezio)
        - Per "trapmf_left": [a, b, c] o [a, a, b, c]
        - Per "trapmf_right": [a, b, c] o [a, b, c, c]
        - Per "gaussmf": [mean, sigma] (media e deviazione standard)
        - Per "sigmf": [a, c] (pendenza e punto centrale)
        - Per "smf": [a, b] (inizio e fine della transizione a S)
        - Per "zmf": [a, b] (inizio e fine della transizione a Z)
    """
    if shape == "trimf":
        return fuzz.trimf(x, params)
    elif shape == "trapmf":
        return fuzz.trapmf(x, params)
    elif shape == "trapmf_left":
        # Se vengono forniti 3 parametri, li trasformiamo in [a, a, b, c]
        if len(params) == 3:
            new_params = [params[0], params[0], params[1], params[2]]
        else:
            new_params = params
        return fuzz.trapmf(x, new_params)
    elif shape == "trapmf_right":
        # Se vengono forniti 3 parametri, li trasformiamo in [a, b, c, c]
        if len(params) == 3:
            new_params = [params[0], params[1], params[2], params[2]]
        else:
            new_params = params
        return fuzz.trapmf(x, new_params)
    elif shape == "gaussmf":
        return fuzz.gaussmf(x, params[0], params[1])
    elif shape == "sigmf":
        return fuzz.sigmf(x, params[0], params[1])
    elif shape == "smf":
        return fuzz.smf(x, params[0], params[1])
    elif shape == "zmf":
        return fuzz.zmf(x, params[0], params[1])
    else:
        raise ValueError(f"Forma non riconosciuta: {shape}")
    

    
# Funzioni per generare fuzzy set equidistanti di gaussiane

def generate_equidistant_gaussians(colum_list, df, output_folder, n, term_labels):
    """
    Genera fuzzy set equidistanti con funzioni gaussiane.

    Parametri:
    - colum_list: list
        Lista delle colonne del DataFrame per cui generare i fuzzy set.
    - df: pandas.DataFrame
        DataFrame contenente i dati da cui calcolare i range delle feature.
    - output_folder: str
        Cartella in cui salvare i grafici generati.
    - n: int
        Numero di gaussiane da generare per ogni feature.
    - term_labels: list
        Etichette dei termini fuzzy. Se il numero di etichette è inferiore a `n`,
        verranno generate etichette predefinite come "Term1", "Term2", ecc.
    """
    os.makedirs(output_folder, exist_ok=True)
    fuzzy_sets_dict = {}
    for feature in colum_list:
        min_val = df[feature].min()
        max_val = df[feature].max()
        print("min",df[feature].min())
        print('max_val', df[feature].max())
        x = np.linspace(min_val, max_val, 100000)
        centers = [min_val + (j + 1) * (max_val - min_val) / (n + 1) for j in range(n)]
        gaussians = {}
        sigma = (max_val - min_val) / (2 * n)
        for j, center in enumerate(centers):
            mf = fuzz.gaussmf(x, center, sigma)
            if j < len(term_labels):
                term_name = term_labels[j]
            else:
                term_name = f"Term{j + 1}"
            gaussians[term_name] = mf
        fuzzy_sets_dict[feature] = {"x": x, "terms": gaussians}

        plt.figure(figsize=(10, 6))
        for term, mf in gaussians.items():
            plt.plot(x, mf, label=term)
        plt.title(f"Equidistant Gaussian Fuzzy Sets for {feature} (n={n})")
        plt.xlabel("x")
        plt.ylabel("Membership Degree")
        plt.legend()
        plt.grid(True)
        file_path_plot = os.path.join(output_folder, f"{feature}_gaussians.png")
        plt.savefig(file_path_plot)
        plt.close()
        print(f"Grafico salvato per feature '{feature}' in: {file_path_plot}")
    return fuzzy_sets_dict

# Funzioni per generare fuzzy set equidistanti di trapezoidali
def generate_equidistant_trapezoids(colum_list, df, output_folder, n, term_labels):
    """
    Genera fuzzy set equidistanti con funzioni trapezoidali.

    Parametri:
    - colum_list: list
        Lista delle colonne del DataFrame per cui generare i fuzzy set.
    - df: pandas.DataFrame
        DataFrame contenente i dati da cui calcolare i range delle feature.
    - output_folder: str
        Cartella in cui salvare i grafici generati.
    - n: int
        Numero di trapezi da generare per ogni feature.
    - term_labels: list
        Etichette dei termini fuzzy. Se il numero di etichette è inferiore a `n`,
        verranno generate etichette predefinite come "Term1", "Term2", ecc.
    """
    os.makedirs(output_folder, exist_ok=True)
    fuzzy_sets_dict = {}
    for feature in colum_list:
        min_val = df[feature].min()
        max_val = df[feature].max()
        print("min",df[feature].min())
        print('max_val', df[feature].max())
        x = np.linspace(min_val, max_val, 100000)
        delta = (max_val - min_val) / (n + 1)
        trapezoids = {}
        for j in range(n):
            a = min_val + j * delta
            b = a + delta / 2
            c = a + 3 * delta / 2
            d = a + 2 * delta
            mf = fuzz.trapmf(x, [a, b, c, d])
            if j < len(term_labels):
                term_name = term_labels[j]
            else:
                term_name = f"Term{j + 1}"
            trapezoids[term_name] = mf
        fuzzy_sets_dict[feature] = {"x": x, "terms": trapezoids}

        plt.figure(figsize=(10, 6))
        for term, mf in trapezoids.items():
            plt.plot(x, mf, label=term)
        plt.title(f"Equal Equidistant Trapezoidal Fuzzy Sets for {feature} (n={n})")
        plt.xlabel("x")
        plt.ylabel("Membership Degree")
        plt.legend()
        plt.grid(True)
        file_path_plot = os.path.join(output_folder, f"{feature}_trapezoids.png")
        plt.savefig(file_path_plot)
        plt.close()
        print(f"Grafico salvato per feature '{feature}' in: {file_path_plot}")
    return fuzzy_sets_dict

# Funzioni per generare fuzzy set equidistanti di triangolari
def generate_equidistant_triangles(colum_list, df, output_folder, n, term_labels):
    """
    Genera fuzzy set equidistanti con funzioni triangolari.

    Parametri:
    - colum_list: list
        Lista delle colonne del DataFrame per cui generare i fuzzy set.
    - df: pandas.DataFrame
        DataFrame contenente i dati da cui calcolare i range delle feature.
    - output_folder: str
        Cartella in cui salvare i grafici generati.
    - n: int
        Numero di triangoli da generare per ogni feature.
    - term_labels: list
        Etichette dei termini fuzzy. Se il numero di etichette è inferiore a `n`,
        verranno generate etichette predefinite come "Term1", "Term2", ecc.
    """
    os.makedirs(output_folder, exist_ok=True)
    fuzzy_sets_dict = {}
    for feature in colum_list:
        min_val = df[feature].min()
        max_val = df[feature].max()
        print("min",df[feature].min())
        print('max_val', df[feature].max())
        knots = np.linspace(min_val, max_val, n + 2)
        x = np.linspace(min_val, max_val, 100000)
        triangles = {}
        for j in range(n):
            left = knots[j]
            center = knots[j + 1]
            right = knots[j + 2]
            mf = fuzz.trimf(x, [left, center, right])
            if j < len(term_labels):
                term_name = term_labels[j]
            else:
                term_name = f"Term{j + 1}"
            triangles[term_name] = mf
        fuzzy_sets_dict[feature] = {"x": x, "terms": triangles}

        plt.figure(figsize=(10, 6))
        for term, mf in triangles.items():
            plt.plot(x, mf, label=term)
        plt.title(f"Equidistant Triangular Fuzzy Sets for {feature} (n={n})")
        plt.xlabel("x")
        plt.ylabel("Membership Degree")
        plt.legend()
        plt.grid(True)
        file_path_plot = os.path.join(output_folder, f"{feature}_triangles.png")
        plt.savefig(file_path_plot)
        plt.close()
        print(f"Grafico salvato per feature '{feature}' in: {file_path_plot}")
    return fuzzy_sets_dict

# ------------------------------------------------------------------------
# Funzione per generare fuzzy set equidistanti per la configurazione:
# ['trapmf_left', 'trimf', 'trapmf_right']
def generate_equidistant_trap_triangle_trap(colum_list, df, output_folder, term_labels):
    """
    Genera fuzzy set equidistanti con la configurazione:
    - Trapezoidale aperta a sinistra
    - Triangolare
    - Trapezoidale aperta a destra

    Parametri:
    - colum_list: list
        Lista delle colonne del DataFrame per cui generare i fuzzy set.
    - df: pandas.DataFrame
        DataFrame contenente i dati da cui calcolare i range delle feature.
    - output_folder: str
        Cartella in cui salvare i grafici generati.
    - term_labels: list
        Etichette dei termini fuzzy. Devono essere almeno 3 per rappresentare
        i tre tipi di funzioni (sinistra, centro, destra).
    """
    os.makedirs(output_folder, exist_ok=True)
    fuzzy_sets_dict = {}
    for feature in colum_list:
        min_val = df[feature].min()
        max_val = df[feature].max()
        print("min",df[feature].min())
        print('max_val', df[feature].max())
        x = np.linspace(min_val, max_val, 100000)
        p0 = min_val
        p1 = min_val + (max_val - min_val) / 4
        p2 = min_val + (max_val - min_val) / 2
        p3 = min_val + 3 * (max_val - min_val) / 4
        p4 = max_val

        left_mf = fuzz.trapmf(x, [p0, p0, p1, p2])
        triangle_mf = fuzz.trimf(x, [p1, p2, p3])
        right_mf = fuzz.trapmf(x, [p2, p3, p4, p4])

        mixed_mf = {}
        if len(term_labels) >= 3:
            mixed_mf[term_labels[0]] = left_mf
            mixed_mf[term_labels[1]] = triangle_mf
            mixed_mf[term_labels[2]] = right_mf
        else:
            mixed_mf["Term1"] = left_mf
            mixed_mf["Term2"] = triangle_mf
            mixed_mf["Term3"] = right_mf

        fuzzy_sets_dict[feature] = {"x": x, "terms": mixed_mf}

        plt.figure(figsize=(10, 6))
        plt.plot(x, left_mf, label=term_labels[0] if len(term_labels) >= 1 else "Term1")
        plt.plot(x, triangle_mf, label=term_labels[1] if len(term_labels) >= 2 else "Term2")
        plt.plot(x, right_mf, label=term_labels[2] if len(term_labels) >= 3 else "Term3")
        plt.title(f"Mixed Fuzzy Sets for {feature}")
        plt.xlabel("x")
        plt.ylabel("Membership Degree")
        plt.legend()
        plt.grid(True)
        file_path_plot = os.path.join(output_folder, f"{feature}_mixed.png")
        plt.savefig(file_path_plot)
        plt.close()
        print(f"Grafico salvato per feature '{feature}' in: {file_path_plot}")
    return fuzzy_sets_dict


def generate_equidistant_trap_Tripletriangle_trap(colum_list, df, output_folder, term_labels):
    """
    Genera fuzzy set equidistanti con la configurazione:
    - Trapezoidale aperta a sinistra
    - Tre triangolari al centro
    - Trapezoidale aperta a destra

    Parametri:
    - colum_list: list
        Lista delle colonne del DataFrame per cui generare i fuzzy set.
    - df: pandas.DataFrame
        DataFrame contenente i dati da cui calcolare i range delle feature.
    - output_folder: str
        Cartella in cui salvare i grafici generati.
    - term_labels: list
        Etichette dei termini fuzzy. Devono essere almeno 5 per rappresentare
        i cinque tipi di funzioni (sinistra, centro, destra).
    """
    os.makedirs(output_folder, exist_ok=True)
    fuzzy_sets_dict = {}
    for feature in colum_list:
        min_val = df[feature].min()
        max_val = df[feature].max()
        print("min",df[feature].min())
        print('max_val', df[feature].max())
        x = np.linspace(min_val, max_val, 100000)

        # Calcolo dei punti equidistanti per 5 intervalli
        p0 = min_val
        p1 = min_val + (max_val - min_val) / 5
        p2 = min_val + 2 * (max_val - min_val) / 5
        p3 = min_val + 3 * (max_val - min_val) / 5
        p4 = min_val + 4 * (max_val - min_val) / 5
        p5 = max_val

        # Definizione delle funzioni di appartenenza
        left_mf = fuzz.trapmf(x, [p0, p0, p1, p2])
        triangle_mf1 = fuzz.trimf(x, [p1, p2, p3])
        triangle_mf2 = fuzz.trimf(x, [p2, p3, p4])
        triangle_mf3 = fuzz.trimf(x, [p3, p4, p5])
        right_mf = fuzz.trapmf(x, [p4, p5, p5, p5])

        # Assegnazione delle etichette
        mixed_mf = {}
        if len(term_labels) >= 5:
            mixed_mf[term_labels[0]] = left_mf
            mixed_mf[term_labels[1]] = triangle_mf1
            mixed_mf[term_labels[2]] = triangle_mf2
            mixed_mf[term_labels[3]] = triangle_mf3
            mixed_mf[term_labels[4]] = right_mf
        else:
            mixed_mf["Term1"] = left_mf
            mixed_mf["Term2"] = triangle_mf1
            mixed_mf["Term3"] = triangle_mf2
            mixed_mf["Term4"] = triangle_mf3
            mixed_mf["Term5"] = right_mf

        fuzzy_sets_dict[feature] = {"x": x, "terms": mixed_mf}

        # Plot delle funzioni di appartenenza
        plt.figure(figsize=(10, 6))
        plt.plot(x, left_mf, label=term_labels[0] if len(term_labels) >= 1 else "Term1")
        plt.plot(x, triangle_mf1, label=term_labels[1] if len(term_labels) >= 2 else "Term2")
        plt.plot(x, triangle_mf2, label=term_labels[2] if len(term_labels) >= 3 else "Term3")
        plt.plot(x, triangle_mf3, label=term_labels[3] if len(term_labels) >= 4 else "Term4")
        plt.plot(x, right_mf, label=term_labels[4] if len(term_labels) >= 5 else "Term5")
        plt.title(f"Mixed Fuzzy Sets for {feature}")
        plt.xlabel("x")
        plt.ylabel("Membership Degree")
        plt.legend()
        plt.grid(True)
        file_path_plot = os.path.join(output_folder, f"{feature}_trap3triatrap.png")
        plt.savefig(file_path_plot)
        plt.close()
        print(f"Grafico salvato per feature '{feature}' in: {file_path_plot}")
    return fuzzy_sets_dict

# ------------------------------------------------------------------------
# Nuova funzione per la configurazione:
# ['zmf', 'gaussmf', 'smf']
def generate_equidistant_zgaussmf_smf(colum_list, df, output_folder, term_labels):
    """
    Genera fuzzy set equidistanti con la configurazione:
    - Z-shaped a sinistra
    - Gaussiana al centro
    - S-shaped a destra

    Parametri:
    - colum_list: list
        Lista delle colonne del DataFrame per cui generare i fuzzy set.
    - df: pandas.DataFrame
        DataFrame contenente i dati da cui calcolare i range delle feature.
    - output_folder: str
        Cartella in cui salvare i grafici generati.
    - term_labels: list
        Etichette dei termini fuzzy. Devono essere almeno 3 per rappresentare
        i tre tipi di funzioni (sinistra, centro, destra).
    """
    os.makedirs(output_folder, exist_ok=True)
    fuzzy_sets_dict = {}
    for feature in colum_list:
        min_val = df[feature].min()
        max_val = df[feature].max()
        print("min",df[feature].min())
        print('max_val', df[feature].max())
        x = np.linspace(min_val, max_val, 100000)
        p0 = min_val
        p1 = min_val + (max_val - min_val) / 4
        p2 = min_val + (max_val - min_val) / 2
        p3 = min_val + 3 * (max_val - min_val) / 4
        p4 = max_val

        left_mf = fuzz.zmf(x, p0, p1)
        sigma = (max_val - min_val) / 6.0
        center_mf = fuzz.gaussmf(x, p2, sigma)
        right_mf = fuzz.smf(x, p3, p4)

        mixed_mf = {}
        if len(term_labels) >= 3:
            mixed_mf[term_labels[0]] = left_mf
            mixed_mf[term_labels[1]] = center_mf
            mixed_mf[term_labels[2]] = right_mf
        else:
            mixed_mf["Term1"] = left_mf
            mixed_mf["Term2"] = center_mf
            mixed_mf["Term3"] = right_mf

        fuzzy_sets_dict[feature] = {"x": x, "terms": mixed_mf}

        plt.figure(figsize=(10, 6))
        plt.plot(x, left_mf, label=term_labels[0] if len(term_labels) >= 1 else "Term1")
        plt.plot(x, center_mf, label=term_labels[1] if len(term_labels) >= 2 else "Term2")
        plt.plot(x, right_mf, label=term_labels[2] if len(term_labels) >= 3 else "Term3")
        plt.title(f"Z-Gaussian-S Fuzzy Sets for {feature}")
        plt.xlabel("x")
        plt.ylabel("Membership Degree")
        plt.legend()
        plt.grid(True)
        file_path_plot = os.path.join(output_folder, f"{feature}_zgauss_s.png")
        plt.savefig(file_path_plot)
        plt.close()
        print(f"Grafico salvato per feature '{feature}' in: {file_path_plot}")
    return fuzzy_sets_dict


def generate_equidistant_zsigmf_triple_smf(colum_list, df, output_folder, term_labels):
    """
    Genera fuzzy set equidistanti con la configurazione:
    - Z-shaped a sinistra
    - Tre sigmoidi al centro
    - S-shaped a destra

    Parametri:
    - colum_list: list
        Lista delle colonne del DataFrame per cui generare i fuzzy set.
    - df: pandas.DataFrame
        DataFrame contenente i dati da cui calcolare i range delle feature.
    - output_folder: str
        Cartella in cui salvare i grafici generati.
    - term_labels: list
        Etichette dei termini fuzzy. Devono essere almeno 5 per rappresentare
        i cinque tipi di funzioni (sinistra, centro, destra).
    """
    os.makedirs(output_folder, exist_ok=True)
    fuzzy_sets_dict = {}
    for feature in colum_list:
        min_val = df[feature].min()
        max_val = df[feature].max()
        print("min",df[feature].min())
        print('max_val', df[feature].max())
        x = np.linspace(min_val, max_val, 100000)

        # Calcolo dei punti equidistanti per 5 intervalli
        p0 = min_val
        p1 = min_val + (max_val - min_val) / 5
        p2 = min_val + 2 * (max_val - min_val) / 5
        p3 = min_val + 3 * (max_val - min_val) / 5
        p4 = min_val + 4 * (max_val - min_val) / 5
        p5 = max_val

        # Definizione delle funzioni di appartenenza
        left_mf = fuzz.zmf(x, p0, p1)
        sigmf1 = fuzz.sigmf(x, p1, (min_val+max_val)/6)
        sigmf2 = fuzz.sigmf(x, p2, (min_val+max_val)/6)
        sigmf3 = fuzz.sigmf(x, p3, (min_val+max_val)/6)
        right_mf = fuzz.smf(x, p4, p5)

        # Assegnazione delle etichette
        mixed_mf = {}
        if len(term_labels) >= 5:
            mixed_mf[term_labels[0]] = left_mf
            mixed_mf[term_labels[1]] = sigmf1
            mixed_mf[term_labels[2]] = sigmf2
            mixed_mf[term_labels[3]] = sigmf3
            mixed_mf[term_labels[4]] = right_mf
        else:
            mixed_mf["Term1"] = left_mf
            mixed_mf["Term2"] = sigmf1
            mixed_mf["Term3"] = sigmf2
            mixed_mf["Term4"] = sigmf3
            mixed_mf["Term5"] = right_mf

        fuzzy_sets_dict[feature] = {"x": x, "terms": mixed_mf}

        # Plot delle funzioni di appartenenza
        plt.figure(figsize=(10, 6))
        plt.plot(x, left_mf, label=term_labels[0] if len(term_labels) >= 1 else "Term1")
        plt.plot(x, sigmf1, label=term_labels[1] if len(term_labels) >= 2 else "Term2")
        plt.plot(x, sigmf2, label=term_labels[2] if len(term_labels) >= 3 else "Term3")
        plt.plot(x, sigmf3, label=term_labels[3] if len(term_labels) >= 4 else "Term4")
        plt.plot(x, right_mf, label=term_labels[4] if len(term_labels) >= 5 else "Term5")
        plt.title(f"Z-Sigmoid-S Fuzzy Sets for {feature}")
        plt.xlabel("x")
        plt.ylabel("Membership Degree")
        plt.legend()
        plt.grid(True)
        file_path_plot = os.path.join(output_folder, f"{feature}_zsigmoid_s.png")
        plt.savefig(file_path_plot)
        plt.close()
        print(f"Grafico salvato per feature '{feature}' in: {file_path_plot}")
    return fuzzy_sets_dict


def generate_equidistant_zgaussmf_triple_smf(colum_list, df, output_folder, term_labels):
    """
    Genera fuzzy set equidistanti con la configurazione:
    - Z-shaped a sinistra
    - Tre gaussiane al centro
    - S-shaped a destra

    Parametri:
    - colum_list: list
        Lista delle colonne del DataFrame per cui generare i fuzzy set.
    - df: pandas.DataFrame
        DataFrame contenente i dati da cui calcolare i range delle feature.
    - output_folder: str
        Cartella in cui salvare i grafici generati.
    - term_labels: list
        Etichette dei termini fuzzy. Devono essere almeno 5 per rappresentare
        i cinque tipi di funzioni (sinistra, centro, destra).
    """
    os.makedirs(output_folder, exist_ok=True)
    fuzzy_sets_dict = {}
    for feature in colum_list:
        min_val = df[feature].min()
        max_val = df[feature].max()
        print("min",df[feature].min())
        print('max_val', df[feature].max())
        x = np.linspace(min_val, max_val, 100000)

        # Calcolo dei punti equidistanti per 5 intervalli
        p0 = min_val
        p1 = min_val + 1 * (max_val - min_val) / 5
        p2 = min_val + 2 * (max_val - min_val) / 5
        p3 = min_val + 3 * (max_val - min_val) / 5
        p4 = min_val + 4 * (max_val - min_val) / 5
        p5 = max_val

        # Definizione delle funzioni di appartenenza
        left_mf = fuzz.zmf(x, p0, p1)
        sigma = (p5 - p0) / 10  # Calcolo di sigma per garantire equidistanza
        gauss1 = fuzz.gaussmf(x, (p1 + p2) / 2, sigma)
        gauss2 = fuzz.gaussmf(x, (p2 + p3) / 2, sigma)
        gauss3 = fuzz.gaussmf(x, (p3 + p4) / 2, sigma)
        right_mf = fuzz.smf(x, p4, p5)

        # Assegnazione delle etichette
        mixed_mf = {}
        if len(term_labels) >= 5:
            mixed_mf[term_labels[0]] = left_mf
            mixed_mf[term_labels[1]] = gauss1
            mixed_mf[term_labels[2]] = gauss2
            mixed_mf[term_labels[3]] = gauss3
            mixed_mf[term_labels[4]] = right_mf
        else:
            mixed_mf["Term1"] = left_mf
            mixed_mf["Term2"] = gauss1
            mixed_mf["Term3"] = gauss2
            mixed_mf["Term4"] = gauss3
            mixed_mf["Term5"] = right_mf

        fuzzy_sets_dict[feature] = {"x": x, "terms": mixed_mf}

        # Plot delle funzioni di appartenenza
        plt.figure(figsize=(10, 6))
        plt.plot(x, left_mf, label=term_labels[0] if len(term_labels) >= 1 else "Term1")
        plt.plot(x, gauss1, label=term_labels[1] if len(term_labels) >= 2 else "Term2")
        plt.plot(x, gauss2, label=term_labels[2] if len(term_labels) >= 3 else "Term3")
        plt.plot(x, gauss3, label=term_labels[3] if len(term_labels) >= 4 else "Term4")
        plt.plot(x, right_mf, label=term_labels[4] if len(term_labels) >= 5 else "Term5")
        plt.title(f"Z-Gaussian-Gaussian-Gaussian-S Fuzzy Sets for {feature}")
        plt.xlabel("x")
        plt.ylabel("Membership Degree")
        plt.legend()
        plt.grid(True)
        file_path_plot = os.path.join(output_folder, f"{feature}_z3gauss_s.png")
        plt.savefig(file_path_plot)
        plt.close()
        print(f"Grafico salvato per feature '{feature}' in: {file_path_plot}")
    return fuzzy_sets_dict

# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
# Funzione per generare i fuzzy set (logica "classica" o da file JSON)
def generate_fuzzy_sets(df, terms, colum_list, output_folder,
                        shape_config=None,
                        use_config=False, config_file=None):
    """
    Genera fuzzy sets per ogni colonna del dataset.

    Parametri:
    - df: pandas.DataFrame
        DataFrame contenente i dati da cui calcolare i fuzzy set.
    - terms: list
        Etichette dei termini fuzzy (es. ["Low", "Medium", "High"]).
    - colum_list: list
        Lista delle colonne del DataFrame per cui generare i fuzzy set.
    - output_folder: str
        Cartella in cui salvare i grafici generati.
    - shape_config: list, opzionale
        Configurazione delle forme delle funzioni di appartenenza (es. ["trimf", "trapmf"]).
        Se None, utilizza la configurazione predefinita.
    - use_config: bool, opzionale
        Se True, utilizza un file JSON per configurare i fuzzy set.
    - config_file: str, opzionale
        Percorso del file JSON contenente la configurazione dei fuzzy set.
    """
    os.makedirs(output_folder, exist_ok=True)
    fuzzy_sets_dict = {}
    

    # Legge il file di configurazione, se richiesto
    config_data = {}
    if use_config and config_file is not None:
        with open(config_file, "r") as f:
            config_data = json.load(f)

    # ----------------- Modalità classica -----------------
    if not use_config:
        # Se la shape_config è esattamente ['trapmf_left', 'trimf', 'trapmf_right'],
        # richiama la funzione apposita.
        if shape_config == ['trapmf_left', 'trimf', 'trapmf_right']:
            return generate_equidistant_trap_triangle_trap(colum_list, df, output_folder, terms)
        if shape_config == ['trapmf_left','trimf', 'trimf', 'trimf', 'trapmf_right']:
            return generate_equidistant_trap_Tripletriangle_trap(colum_list, df, output_folder, terms)
        # Se la shape_config è esattamente ['zmf', 'gaussmf', 'smf'],
        # richiama la funzione apposita.
        if shape_config == ['zmf', 'gaussmf', 'smf']:
            return generate_equidistant_zgaussmf_smf(colum_list, df, output_folder, terms)
        if shape_config == ['zmf', 'gaussmf', 'gaussmf', 'gaussmf', 'smf']:
            return generate_equidistant_zgaussmf_triple_smf(colum_list, df, output_folder, terms)
        if shape_config is None:
            shape_config = ['trimf'] * len(terms)

        if all(s == "trimf" for s in shape_config):
            n = len(shape_config)
            return generate_equidistant_triangles(colum_list, df, output_folder, n, terms)
        elif all(s == "trapmf" for s in shape_config):
            n = len(shape_config)
            return generate_equidistant_trapezoids(colum_list, df, output_folder, n, terms)
        elif all(s == "gaussmf" for s in shape_config):
            n = len(shape_config)
            return generate_equidistant_gaussians(colum_list, df, output_folder, n, terms)

        # Se la logica non è interamente basata su funzioni equidistanti, si procede per colonna
        # Verrà preso lo stesso minimo e massimo per ogni fuzzy set, quindi è consigliato usare il file di configurazione
        ii = 0
        for column in range(len(colum_list)):
            min_value = df.iloc[:, column].min()
            max_value = df.iloc[:, column].max()
            x = np.linspace(min_value, max_value, 100000)
            print(min_value, max_value)
            fuzzy_sets = {}
            for i, term in enumerate(terms):
                shape = shape_config[i]
                if shape == 'trapmf_left':
                    params = [min_value, min_value, min_value + (max_value-min_value)*0.33, min_value + (max_value-min_value)*0.66]
                    fuzzy_set = fuzz.trapmf(x, params)
                elif shape == 'trapmf_right':
                    params = [min_value + (max_value-min_value)*0.33, min_value + (max_value- min_value)*0.66, max_value, max_value]
                    fuzzy_set = fuzz.trapmf(x, params)
                elif shape == 'trimf':
                    params = [min_value, (min_value+max_value)/2, max_value]
                    fuzzy_set = fuzz.trimf(x, params)
                elif shape == 'gaussmf':
                    center = (min_value + max_value) / 2
                    sigma = (max_value - min_value) / 6.0
                    fuzzy_set = fuzz.gaussmf(x, center, sigma)
                elif shape == 'sigmf':
                    fuzzy_set = fuzz.sigmf(x, 1.0, (min_value+max_value)/2)
                elif shape == 'smf':
                    fuzzy_set = fuzz.smf(x, min_value, max_value)
                elif shape == 'zmf':
                    fuzzy_set = fuzz.zmf(x, min_value, max_value)
                else:
                    raise ValueError(f"Forma non riconosciuta: {shape}")
                fuzzy_sets[term] = fuzzy_set

            fuzzy_sets_dict[f"Feature_{colum_list[ii]}"] = {"x": x, "terms": fuzzy_sets}

            plt.figure(figsize=(10, 6))
            for term, fs_ in fuzzy_sets.items():
                plt.plot(x, fs_, label=term)
            plt.title(f"Fuzzy Sets for Feature {colum_list[ii]}")
            plt.xlabel("x")
            plt.ylabel("Membership Degree")
            plt.legend()
            plt.grid(True)
            file_path_plot = os.path.join(output_folder, f"{colum_list[ii]}.png")
            plt.savefig(file_path_plot)
            plt.close()
            print(f"Grafico salvato per Feature {colum_list[ii]} in: {file_path_plot}")
            ii += 1
        return fuzzy_sets_dict

    # ----------------- Modalità configurazione da file JSON -----------------
    else:
        for feature in colum_list:
            if feature not in config_data:
                print(f"Attenzione: la feature '{feature}' non è presente nel file di configurazione; salto.")
                continue
            config_feature = config_data[feature]
            if isinstance(config_feature, list):
                all_ranges = []
                for conf_entry in config_feature:
                    shape = conf_entry["shape"]
                    params = conf_entry["params"]
                    if "range" in conf_entry:
                        x_min, x_max = conf_entry["range"]
                    else:
                        x_min, x_max = get_min_max_for_shape(shape, params)
                    all_ranges.append((x_min, x_max))
                global_min = min(r[0] for r in all_ranges)
                global_max = max(r[1] for r in all_ranges)
                x_final = np.linspace(global_min, global_max, 100000)
                terms_dict = {}
                for idx, conf_entry in enumerate(config_feature):
                    shape = conf_entry["shape"]
                    params = conf_entry["params"]
                    membership_values = membership_from_shape(x_final, shape, params)
                    term_label = conf_entry.get("label", f"{shape}_{idx + 1}")
                    terms_dict[term_label] = membership_values
                fuzzy_sets_dict[feature] = {"x": x_final, "terms": terms_dict}
                plt.figure(figsize=(10, 6))
                for label, fs_ in terms_dict.items():
                    plt.plot(x_final, fs_, label=label)
                plt.title(f"{feature} - Configurazione multipla")
                plt.xlabel("x")
                plt.ylabel("Membership Degree")
                plt.legend()
                plt.grid(True)
                file_path_plot = os.path.join(output_folder, f"{feature}_config.png")
                plt.savefig(file_path_plot)
                plt.close()
                print(f"Grafico salvato per feature '{feature}' in: {file_path_plot}")
            else:
                shape = config_feature["shape"]
                params = config_feature["params"]
                if "range" in config_feature:
                    x_min, x_max = config_feature["range"]
                else:
                    x_min, x_max = get_min_max_for_shape(shape, params)
                x = np.linspace(x_min, x_max, 100000)
                fs_ = membership_from_shape(x, shape, params)
                term_label = config_feature.get("label", f"{shape}")
                fuzzy_sets_dict[feature] = {"x": x, "terms": {term_label: fs_}}
                plt.figure(figsize=(10, 6))
                plt.plot(x, fs_, label=term_label)
                plt.title(f"{feature} - {term_label} Fuzzy Set (configurato)")
                plt.xlabel("x")
                plt.ylabel("Membership Degree")
                plt.legend()
                plt.grid(True)
                file_path_plot = os.path.join(output_folder, f"{feature}_{shape}.png")
                plt.savefig(file_path_plot)
                plt.close()
                print(f"Grafico salvato per feature '{feature}' in: {file_path_plot}")
        return fuzzy_sets_dict

