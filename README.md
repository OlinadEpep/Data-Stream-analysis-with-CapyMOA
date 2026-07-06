# Data Stream Analysis with CapyMOA

Pipeline per la preparazione e l'analisi di dati clinici (pazienti) in uno scenario di **data stream**, con generazione di chunk in diversi scenari (stratificato, con classi assenti, a 10 o 5 blocchi) e successiva elaborazione tramite [CapyMOA](https://capymoa.org/).

## Cosa fa il progetto

Il flusso, gestito da `main.py`, si divide in tre fasi:

1. **Generazione del dataset**
   - Legge tutti i file `.csv`/`.xlsx` dalla cartella `dati/` e li unisce in un unico dataset.
   - Verifica che tutti i file abbiano le stesse colonne.
   - Salva il dataset completo in `pazienti.csv` e una versione ridotta alle sole feature rilevanti in `pazienti_feature.csv`.
   - Calcola e salva la distribuzione delle classi (`distribuzione_classi_completa.csv`).

2. **Generazione dei chunk (simulazione dello stream)**
   I pazienti (identificati da `ID_Paziente`) vengono suddivisi in blocchi (chunk) per simulare l'arrivo progressivo dei dati in uno stream. Vengono generati diversi scenari:
   - **`chunks/`** – suddivisione di base (10 chunk).
   - **`chunks_scenario_1/`** – 10 chunk stratificati per classe.
   - **`chunks_scenario_2/`** – 10 chunk stratificati, ma con classi assenti a rotazione in alcuni chunk (per simulare drift/assenza di concetti).
   - **`chunks_scenario_3/`** – variante a 5 chunk stratificati.
   - **`chunks_scenario_4/`** – variante a 5 chunk con classi assenti.

   Per ogni chunk stratificato vengono inoltre generate **versioni con etichettatura parziale** (100%, 75%, 50%, 25% delle etichette disponibili, le restanti marcate come `-1`), utili per testare scenari di semi-apprendimento su stream.

3. **Elaborazione ed esecuzione esterna**
   Dopo aver generato i chunk, `main.py` lancia in sequenza due script esterni nella cartella `Algoritmo/code/`:
   - `process_chunks.py` – elabora i chunk generati.
   - `validate_results.py` – valida i risultati prodotti.

## Struttura del repository

```
├── main.py                              # Script principale (generazione chunk + orchestrazione)
├── dati/                                 # File sorgente (csv/xlsx) con i dati grezzi dei pazienti
├── pazienti.csv                          # Dataset completo unito
├── pazienti_feature.csv                  # Dataset ridotto alle feature selezionate
├── distribuzione_classi_completa.csv     # Distribuzione globale delle classi
├── chunks/                               # Chunk di base (10 blocchi)
├── chunks_scenario_1/                    # 10 chunk stratificati (+ versioni 100/75/50/25%)
├── chunks_scenario_2/                    # 10 chunk stratificati con classi assenti
├── chunks_scenario_3/                    # 5 chunk stratificati
├── chunks_scenario_4/                    # 5 chunk con classi assenti
└── Algoritmo/                            # Script di elaborazione ed evaluation (chiamati da main.py)
    └── code/
        ├── process_chunks.py
        └── validate_results.py
```

## Requisiti

- Python 3
- `pandas`, `numpy`
- I file di input devono trovarsi in `dati/` con colonne coerenti tra loro e contenere almeno le colonne `ID_Paziente` e `Classe`.

Installazione rapida delle dipendenze:

```bash
pip install pandas numpy
```

## Esecuzione

```bash
python main.py
```

Lo script:
1. genera dataset e chunk in tutte le cartelle previste;
2. esegue automaticamente `Algoritmo/code/process_chunks.py`;
3. esegue automaticamente `Algoritmo/code/validate_results.py`.

## Configurazione

I principali parametri sono definiti in testa a `main.py`:

| Parametro | Descrizione | Default |
|---|---|---|
| `NUM_CHUNK` | Numero di chunk per gli scenari "a 10 blocchi" | `10` |
| `NUM_CHUNK_5` | Numero di chunk per gli scenari "a 5 blocchi" | `5` |
| `SHUFFLE` | Se mescolare i soggetti prima di suddividerli nei chunk | `True` |
| `SEED` | Seed per la riproducibilità | `42` |
| `DESIRED_COLUMNS` | Feature (forma, intensità, centroide) mantenute nei dataset ridotti | vedi codice |

## Note

- Il repository non contiene una descrizione ufficiale né una licenza dichiarata: verificare con l'autore prima di un uso esterno.
- I file `debug-*.log` e `hs_err_pid*.log` sono log residui di un crash della JVM (probabilmente generati durante l'esecuzione di componenti Java sottostanti a CapyMOA) e non sono necessari per l'esecuzione del progetto.
