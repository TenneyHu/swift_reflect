import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree  
import os
import re
import sys

"""errtst.py – Error Latent-Space Visualiser (v2 with Seed-based Classification)
===============================================================================
This script ingests *het_foa* runner logs, projects model responses into an
"error latent space," and classifies them based on proximity to known error
examples.

How it works
------------
1. We parse the latest runs from the `het_foa` log files.
2. A hand-labeled set of "seed examples" for known error types (Hallucination,
   Flawed Code, etc.) is created.
3. The log data and seed examples are combined and then projected into a 2D
   latent space using Sentence-Transformers and UMAP.
4. Each unlabeled point from the logs is classified by finding the `error_type`
   of its nearest labeled seed point in the 2D space.
5. The final plot visualises the now-classified log data, revealing the
   distribution of different error types.
"""

THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
CACHE_SAVER_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(CACHE_SAVER_ROOT)
os.environ.setdefault("VISUALISE_HET_FOA_IMPORT_ONLY", "1")
try:
    from visualise_het_foa import process_log_bundle
except ImportError:
    process_log_bundle = None

def _guess_task_type(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if any(tag in prompt_lower for tag in ["capital", "who", "what", "when", "where"]): return "HotpotQA"
    if "use numbers" in prompt_lower or "game24" in prompt_lower: return "Game24"
    if prompt.strip().startswith("def "): return "HumanEval"
    return "SciBench"

def _extract_records_from_log(log_path: str) -> pd.DataFrame:
    if process_log_bundle is None or not os.path.isfile(log_path): return pd.DataFrame()
    with open(log_path, "r") as f:
        raw_content = re.split(r"#{50,}", f.read())[-1].strip()
    if not raw_content: return pd.DataFrame()
    bundle = process_log_bundle(raw_content)
    graph = bundle["graph"]
    records = []
    for puzzle_idx, timesteps in graph.items():
        if not timesteps: continue
        last_ts = timesteps[-1]
        if last_ts.agent_output_states:
            state = last_ts.agent_output_states[0]
            sd = state.serial_data
            prompt = sd.get("puzzle", "")
            model_resp = str(sd.get("answer", ""))
            ref_answer = str(sd.get("ground_truth", ""))
            records.append({
                "puzzle_idx": puzzle_idx,
                "stage": "After Reflection" if len(timesteps) > 1 else "Before Reflection",
                "task_type": _guess_task_type(prompt), "task_prompt": prompt,
                "model_response": model_resp, "reference_answer": ref_answer,
                "error_type": "Unknown",
            })
    return pd.DataFrame(records)


log_path = os.path.join(CACHE_SAVER_ROOT, "logs", "het_foa_with_reflect.log") 
df_logs = _extract_records_from_log(log_path)

if df_logs.empty:
    print("[errtst] Could not parse log files. The visualization will only show the labeled seed examples.")

seed_examples = [
    # -- Correct --
    {"task_type": "HumanEval", "task_prompt": "def add(a, b):", "model_response": "    return a + b", "reference_answer": "    return a + b", "error_type": "Correct (Code)"},
    {"task_type": "HotpotQA", "task_prompt": "What is the capital of France?", "model_response": "The capital of France is Paris.", "reference_answer": "Paris.", "error_type": "Correct (Text)"},
    {"task_type": "SciBench", "task_prompt": "Calculate the velocity of a car that travels 100m in 4s.", "model_response": "25.0", "reference_answer": "25.0", "error_type": "Correct (Numerical)"},
    # -- Hallucination --
    {"task_type": "SciBench", "task_prompt": "What is the atomic weight of Adamantium?", "model_response": "Adamantium is a rare earth metal with an atomic weight of 256.4 g/mol.", "reference_answer": "Adamantium is a fictional metal.", "error_type": "Hallucination (Factual)"},
    {"task_type": "SciBench", "task_prompt": "What is the value of the cosmological constant?", "model_response": "-500.2", "reference_answer": "The value is very close to zero, approximately 1.11e-52 m^-2.", "error_type": "Hallucination (Numerical)"},
    # -- Unfaithful Reasoning --
    {"task_type": "Game24", "task_prompt": "Use 1, 2, 3, 4 to make 24.", "model_response": "Step 1: 4 * 3 = 12. Step 2: 2 * 1 = 2. Step 3: 12 + 2 = 14. The result is 14.", "reference_answer": "(1+3)*(4+2) = 24", "error_type": "Unfaithful Reasoning"},

    # -- Flawed Code --
    {"task_type": "HumanEval", "task_prompt": "def get_last_item(my_list):", "model_response": "    return my_list[len(my_list)]", "reference_answer": "    return my_list[-1]", "error_type": "Flawed Code (Off-by-one)"},
    {"task_type": "HumanEval", "task_prompt": "def divide(a, b):", "model_response": "    return a / b", "reference_answer": "    if b == 0: return None\n    return a / b", "error_type": "Flawed Code (Edge Case)"},


    {"task_type": "SciBench", "task_prompt": "What is 9 * 9?", "model_response": "82", "reference_answer": "81", "error_type": "Calculation Error"},
    {"task_type": "SciBench", "task_prompt": "Calculate the area of a circle with radius 5. (Use pi=3.14)", "model_response": "78.0", "reference_answer": "78.5", "error_type": "Calculation Error"},
    {"task_type": "SciBench", "task_prompt": "Convert 100 degrees Fahrenheit to Celsius.", "model_response": "38.8", "reference_answer": "37.77", "error_type": "Calculation Error"},
]
df_seeds = pd.DataFrame(seed_examples)


df_seeds['is_seed'] = True
df_logs['is_seed'] = False
df_full_dataset = pd.concat([df_logs, df_seeds], ignore_index=True)


def transform_to_error_latent_space(df: pd.DataFrame, model: SentenceTransformer):
    df['feature_string'] = "PROMPT: " + df['task_prompt'].astype(str) + \
                           "\nMODEL RESPONSE: " + df['model_response'].astype(str) + \
                           "\nREFERENCE_ANSWER: " + df['reference_answer'].astype(str)
    embeddings = model.encode(df['feature_string'].tolist(), show_progress_bar=True)
    reducer = umap.UMAP(n_neighbors=min(15, len(embeddings)-1), n_components=2, min_dist=0.0, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    df['x'] = embeddings_2d[:, 0]
    df['y'] = embeddings_2d[:, 1]
    return df

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# Check if there's enough data to run UMAP
if len(df_full_dataset) > 1:
    df_latent_space = transform_to_error_latent_space(df_full_dataset.copy(), embedding_model)
else:
    print("[errtst] Not enough data to create a visualization. Exiting.")
    sys.exit()


seeds_in_space = df_latent_space[df_latent_space['is_seed'] == True]
logs_in_space = df_latent_space[df_latent_space['is_seed'] == False].copy()

if not seeds_in_space.empty and not logs_in_space.empty:
    print(f"\n[errtst] Classifying {len(logs_in_space)} log entries based on {len(seeds_in_space)} seed examples...")
    kdtree = cKDTree(seeds_in_space[['x', 'y']].values)

    distances, indices = kdtree.query(logs_in_space[['x', 'y']].values)

    logs_in_space['error_type'] = seeds_in_space.iloc[indices]['error_type'].values

    df_classified = pd.concat([seeds_in_space, logs_in_space])
    print("[errtst] Classification complete.")
else:
    df_classified = df_latent_space # If no logs, just plot the seeds


plt.figure(figsize=(14, 12))
sns.scatterplot(
    data=df_classified,
    x='x',
    y='y',
    hue='error_type',
    style='is_seed',
    markers={True: 'X', False: 'o'},
    s=150,
    size='is_seed',
    sizes={True: 250, False: 100},
    palette='deep',
    alpha=0.9
)
plt.title('Error Latent Space: Log Data Classified by Seed Examples', fontsize=16)
plt.xlabel('Latent Dimension 1', fontsize=12)
plt.ylabel('Latent Dimension 2', fontsize=12)
plt.legend(title='Error Type / Source', bbox_to_anchor=(1.05, 1), loc=2)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

print("\n--- Classification Results for Log Data ---")
if not logs_in_space.empty:
    display_cols = ['puzzle_idx', 'error_type', 'model_response']
    print(logs_in_space[display_cols].to_string())
else:
    print("No log data was processed.")


if len(df_classified) >= 4:
    kmeans = KMeans(n_clusters=min(8, len(df_classified)), random_state=42, n_init='auto')
    df_classified['discovered_cluster'] = kmeans.fit_predict(df_classified[['x', 'y']])
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df_classified, x='x', y='y', hue='discovered_cluster', palette='viridis', s=100)
    plt.title('Unsupervised KMeans Clusters (For Discovering New Error Types)', fontsize=16)
    plt.show()