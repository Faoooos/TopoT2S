import os as _os0
_os0.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
import re
from tqdm import tqdm
import torch
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from scipy.stats import wasserstein_distance
import pickle
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, from_networkx
from tslearn.metrics import dtw
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
from tslearn.metrics import dtw
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import networkx as nx


# Import data
df_time = pd.read_csv("data/fitness.csv").drop(columns=['health_condition'])
df_tab1 = pd.read_csv("data/fitness_tab1.csv")
df_tab2 = pd.read_csv("data/fitness_tab2.csv")
df_tab3 = pd.read_csv("data/fitness_tab3.csv")

for _c in df_time.columns:
    if _c in ('date', 'participant_id'):
        continue
    if df_time[_c].dtype == 'object':
        df_time[_c] = df_time[_c].astype('category').cat.codes

# Importing LLM (OpenBioLLM-Llama3)
from transformers import AutoModel, AutoTokenizer

# Local import (example usage); keep the model folder alongside this script
mpnet_path = 'OpenBioLLM-Llama3-8B'
# mpnet_path = 'aaditya/Llama3-OpenBioLLM-8B'  # Hugging Face download (alternative)

def load_mpnet(model_name='OpenBioLLM-Llama3-8B'):
    tokenizer = AutoTokenizer.from_pretrained(mpnet_path)
    model = AutoModel.from_pretrained(mpnet_path)
    return tokenizer, model

tokenizer, model = load_mpnet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

time_features = df_time.columns.tolist()

# List of tabular datasets
tab_files = ['data/fitness_tab1.csv', 'data/fitness_tab2.csv', 'data/fitness_tab3.csv']
df_tabs = {}
tab_features_list = {}

for file in tab_files:
    df = pd.read_csv(file)
    df_tabs[file] = df
    tab_features_list[file] = df.columns.tolist()

def get_feature_embeddings(feature_names: list):
    encoded_input = tokenizer(feature_names, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**encoded_input)
    last_hidden_states = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask']
    embeddings = (last_hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1).unsqueeze(-1)
    return embeddings

# Obtain the embedding vector of time series features
time_embeddings = get_feature_embeddings(time_features)

# Obtain the embedding vectors of the features of the table to be compared
tab_embeddings = {}
for file, features in tab_features_list.items():
    embeddings = get_feature_embeddings(features)
    tab_embeddings[file] = embeddings

# Similarity threshold
SIMILARITY_THRESHOLD = 0.7
similarity_results = {}
cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

for file, tab_emb in tab_embeddings.items():
    
    tab_feat_names = tab_features_list[file]
    matches_for_this_tab = {}
    for i, time_feat in enumerate(time_features):
        current_time_emb = time_embeddings[i].unsqueeze(0).expand(tab_emb.size(0), -1)
        similarity_scores = cosine_similarity(current_time_emb, tab_emb)

        high_sim_indices = (similarity_scores > SIMILARITY_THRESHOLD).nonzero(as_tuple=True)[0]
        if len(high_sim_indices) == 0:
            matches_for_this_tab[time_feat] = []
            continue
        
        found_matches = []
        for idx in high_sim_indices:
            matched_feature = tab_feat_names[idx.item()]
            score = similarity_scores[idx].item()
            found_matches.append((matched_feature, score))

        found_matches.sort(key=lambda x: x[1], reverse=True)
        
        matches_for_this_tab[time_feat] = found_matches

    similarity_results[file] = matches_for_this_tab

new_dfs = {}

# Extract similar columns from the table
for file, matches in similarity_results.items():
    
    original_df = df_tabs[file]
    columns_to_extract = []
    new_column_names = []

    for base_feat, match_list in matches.items():

        if match_list:
            for i, (matched_col, score) in enumerate(match_list):
                columns_to_extract.append(matched_col)

                if len(match_list) == 1:
                    new_column_names.append(base_feat)
                else:
                    new_column_names.append(f"{base_feat}_{i+1}")

    if columns_to_extract:

        new_df = original_df[columns_to_extract].copy()
        new_df.columns = new_column_names
        new_dfs[file] = new_df

# Save the table data as new DF
df_tab1_relevant = new_dfs['data/fitness_tab1.csv']
df_tab2_relevant = new_dfs['data/fitness_tab2.csv']
df_tab3_relevant = new_dfs['data/fitness_tab3.csv']


# Import Logical Alignment LLM (RoBERTa-Large MNLI)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Local import (example usage); keep the model folder alongside this script
mnli_path = 'mnli'
# mnli_path = 'FacebookAI/roberta-large-mnli'  # Hugging Face download (alternative)

def load_mnli(model_name='mnli'):
    tokenizer_mnli = AutoTokenizer.from_pretrained(mnli_path)
    model_mnli = AutoModelForSequenceClassification.from_pretrained(mnli_path)
    return tokenizer_mnli, model_mnli

tokenizer_mnli, model_mnli = load_mnli()
model_mnli.to(device)


id2label = model_mnli.config.id2label
label2id = model_mnli.config.label2id
CONTRADICTION_ID = label2id['CONTRADICTION']

def is_logical_contradiction(feature_base: str, feature_candidate: str, threshold: float = 0.5):
    """
    Use the LLM model to determine whether there is a logical contradiction between two feature names.

    Args:
        feature_base (str): Baseline feature name (e.g., 'is_delayed').
        feature_candidate (str): The feature name to be judged (e.g., 'on_time').
        threshold (float): The probability threshold for determining a contradiction.

    Returns:
        bool: If there is a logical contradiction, return True; otherwise, return False.
    """
    # Construct sentence pairs for NLI tasks
    premise = f"The definition of the data column is '{feature_base}'."
    hypothesis = f"The definition of the data column is '{feature_candidate}'."

    inputs = tokenizer_mnli(premise, hypothesis, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model_mnli(**inputs)
    
    logits = outputs.logits
    
    probabilities = torch.softmax(logits, dim=-1)

    contradiction_prob = probabilities[0][CONTRADICTION_ID].item()
    
    return contradiction_prob > threshold

relevant_dfs = {
    'df_tab1_relevant': df_tab1_relevant,
    'df_tab2_relevant': df_tab2_relevant,
    'df_tab3_relevant': df_tab3_relevant
}


processed_dfs = {}

for df_name, df in relevant_dfs.items():

    if df.empty:
        processed_dfs[df_name] = df
        continue

    base_features = df_time.columns.tolist()
    
    columns_to_drop = []
    
    df_processed = df.copy()

    for col_name in df_processed.columns:
        base_feature = col_name.split('_')[0] if '_' in col_name else col_name
        
        if base_feature not in base_features:
            continue

        is_contradiction = is_logical_contradiction(base_feature, col_name)
        
        if is_contradiction:

            if df_processed[col_name].dtype == bool:
                df_processed[col_name] = ~df_processed[col_name] 

            elif pd.api.types.is_numeric_dtype(df_processed[col_name]) and set(df_processed[col_name].unique()).issubset({0, 1}):

                df_processed[col_name] = 1 - df_processed[col_name]
            else:

                columns_to_drop.append(col_name)
        else:
            print(f"[Logical Consistency] Baseline Feature:'{base_feature}' vs. candidate features:'{col_name}'")
    

    if columns_to_drop:
        df_processed.drop(columns=columns_to_drop, inplace=True)

    processed_dfs[df_name] = df_processed

df_tab1_correlation = processed_dfs['df_tab1_relevant']
df_tab2_correlation = processed_dfs['df_tab2_relevant']
df_tab3_correlation = processed_dfs['df_tab3_relevant']



#  Predefined unit dictionary (unit: conversion factor), it is recommended to expand it according to the dataset domain.
UNIT_CONVERSION = {
    'weight': {
        'kg': 1.0,
        'g': 0.001,
        'mg': 0.000001,
        'lb': 0.453592,
        'oz': 0.0283495
    },
    'length': {
        'm': 1.0,
        'cm': 0.01,
        'mm': 0.001,
        'km': 1000.0,
        'in': 0.0254,
        'ft': 0.3048,
        'mi': 1609.34
    },
    'time': {
        's': 1.0,
        'ms': 0.001,
        'min': 60.0,
        'h': 3600.0,
        'day': 86400.0
    },
    'volume': {
        'l': 1.0,
        'ml': 0.001,
        'm3': 1000.0,
        'gal': 3.78541,
        'qt': 0.946353,
        'pt': 0.473176
    }
}

# Unit Category Mapping
UNIT_CATEGORIES = {
    'kg': 'weight', 'g': 'weight', 'mg': 'weight', 'lb': 'weight', 'oz': 'weight',
    'm': 'length', 'cm': 'length', 'mm': 'length', 'km': 'length', 'in': 'length',
    'ft': 'length', 'mi': 'length',
    's': 'time', 'ms': 'time', 'min': 'time', 'h': 'time', 'day': 'time',
    'l': 'volume', 'ml': 'volume', 'm3': 'volume', 'gal': 'volume', 'qt': 'volume', 'pt': 'volume'
}


def preprocess_column_name(col_name):

    return str(col_name).lower().replace('_', ' ').strip()


def extract_unit_from_column(col_name):
    """Extract unit information from column names"""
    col_name = str(col_name).lower()
    # #Matches units ending with an underscore
    underscore_match = re.search(r'_([a-z]{1,4})$', col_name)
    if underscore_match:
        return underscore_match.group(1)

    # Matches units within parentheses
    bracket_match = re.search(r'\(([a-z]{1,4})\)$', col_name)
    if bracket_match:
        return bracket_match.group(1)

    # Matches space-separated units
    space_match = re.search(r'\s([a-z]{1,4})$', col_name)
    if space_match:
        return space_match.group(1)

    return None

def detect_unit_category(unit):

    return UNIT_CATEGORIES.get(unit.lower(), None)

def are_units_convertible(unit1, unit2):

    category1 = detect_unit_category(unit1)
    category2 = detect_unit_category(unit2)
    return category1 is not None and category1 == category2

def convert_units(value, from_unit, to_unit):

    category = detect_unit_category(from_unit)
    if category is None or not are_units_convertible(from_unit, to_unit):
        return value

    factor_from = UNIT_CONVERSION[category].get(from_unit.lower(), 1.0)
    factor_to = UNIT_CONVERSION[category].get(to_unit.lower(), 1.0)

    return value * (factor_from / factor_to)

def statistical_unit_check(series1, series2):
    """
    Statistical tests can be used to infer whether two sequences might contain the same physical quantity.
    """

    s1 = series1.dropna()
    s2 = series2.dropna()

    if len(s1) < 10 or len(s2) < 10:
        return False, 1.0

    ratio = (s2.mean() / s1.mean()) if s1.mean() != 0 else 1.0

    min_length = min(len(s1), len(s2))

    s1 = s1.iloc[:min_length]
    s2 = s2.iloc[:min_length]

    slope, intercept, r_value, p_value, std_err = stats.linregress(s1, s2)

    if r_value > 0.9 and abs(intercept) < 0.1 * max(abs(s2.mean()), abs(s1.mean())):
        return True, slope
    return False, 1.0

def find_most_similar_time_col(tab_col, time_cols):

    processed_tab_col = preprocess_column_name(tab_col)
    processed_time_cols = [preprocess_column_name(col) for col in time_cols]

    tab_embedding = get_feature_embeddings([processed_tab_col])
    time_embeddings = get_feature_embeddings(processed_time_cols)

    similarities = cosine_similarity(tab_embedding, time_embeddings)
    most_similar_idx = int(torch.argmax(similarities))

    return time_cols[most_similar_idx], similarities[most_similar_idx].item()

def target_minmax_scale(source_series, target_series):
    """
    Scaling the source sequence to the range of the target sequence
    """
    target_min = target_series.min()
    target_max = target_series.max()
    target_range = target_max - target_min

    source_min = source_series.min()
    source_max = source_series.max()
    source_range = source_max - source_min

    if source_range == 0 or target_range == 0:
        return source_series

    scaled_series = (source_series - source_min) / source_range
    scaled_series = scaled_series * target_range + target_min

    return scaled_series


def align_and_standardize_units(df_time, df_tab_aligned, similarity_threshold=0.6):

    processed_df = df_tab_aligned.copy()
    time_cols = df_time.columns.tolist()

    common_cols = set(processed_df.columns) & set(time_cols)

    for tab_col in df_tab_aligned.columns:
        if tab_col in common_cols:
            time_col = tab_col
            similarity = 1.0
        else:

            time_col, similarity = find_most_similar_time_col(tab_col, time_cols)
            if similarity < similarity_threshold:

                processed_df.drop(columns=[tab_col], inplace=True)
                continue

            if time_col in processed_df.columns:
                processed_df.drop(columns=[tab_col], inplace=True)
                continue

            processed_df = processed_df.rename(columns={tab_col: time_col})

        if not np.issubdtype(processed_df[time_col].dtype, np.number):
            processed_df.drop(columns=[time_col], inplace=True)
            continue

        time_unit = extract_unit_from_column(time_col)
        tab_unit = extract_unit_from_column(tab_col)  

        if time_unit and tab_unit:
            if are_units_convertible(time_unit, tab_unit):

                processed_df[time_col] = processed_df[time_col].apply(
                    lambda x: convert_units(x, tab_unit, time_unit)
                )

            else:
                processed_df[time_col] = target_minmax_scale(
                    processed_df[time_col],
                    df_time[time_col]
                )


        elif time_unit and not tab_unit:
            is_proportional, factor = statistical_unit_check(
                processed_df[time_col], df_time[time_col]
            )

            if is_proportional:
                processed_df[time_col] = processed_df[time_col] * factor

            else:
                processed_df[time_col] = target_minmax_scale(
                    processed_df[time_col],
                    df_time[time_col]
                )

        elif not time_unit and tab_unit:
            continue

        else:

            is_proportional, factor = statistical_unit_check(
                processed_df[time_col], df_time[time_col]
            )

            if is_proportional and abs(factor - 1.0) > 0.01:
                processed_df[time_col] = processed_df[time_col] * factor

            else:
                processed_df[time_col] = target_minmax_scale(
                    processed_df[time_col],
                    df_time[time_col]
                )

    return processed_df

# Execution unit alignment and standardization
df_tab1_final = align_and_standardize_units(df_time, df_tab1_correlation)
df_tab2_final = align_and_standardize_units(df_time, df_tab2_correlation)
df_tab3_final = align_and_standardize_units(df_time, df_tab3_correlation)



def robust_clean_df(df, name):
    df = df.copy().replace([np.inf, -np.inf], np.nan)
    if df.isnull().values.any():
        df = df.fillna(df.mean().fillna(0))
    constant_cols = [col for col in df.columns if df[col].std() <= 1e-8]
    if constant_cols:
        for col in constant_cols:
            df[col] = df[col] + np.random.normal(0, 1e-6, size=len(df))
    return df

print("\nDeep-cleaning tabular data...")
scaler_tab = StandardScaler()
tabular_data_list = []
for i, df_raw in enumerate([df_tab1_final, df_tab2_final, df_tab3_final]):
    df_cleaned = robust_clean_df(df_raw, f"Tab {i+1}")
    df_norm = pd.DataFrame(scaler_tab.fit_transform(df_cleaned), columns=df_cleaned.columns)
    tabular_data_list.append(df_norm)

all_features = sorted(list(set().union(*(df.columns for df in tabular_data_list))))
# CMH/InfoNCE alignment
train_days, _ = train_test_split(df_time['participant_id'].unique(), test_size=0.4, random_state=int(_os0.environ.get('SPLIT', '42')))
train_df = df_time[df_time['participant_id'].isin(train_days)]
df_time_selected = train_df[all_features]
scaler_time = StandardScaler()
df_time_normalized = pd.DataFrame(scaler_time.fit_transform(df_time_selected), columns=all_features)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TSColumnEncoder(nn.Module):
    def __init__(self, seq_len, d_model=128):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True), num_layers=2)
    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        x = self.pos_encoder(x)
        return self.transformer(x).mean(dim=1)

class SharedTabularProjector(nn.Module):
    def __init__(self, fixed_len, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(fixed_len, 256), nn.GELU(), nn.Linear(256, latent_dim))
    def forward(self, x):
        return self.net(x)

class RobustInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, z_ts, z_tab, mask):
        z_ts, z_tab = F.normalize(z_ts, dim=1), F.normalize(z_tab, dim=1)
        logits = torch.matmul(z_ts, z_tab.t()) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits_stable = logits - logits_max.detach()
        log_prob_denom = torch.log(torch.exp(logits_stable).sum(dim=1) + 1e-8)
        
        valid_indices = torch.where(mask.sum(dim=1) > 0)[0]
        if len(valid_indices) == 0: return torch.tensor(0.0, requires_grad=True).to(z_ts.device)
        
        loss = 0.0
        for i in valid_indices:
            pos_logits = logits_stable[i][mask[i]]
            loss += -(pos_logits - log_prob_denom[i]).mean()
        return loss / len(valid_indices)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIXED_TAB_LEN = 256
latent_dim = 128

filtered_tab_features = []
filtered_tab_names = []

print("\nBuilding global feature pool...")
for df in tabular_data_list:
    if df.shape[1] < 1: continue
    for col in df.columns:
        col_data = torch.tensor(df[col].values, dtype=torch.float32).view(1, 1, -1)
        col_resized = F.interpolate(col_data, size=FIXED_TAB_LEN, mode='linear', align_corners=False).view(-1)
        filtered_tab_features.append(col_resized)
        filtered_tab_names.append(col)

tab_data_tensor = torch.stack(filtered_tab_features).to(device)
ts_feature_names = list(df_time_normalized.columns)

active_ts_indices = [i for i, name in enumerate(ts_feature_names) if name in filtered_tab_names]
_MAX_CMH_SEQ = 3000
_cmh_step = max(1, len(df_time_normalized) // _MAX_CMH_SEQ)
df_time_normalized_cmh = df_time_normalized.iloc[::_cmh_step, :]
ts_data_tensor = torch.tensor(df_time_normalized_cmh.values.T[active_ts_indices], dtype=torch.float32).to(device)
active_ts_names = [ts_feature_names[i] for i in active_ts_indices]

positive_mask = torch.zeros((len(active_ts_names), len(filtered_tab_names)), dtype=torch.bool).to(device)
for i, t_name in enumerate(active_ts_names):
    for j, f_name in enumerate(filtered_tab_names):
        if t_name == f_name: positive_mask[i, j] = True



df_tab1_final1 = pd.DataFrame(scaler_tab.fit_transform(df_tab1_final), columns=df_tab1_final.columns)

df_tab2_final1 = pd.DataFrame(scaler_tab.fit_transform(df_tab2_final), columns=df_tab2_final.columns)

df_tab3_final1 = pd.DataFrame(scaler_tab.fit_transform(df_tab3_final), columns=df_tab3_final.columns)

tabular_data_list = [df_tab1_final1, df_tab2_final1, df_tab3_final1]

global_tab_features = [] 
global_tab_names = []    
global_tab_sources = []  

for t_idx, df_tab in enumerate(tabular_data_list):
    for col in df_tab.columns:
        col_data = torch.tensor(df_tab[col].values, dtype=torch.float32).view(1, 1, -1)
        col_resized = F.interpolate(col_data, size=FIXED_TAB_LEN, mode='linear', align_corners=False)
        
        global_tab_features.append(col_resized.view(-1))
        global_tab_names.append(col)
        global_tab_sources.append(t_idx)


ts_encoder = TSColumnEncoder(seq_len=ts_data_tensor.size(1), d_model=latent_dim).to(device)
tab_projector = SharedTabularProjector(fixed_len=FIXED_TAB_LEN, latent_dim=latent_dim).to(device)
criterion = RobustInfoNCELoss(temperature=0.07).to(device)
optimizer = optim.AdamW(list(ts_encoder.parameters()) + list(tab_projector.parameters()), lr=1e-3)


for epoch in range(10):
    optimizer.zero_grad()
    z_ts = ts_encoder(ts_data_tensor)
    z_tab = tab_projector(tab_data_tensor)
    
    loss = criterion(z_ts, z_tab, positive_mask)
    
    if torch.isnan(loss):
        print(f"NaN at Epoch {epoch}. TS norm: {z_ts.norm().item()}, Tab norm: {z_tab.norm().item()}")
        break
        
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ts_encoder.parameters(), 1.0)
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")


ts_encoder.eval()
tab_projector.eval()

with torch.no_grad():
    Z_tab_final = tab_projector(tab_data_tensor).cpu()


fused_graph = nx.Graph()
alpha = 0.5 
threshold = 0.1

all_unique_features = sorted(list(set(global_tab_names)))
for node_name in all_unique_features:
    vals = []
    for t_idx, df_tab in enumerate(tabular_data_list):
        if node_name in df_tab.columns:
            vals.append(df_tab[node_name].mean())
    avg_value = np.mean(vals) if vals else 0.0
    fused_graph.add_node(node_name, value=avg_value)



feat_dict = {name: [] for name in all_unique_features}
for idx, name in enumerate(global_tab_names):
    feat_dict[name].append({
        'table_idx': global_tab_sources[idx],
        'embedding': Z_tab_final[idx],
        'raw_data': tabular_data_list[global_tab_sources[idx]][name].values
    })

for i in range(len(all_unique_features)):
    for j in range(i + 1, len(all_unique_features)):
        node_i = all_unique_features[i]
        node_j = all_unique_features[j]
        
        info_i_list = feat_dict[node_i]
        info_j_list = feat_dict[node_j]
        
        emb_i_avg = torch.stack([x['embedding'] for x in info_i_list]).mean(dim=0)
        emb_j_avg = torch.stack([x['embedding'] for x in info_j_list]).mean(dim=0)
        cos_sim = F.cosine_similarity(emb_i_avg.unsqueeze(0), emb_j_avg.unsqueeze(0)).item()
        
        common_tables = set([x['table_idx'] for x in info_i_list]) & set([x['table_idx'] for x in info_j_list])
        
        pearson_list = []
        for t_idx in common_tables:
            raw_i = next(x['raw_data'] for x in info_i_list if x['table_idx'] == t_idx)
            raw_j = next(x['raw_data'] for x in info_j_list if x['table_idx'] == t_idx)
            p_corr = np.corrcoef(raw_i, raw_j)[0, 1]
            if not np.isnan(p_corr):
                pearson_list.append(abs(p_corr))
                
        if len(pearson_list) > 0:
            A_base = np.mean(pearson_list)
            final_weight = alpha * A_base + (1 - alpha) * cos_sim
        else:
            final_weight = cos_sim
            

        if final_weight > threshold:
            fused_graph.add_edge(node_i, node_j, weight=final_weight)

config = {
    'batch_size': 16,
    'hidden_dim': 96,
    'lr': 1e-3,
    'attention_lr': 1e-3,
    'weight_decay': 1e-2,
    'epochs': 60,
    'label_smoothing': 0.1,
    'dropout': 0.5,
    'lambda_init': 0.05,
    'use_learned_adj': False,
    'skip_graph': False,
    'keep_node': True,
    'use_last_step': True,
    'drop_static_nodes': True,
    'node_columns': ['weight_kg', 'bmi', 'activity_type', 'duration_minutes', 'intensity',
                     'calories_burned', 'daily_steps', 'avg_heart_rate', 'sleep_hours',
                     'stress_level', 'hydration_level', 'fitness_level',
                     'age', 'gender', 'height_cm', 'resting_heart_rate',
                     'blood_pressure_systolic', 'blood_pressure_diastolic', 'smoking_status'],
    'n_channels': 2,
    'node_agg': 'linear',
    'graph_mode': 'gated_fusion',
    'dy_temp': 2.0,
    'dy_no_self': True,
    'dy_entropy_reg': 0.0,
    'dy_entropy_target': 0.9,
    'gat_heads': 6,
    'gru_layers': 3,
}


import os as _os
_SEED = int(_os.environ.get('SEED', '42'))
_SPLIT = int(_os.environ.get('SPLIT', '42'))
_FREE_GPU_SEED = _os.environ.get('FREE_GPU_SEED', '1') == '1'
torch.manual_seed(_SEED)
if not _FREE_GPU_SEED:
    torch.cuda.manual_seed_all(_SEED)
np.random.seed(_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

df_time = pd.read_csv("data/fitness.csv").drop(columns=['health_condition'])

def convert_to_timesteps(df, time_col='date', patient_id_col='participant_id'):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df['timestep'] = df.groupby(patient_id_col)[time_col].rank(method='first').astype(int) - 1
    df.drop(columns=[time_col], inplace=True)
    return df

def auto_encode_features(df, skip_columns=None, max_unique_for_label=20):
    df = df.copy()
    skip_columns = skip_columns or []
    for col in df.columns:
        if col in skip_columns:
            print(f"jump: {col}")
            continue
        dtype = df[col].dtype
        if dtype == 'bool':
            df[col] = df[col].astype(int)
        elif dtype == 'object' or isinstance(df[col].iloc[0], str):
            num_unique = df[col].nunique()
            if 1 < num_unique <= max_unique_for_label:
                df[col] = df[col].astype('category').cat.codes
            elif num_unique > max_unique_for_label:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[col], inplace=True)
    return df


df_time = df_time.dropna()
df_time = convert_to_timesteps(df_time)
df_time = auto_encode_features(df_time, skip_columns=['participant_id'])

_label_map = df_time.sort_values('timestep').groupby('participant_id')['endurance_level'].last()

min_timesteps = df_time.groupby('participant_id')['timestep'].max().min()
df_time = df_time[df_time['timestep'] <= min_timesteps]

def prepare_data(df_time):
    samples = []
    labels = []
    for pid, group in df_time.groupby('participant_id'):
        time_steps = sorted(group['timestep'].unique())
        seq = []
        for t in time_steps:
            data_t = group[group['timestep'] == t].drop(columns=['participant_id', 'timestep', 'endurance_level'])
            seq.append(data_t.values)
        sample = np.concatenate(seq, axis=0)
        label = _label_map[pid]
        samples.append(sample)
        labels.append(label)
    X = np.stack(samples)
    y = np.array(labels)
    return X, y

X, y = prepare_data(df_time)
X_tensor = torch.tensor(X, dtype=torch.float32)
# 60:10:30 subject-wise split loaded from precomputed indices (SPLIT=42)
idx_train = np.load("train_idx.npy")
idx_val = np.load("val_idx.npy")
idx_test = np.load("test_idx.npy")
_edges = np.quantile(y[idx_train], [0.2, 0.4, 0.6, 0.8])
y = np.digitize(y, _edges, right=False).astype(int)
y_tensor = torch.tensor(y, dtype=torch.long)
_, y_tensor = torch.unique(y_tensor, return_inverse=True)

_static_cols = ['age', 'gender', 'height_cm', 'resting_heart_rate',
                'blood_pressure_systolic', 'blood_pressure_diastolic', 'smoking_status']
_numeric_static = {'age', 'height_cm', 'resting_heart_rate',
                   'blood_pressure_systolic', 'blood_pressure_diastolic'}
_static_raw = []
for _pid, _grp in df_time.groupby('participant_id'):
    _first = _grp.iloc[0]
    _static_raw.append([float(_first[c]) for c in _static_cols])
_static_raw = np.array(_static_raw)  # (B, 7)
_static_parts = []
for _ci, _c in enumerate(_static_cols):
    if _c in _numeric_static:  # numeric static columns standardized
        _v = _static_raw[:, _ci:_ci+1]
        _v = (_v - _v[idx_train].mean()) / (_v[idx_train].std() + 1e-6)
        _static_parts.append(_v)
    else:
        _vals = _static_raw[:, _ci].astype(int)
        _n = int(_vals.max()) + 1
        _static_parts.append(np.eye(_n)[_vals])
static_ctx = np.concatenate(_static_parts, axis=1).astype(np.float32)  # (B, 5+2+3=10)
static_ctx_tensor = torch.tensor(static_ctx)
print(f'[static-ctx] static context dim={static_ctx.shape[1]} (age+height_cm+resting_hr+bp_sys+bp_dia + gender + smoking_status)')

graph_node_names = list(fused_graph.nodes())
num_nodes = len(graph_node_names) 


x_feature_names = [c for c in df_time.columns if c not in ('participant_id', 'timestep', 'endurance_level')]
graph_node_names = x_feature_names
num_nodes = len(graph_node_names)


X_tensor_aligned = X_tensor

if config.get('node_columns') is not None:
    keep_idx = [x_feature_names.index(c) for c in config['node_columns'] if c in x_feature_names]
    X_tensor_aligned = X_tensor_aligned[:, :, keep_idx]
    graph_node_names = [graph_node_names[i] for i in keep_idx]
    num_nodes = len(graph_node_names)
elif config.get('drop_static_nodes', False):
    diff_sum = (X_tensor_aligned[:, 1:, :] - X_tensor_aligned[:, :-1, :]).abs().sum(dim=(0, 1))
    dyn_idx = torch.nonzero(diff_sum > 0).squeeze(-1)
    X_tensor_aligned = X_tensor_aligned[:, :, dyn_idx]
    graph_node_names = [graph_node_names[i] for i in dyn_idx.tolist()]
    num_nodes = len(graph_node_names)

print(f'[nodes] N={num_nodes}: {graph_node_names}')

B, T, N = X_tensor_aligned.shape

if config.get('n_channels', 2) == 2:
    X_diff = torch.zeros_like(X_tensor_aligned)
    X_diff[:, 1:, :] = X_tensor_aligned[:, 1:, :] - X_tensor_aligned[:, :-1, :]
    X_node_view = torch.stack([X_tensor_aligned, X_diff], dim=-1)  # (B, T, N, 2)
else:
    X_node_view = X_tensor_aligned.unsqueeze(-1)  # (B, T, N, 1)


X_flat = X_tensor_aligned[idx_train].reshape(-1, N).cpu().numpy()
adj_pearson = np.abs(np.nan_to_num(np.corrcoef(X_flat, rowvar=False)))  # (N,N) data correlation

ts_idx = [time_features.index(f) for f in graph_node_names]
_emb_norm = F.normalize(time_embeddings, dim=-1)
_sem = (_emb_norm @ _emb_norm.T)[ts_idx][:, ts_idx].cpu().numpy()
adj_semantic = np.clip(_sem, 0.0, 1.0)

adj_matrix = 0.5 * adj_pearson + 0.5 * adj_semantic  # Eq.14: alpha*A_base + (1-alpha)*cos_sim

name2idx = {name: i for i, name in enumerate(graph_node_names)}
_n_fused = 0
for u, v, w in fused_graph.edges(data='weight'):
    if u in name2idx and v in name2idx:
        i, j = name2idx[u], name2idx[v]
        adj_matrix[i, j] = w
        adj_matrix[j, i] = w
        _n_fused += 1
print(f'[static-adj] semantic+Pearson fused ({N} nodes), fused_graph fine edges override {_n_fused}')
static_adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).cuda()

feature_names = list(df_time_selected.columns)

x_mean = X_node_view[idx_train].mean(dim=(0, 1), keepdim=True)
x_std = X_node_view[idx_train].std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
X_node_view = (X_node_view - x_mean) / x_std
print(f'[X-norm] shape={tuple(X_node_view.shape)} ch0 max={X_node_view[:, :, :, 0].abs().max().item():.1f}')
X_node_view = X_node_view.clamp(-4.0, 4.0)

train_loader = DataLoader(TensorDataset(X_node_view[idx_train], y_tensor[idx_train], static_ctx_tensor[idx_train]),
                          batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(TensorDataset(X_node_view[idx_val], y_tensor[idx_val], static_ctx_tensor[idx_val]),
                        batch_size=config['batch_size'])
test_loader = DataLoader(TensorDataset(X_node_view[idx_test], y_tensor[idx_test], static_ctx_tensor[idx_test]),
                         batch_size=config['batch_size'])


class DGTI_Model(nn.Module):
    def __init__(self, num_nodes, hidden_dim, num_classes, static_adj, in_channels=2, static_dim=0):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.register_buffer('static_adj', static_adj)  # (N, N)

        self.g0 = nn.Parameter(torch.tensor(1.0))
        self.reg_lambda = nn.Parameter(torch.tensor([config['lambda_init']]))

        self.W0 = nn.Linear(in_channels, hidden_dim)
        self.norm_h = nn.LayerNorm(hidden_dim)

        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(hidden_dim, 2)

        self.W_agg = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.use_learned_adj = config['use_learned_adj']
        self.skip_graph = config['skip_graph']
        self.dynamic_adj_learner = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.01)

        self.graph_mode = config.get('graph_mode', 'learned_fusion')  # learned_fusion / fixed_fusion / dynamic_only
        self.dy_temp = config.get('dy_temp', 1.0)
        self.dy_no_self = config.get('dy_no_self', False)
        self.dy_entropy = None

        for _m in (self.W_Q, self.W_K, self.W_V, self.W_agg):
            nn.init.normal_(_m.weight, std=0.02)
        nn.init.zeros_(self.W_o.weight)
        nn.init.zeros_(self.W_o.bias)
        print(f'[init-check] W_Q fro={self.W_Q.weight.norm().item():.4f} max={self.W_Q.weight.abs().max().item():.4f} '
              f'W_o fro={self.W_o.weight.norm().item():.4f} bias={self.W_o.bias.tolist()}')

        _heads = config.get('gat_heads', 4)
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim // _heads, heads=_heads, dropout=0.2, edge_dim=1)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=0.2, edge_dim=1)

        self.norm = nn.LayerNorm(hidden_dim)

        self.keep_node = config['keep_node']
        self.node_agg = config.get('node_agg', 'concat')
        if self.node_agg == 'linear':
            self.node_proj = nn.Linear(hidden_dim * num_nodes, hidden_dim)
        elif self.node_agg == 'attn':
            self.node_attn = nn.Linear(hidden_dim, 1)
        gru_in_dim = hidden_dim if self.node_agg in ('linear', 'attn') else (hidden_dim * num_nodes if self.keep_node else hidden_dim)
        self.gru = nn.GRU(gru_in_dim, hidden_dim, batch_first=True, num_layers=config.get('gru_layers', 2), dropout=config['dropout'])

        self.time_attention = nn.Linear(hidden_dim, 1)

        self.static_dim = static_dim
        self.static_proj = nn.Linear(static_dim, hidden_dim)
        self.static_gate = nn.Linear(hidden_dim + static_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x_seq, static_ctx=None):
        B, T, N, F_dim = x_seq.shape
        device = x_seq.device
        _cd = getattr(self, 'collect_diag', False)

        h_all = self.norm_h(self.W0(x_seq))

        step_representations = []
        gating_values = []
        _diag_alpha_static = []
        _diag_alpha_dy = []
        _diag_ady_max = []
        _diag_ady_ent = []
        _diag_attn_max = []
        _diag_attn_min = []
        _diag_q_norm = []
        _diag_k_norm = []
        _diag_h_norm = []

        for t in range(T):
            h_t = h_all[:, t, :, :]  # (B, N, d)
            if _cd: _diag_h_norm.append(h_t.norm(dim=-1).mean().item())

            if self.skip_graph:
                if self.node_agg == 'linear':
                    step_representations.append(self.node_proj(h_t.reshape(B, N * self.hidden_dim)))  # (B, d)
                elif self.node_agg == 'attn':
                    scores = self.node_attn(h_t).squeeze(-1)  # (B, N)
                    w = torch.softmax(scores, dim=1)
                    step_representations.append(torch.bmm(w.unsqueeze(1), h_t).squeeze(1))  # (B, d)
                elif self.keep_node:
                    step_representations.append(h_t.reshape(B, N * self.hidden_dim))
                else:
                    step_representations.append(h_t.mean(dim=1))
                continue

            if self.use_learned_adj:
                dyn_adj = F.relu(self.dynamic_adj_learner + self.dynamic_adj_learner.T)  # (N, N)
                A_dy = dyn_adj.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
                if _cd: _diag_ady_max.append(A_dy.max(dim=-1).values.mean().item())

                gt = torch.clamp(self.g0, min=0.0) * torch.exp(-torch.clamp(self.reg_lambda, min=0.01) * t)
                if _cd: gating_values.append(gt.item())

                A_fused = gt * self.static_adj.unsqueeze(0) + (1.0 - gt) * A_dy  # (B, N, N)
                if _cd: _diag_alpha_static.append(gt.item())
                if _cd: _diag_alpha_dy.append(1.0 - gt.item())
            else:
                Q = self.W_Q(h_t)  # (B, N, d)
                K = self.W_K(h_t)  # (B, N, d)
                if _cd: _diag_q_norm.append(Q.norm(dim=-1).mean().item())
                if _cd: _diag_k_norm.append(K.norm(dim=-1).mean().item())
                attn_logits = torch.bmm(Q, K.transpose(1, 2)) / (math.sqrt(self.hidden_dim) * self.dy_temp)
                if self.dy_no_self:
                    _no_self = torch.eye(N, device=device, dtype=torch.bool)
                    attn_logits = attn_logits.masked_fill(_no_self, float('-inf'))
                if _cd: _diag_attn_max.append(attn_logits.max().item())
                if _cd: _diag_attn_min.append(attn_logits.min().item())
                A_dy = torch.softmax(attn_logits, dim=-1)  # (B, N, N)
                if _cd: _diag_ady_max.append(A_dy.max(dim=-1).values.mean().item())
                self.dy_entropy = -(A_dy * torch.log(A_dy + 1e-9)).sum(dim=-1).mean()
                if _cd: _diag_ady_ent.append(self.dy_entropy.item())

                gt = torch.clamp(self.g0, min=0.0) * torch.exp(-torch.clamp(self.reg_lambda, min=0.01) * t)
                if _cd: gating_values.append(gt.item())

                if self.graph_mode == 'dynamic_only':
                    A_fused = gt * A_dy
                    _diag_alpha_static.append(0.0)
                    if _cd: _diag_alpha_dy.append(gt.item())
                elif self.graph_mode == 'fixed_fusion':
                    A_fused = 0.5 * gt * self.static_adj.unsqueeze(0) + 0.5 * A_dy
                    _diag_alpha_static.append(0.5)
                    _diag_alpha_dy.append(0.5)
                elif self.graph_mode == 'gated_fusion':
                    A_fused = gt * self.static_adj.unsqueeze(0) + (1.0 - gt) * A_dy
                    if _cd: _diag_alpha_static.append(gt.item())
                    if _cd: _diag_alpha_dy.append(1.0 - gt.item())
                else:  # learned_fusion
                    V = self.W_V(h_t)  # (B, N, d)
                    z_t = torch.bmm(A_dy, V).mean(dim=1)  # (B, d)
                    alpha = torch.softmax(self.W_o(z_t), dim=-1)  # (B, 2)
                    alpha_static = alpha[:, 0].view(B, 1, 1)
                    alpha_dy = alpha[:, 1].view(B, 1, 1)
                    if _cd: _diag_alpha_static.append(alpha_static.mean().item())
                    if _cd: _diag_alpha_dy.append(alpha_dy.mean().item())
                    A_fused = alpha_static * gt * self.static_adj.unsqueeze(0) + alpha_dy * A_dy

            h_flat = h_t.reshape(B * N, self.hidden_dim)  # (B*N, d)
            _nz = torch.nonzero(A_fused)  # (num_nz, 3): (b, i, j)
            edge_index = torch.stack([_nz[:, 1] + _nz[:, 0] * N, _nz[:, 2] + _nz[:, 0] * N], dim=0)  # (2, num_nz)
            edge_weight = A_fused[_nz[:, 0], _nz[:, 1], _nz[:, 2]]  # (num_nz,)
            h_spatial = F.elu(self.gat1(h_flat, edge_index, edge_attr=edge_weight.unsqueeze(-1)))
            h_spatial = F.elu(self.gat2(h_spatial, edge_index, edge_attr=edge_weight.unsqueeze(-1)))
            h_spatial = h_spatial.reshape(B, N, self.hidden_dim)

            h_dy = torch.bmm(A_dy, self.W_agg(h_t))  # (B, N, d)

            h_out = self.norm(h_t + h_spatial + h_dy)  # (B, N, d)

            if self.node_agg == 'linear':
                step_representations.append(self.node_proj(h_out.reshape(B, N * self.hidden_dim)))  # (B, d)
            elif self.keep_node:
                step_representations.append(h_out.reshape(B, N * self.hidden_dim))
            else:
                step_representations.append(h_out.mean(dim=1))

        h_seq = torch.stack(step_representations, dim=1)  # (B, T, d)
        gru_out, _ = self.gru(h_seq)  # (B, T, d)
        if config['use_last_step']:
            final_emb = gru_out[:, -1]
            att_scores = torch.zeros(B, T, device=x_seq.device); att_scores[:, -1] = 1.0
        else:
            att_scores = F.softmax(self.time_attention(gru_out).squeeze(-1), dim=1)  # (B, T)
            final_emb = torch.bmm(att_scores.unsqueeze(1), gru_out).squeeze(1)  # (B, d)

        if static_ctx is not None and self.static_dim > 0:
            s_proj = self.static_proj(static_ctx)  # (B, d)
            gate = torch.sigmoid(self.static_gate(torch.cat([final_emb, static_ctx], dim=-1)))  # (B, d)
            final_emb = gate * s_proj + (1.0 - gate) * final_emb

        def _avg(vals):
            return sum(vals) / len(vals) if vals else 0.0

        self._diag = {
            'alpha_static': _avg(_diag_alpha_static),
            'alpha_dy': _avg(_diag_alpha_dy),
            'A_dy_max': _avg(_diag_ady_max),
            'A_dy_ent': _avg(_diag_ady_ent),
            'attn_max': _avg(_diag_attn_max),
            'attn_min': _avg(_diag_attn_min),
            'q_norm': _avg(_diag_q_norm),
            'k_norm': _avg(_diag_k_norm),
            'h_norm': _avg(_diag_h_norm),
        }
        return self.classifier(final_emb), gating_values, att_scores
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=0.1)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


num_classes=len(np.unique(y_tensor))


model = DGTI_Model(num_nodes, config['hidden_dim'], num_classes, static_adj_tensor, in_channels=config.get('n_channels', 2), static_dim=static_ctx.shape[1]).cuda()

_n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
_n_total = sum(p.numel() for p in model.parameters())
print(f'[model] trainable_params={_n_params:,}  total_params={_n_total:,}  hidden_dim={config["hidden_dim"]}  num_nodes={num_nodes}  gat_heads={config["gat_heads"]}  gru_layers={config["gru_layers"]}')

attention_prefixes = ('W_Q', 'W_K', 'W_V', 'W_o', 'time_attention', 'g0', 'reg_lambda')
attention_params = [p for n, p in model.named_parameters() if n.startswith(attention_prefixes)]
backbone_params = [p for n, p in model.named_parameters() if not n.startswith(attention_prefixes)]

optimizer = torch.optim.AdamW([
    {'params': attention_params, 'lr': config['attention_lr']},
    {'params': backbone_params, 'lr': config['lr']},
], weight_decay=config['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
criterion = nn.CrossEntropyLoss()


best_val_f1 = 0

swa_start = int(config['epochs'] * 2 / 3)
swa_state = None
swa_n = 0

torch.manual_seed(_SEED)
if not _FREE_GPU_SEED:
    torch.cuda.manual_seed_all(_SEED)

for epoch in range(config['epochs']):
    model.train()
    total_loss = 0
    _diag_epoch = (epoch % 5 == 0)
    _n_batches = len(train_loader)
    for bi, (bx, by, bs) in enumerate(train_loader):
        bx, by, bs = bx.cuda(), by.cuda(), bs.cuda()
        optimizer.zero_grad()
        model.collect_diag = _diag_epoch and (bi == _n_batches - 1)

        logits, g_vals, att_weights = model(bx, bs)

        loss = criterion(logits, by)
        if config.get('dy_entropy_reg', 0.0) > 0 and model.dy_entropy is not None:
            loss = loss + config['dy_entropy_reg'] * torch.relu(config['dy_entropy_target'] - model.dy_entropy)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()

    if epoch >= swa_start:
        swa_n += 1
        _cur = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if swa_state is None:
            swa_state = _cur
        else:
            for _k in swa_state:
                swa_state[_k].add_(_cur[_k] - swa_state[_k], alpha=1.0 / swa_n)

    if epoch % 5 == 0:
        att_detail = {n.split('.')[0]: (p.grad.norm().item() if p.grad is not None else 0.0)
                      for n, p in model.named_parameters() if n.startswith(attention_prefixes)}
        bkb_grad = sum(p.grad.norm().item() for p in backbone_params if p.grad is not None)
        print('[grad-att] ' + ' '.join(f'{k}={v:.2e}' for k, v in att_detail.items()))
        print(f'[grad-backbone] {bkb_grad:.2e}')
        if hasattr(model, '_diag'):
            d = model._diag
            print(f'[diag] alpha_static={d["alpha_static"]:.4f} alpha_dy={d["alpha_dy"]:.4f} A_dy_max={d["A_dy_max"]:.4f} A_dy_ent={d["A_dy_ent"]:.3f} '
                  f'attn=[{d["attn_min"]:.2f},{d["attn_max"]:.2f}] q_norm={d["q_norm"]:.2f} k_norm={d["k_norm"]:.2f} h_norm={d["h_norm"]:.2f}')
        model.eval()
        model.collect_diag = False
        v_preds, v_labels = [], []
        with torch.no_grad():
            for vx, vy, vs in val_loader:
                vx, vy, vs = vx.cuda(), vy.cuda(), vs.cuda()
                v_logits, v_g, v_att = model(vx, vs)
                v_preds.append(v_logits.argmax(dim=1).cpu())
                v_labels.append(vy.cpu())
        
        y_true = torch.cat(v_labels).numpy()
        y_pred = torch.cat(v_preds).numpy()
        val_f1 = f1_score(y_true, y_pred, average='macro')
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f'best_model_seed{_SEED}_split{_SPLIT}.pt')
        print(f'Epoch {epoch:03d} | Loss {total_loss/len(train_loader):.4f} | val F1 {val_f1:.4f} (best {best_val_f1:.4f})')

# Final test evaluation (test split of the 60:10:30 split; best-val checkpoint)
model.load_state_dict(torch.load(f'best_model_seed{_SEED}_split{_SPLIT}.pt'))
model.eval()
t_preds, t_labels, t_logits_all = [], [], []
with torch.no_grad():
    for tx, ty, ts in test_loader:
        tx, ty, ts = tx.cuda(), ty.cuda(), ts.cuda()
        t_logits, _, _ = model(tx, ts)
        t_preds.append(t_logits.argmax(dim=1).cpu())
        t_labels.append(ty.cpu())
        t_logits_all.append(t_logits.cpu())
y_true_test = torch.cat(t_labels).numpy()
y_pred_test = torch.cat(t_preds).numpy()
test_f1 = f1_score(y_true_test, y_pred_test, average='macro')
print(f'Final test macro-F1: {test_f1:.4f}')
np.save(f'test_logits_best_seed{_SEED}_split{_SPLIT}.npy', torch.cat(t_logits_all).numpy())

if swa_state is not None:
    model.load_state_dict(swa_state)
    model.eval()
    sp, sl = [], []
    with torch.no_grad():
        for tx, ty, ts in test_loader:
            tx, ty, ts = tx.cuda(), ty.cuda(), ts.cuda()
            t_logits, _, _ = model(tx, ts)
            sp.append(t_logits.cpu())
            sl.append(ty.cpu())
    swa_logits = torch.cat(sp).numpy()
    swa_labels = torch.cat(sl).numpy()
    swa_f1 = f1_score(swa_labels, swa_logits.argmax(1), average='macro')
    print(f'SWA test macro-F1: {swa_f1:.4f}')
    np.save(f'test_logits_swa_seed{_SEED}_split{_SPLIT}.npy', swa_logits)
    np.save(f'test_labels_seed{_SEED}_split{_SPLIT}.npy', swa_labels)
    torch.save(swa_state, f'swa_model_seed{_SEED}_split{_SPLIT}.pt')
        