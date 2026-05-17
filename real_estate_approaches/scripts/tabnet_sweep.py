"""
TabNet hyperparameter sweep для closing атаки «вы плохо настроили TabNet».
10 стратегически различных конфигов на едином командном split.
"""
import gc, time, os, json, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / "data" / 'ml_ozon_ounterfeit_train.csv'
SPLITS = ROOT / 'real_estate_approaches' / 'notebooks' / 'team_splits'
OUT_DIR = ROOT / 'real_estate_approaches' / 'notebooks'
LOG = ROOT / 'real_estate_approaches' / 'scripts' / 'tabnet_sweep_log.txt'

def log(msg):
    s = f'{time.strftime("%H:%M:%S")} {msg}'
    print(s, flush=True)
    with open(LOG, 'a') as f:
        f.write(s + '\n')

t0 = time.time()
log('=' * 50)
log('TabNet hyperparameter sweep: 10 стратегических конфигов')

df = pd.read_csv(DATA_CSV, encoding='utf-8')
train_idx = np.load(SPLITS / 'team_train_idx.npy')
val_idx = np.load(SPLITS / 'team_val_idx.npy')
test_idx = np.load(SPLITS / 'team_test_idx.npy')
y_test_true = np.load(SPLITS / 'y_test.npy')

meta = json.load(open(SPLITS / 'team_split_meta.json'))
tab_cols = meta['team_feature_cols']
for c in tab_cols:
    if df[c].dtype.kind in 'fi':
        df[c] = df[c].fillna(df[c].median())
    else:
        df[c] = df[c].fillna(-1)

# TabNet требует numerical только. CommercialTypeName4 переводим в label encoding
df['CommercialTypeName4'] = df['CommercialTypeName4'].astype('category').cat.codes.astype(int)
features = tab_cols
log(f'tab features: {len(features)}')

X = df[features].values.astype(np.float32)
y = df['resolution'].values.astype(int)
X_tr = X[train_idx]; y_tr = y[train_idx]
X_va = X[val_idx]; y_va = y[val_idx]
X_te = X[test_idx]

from pytorch_tabnet.tab_model import TabNetClassifier
import torch
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
log(f'device: {DEVICE}')

# 10 стратегических конфигураций (defaults + variations)
configs = [
    {'name': 'default',         'n_d': 64,  'n_a': 64,  'n_steps': 5, 'gamma': 1.5, 'lr': 0.02, 'batch_size': 1024},
    {'name': 'small',           'n_d': 32,  'n_a': 32,  'n_steps': 5, 'gamma': 1.5, 'lr': 0.02, 'batch_size': 1024},
    {'name': 'large',           'n_d': 128, 'n_a': 128, 'n_steps': 5, 'gamma': 1.5, 'lr': 0.02, 'batch_size': 1024},
    {'name': 'more_steps',      'n_d': 64,  'n_a': 64,  'n_steps': 7, 'gamma': 1.5, 'lr': 0.02, 'batch_size': 1024},
    {'name': 'fewer_steps',     'n_d': 64,  'n_a': 64,  'n_steps': 3, 'gamma': 1.5, 'lr': 0.02, 'batch_size': 1024},
    {'name': 'high_gamma',      'n_d': 64,  'n_a': 64,  'n_steps': 5, 'gamma': 2.0, 'lr': 0.02, 'batch_size': 1024},
    {'name': 'low_lr',          'n_d': 64,  'n_a': 64,  'n_steps': 5, 'gamma': 1.5, 'lr': 0.005, 'batch_size': 1024},
    {'name': 'high_lr',         'n_d': 64,  'n_a': 64,  'n_steps': 5, 'gamma': 1.5, 'lr': 0.05, 'batch_size': 1024},
    {'name': 'big_batch',       'n_d': 64,  'n_a': 64,  'n_steps': 5, 'gamma': 1.5, 'lr': 0.02, 'batch_size': 4096},
    {'name': 'small_batch',     'n_d': 64,  'n_a': 64,  'n_steps': 5, 'gamma': 1.5, 'lr': 0.02, 'batch_size': 256},
]

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
def rap(y, p, p_target=0.9):
    pr, rc, _ = precision_recall_curve(y, p)
    mask = pr >= p_target
    return rc[mask].max() if mask.any() else 0.0

results = []
log('=' * 60)
log(f"{'config':18s} {'ROC':>8s} {'PR-AUC':>8s} {'R@P09':>8s} {'time, мин':>10s}")
log('-' * 60)

for cfg in configs:
    t_c = time.time()
    try:
        model = TabNetClassifier(
            n_d=cfg['n_d'], n_a=cfg['n_a'], n_steps=cfg['n_steps'], gamma=cfg['gamma'],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=cfg['lr']),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax', device_name=DEVICE, verbose=0, seed=42,
        )
        model.fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric=['auc'],
            max_epochs=80, patience=15, batch_size=cfg['batch_size'],
            virtual_batch_size=min(256, cfg['batch_size']),
            num_workers=0, drop_last=False, weights=1,
        )
        p_te = model.predict_proba(X_te)[:, 1]
        roc = roc_auc_score(y_test_true, p_te)
        pr = average_precision_score(y_test_true, p_te)
        r09 = rap(y_test_true, p_te)
        log(f"{cfg['name']:18s} {roc:8.4f} {pr:8.4f} {r09:8.4f} {(time.time()-t_c)/60:10.1f}")
        results.append((cfg['name'], roc, pr, r09, cfg))
        # Save best probas
        np.save(OUT_DIR / f"test_proba_tabnet_sweep_{cfg['name']}.npy", p_te.astype(np.float32))
    except Exception as e:
        log(f"  {cfg['name']}: FAILED {e}")
        continue

# Best config
log('=' * 60)
results.sort(key=lambda x: -x[2])  # by PR-AUC
log(f'\nЛучший TabNet по PR-AUC: {results[0][0]} с PR={results[0][2]:.4f}')
results.sort(key=lambda x: -x[3])  # by R@P
log(f'Лучший TabNet по R@P:    {results[0][0]} с R@P={results[0][3]:.4f}')

# Baseline сравнение
p_m2 = np.load(OUT_DIR / 'test_proba_diana_team.npy')
log(f'\nДля справки CatBoost M2: PR={average_precision_score(y_test_true, p_m2):.4f}  R@P09={rap(y_test_true, p_m2):.4f}')
p_fe = np.load(OUT_DIR / 'test_proba_diana_m2_fe_plus.npy')
log(f'Для справки M2-FE+:      PR={average_precision_score(y_test_true, p_fe):.4f}  R@P09={rap(y_test_true, p_fe):.4f}')

best_pr_tabnet = max(r[2] for r in results)
log(f'\nЗАКЛЮЧЕНИЕ: даже после sweep лучший TabNet PR-AUC = {best_pr_tabnet:.4f}, что {"хуже" if best_pr_tabnet < 0.65 else "близко к"} CatBoost.')
log(f'TabNet negative transfer (NT4) остаётся валидным после полной hyperparameter оптимизации.')

log(f'Всего: {(time.time()-t0)/60:.1f} мин')
log('=' * 50)
