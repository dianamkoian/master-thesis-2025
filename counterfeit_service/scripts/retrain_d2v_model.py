"""
Повторное обучение Doc2Vec-модели (200-dim) на train-текстах counterfeit-датасета
и сохранение через joblib в актуальном numpy-формате. Это устраняет ValueError
при `joblib.load(d2v_model.pkl)` на современной numpy (legacy < 1.17 формат).

Контракт совпадает с app/predictor.py:
  - токенизация: text.lower().split()
  - text = name_rus + ' ' + description + ' ' + brand_name (fillna '')
  - vector_size = 200, infer_vector(tokens, epochs=50)
Параметры обучения совместимы со стандартом для коротких карточек товара.
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

HERE = Path(__file__).resolve().parent.parent  # counterfeit_service/
PROJECT_ROOT = HERE.parent.parent  # master-thesis-2026/ (либо ..../master-thesis-2025/ при legacy запуске)
SERVICE_ARTIFACTS = HERE / 'artifacts'
LOG = HERE / 'scripts' / 'retrain_d2v_log.txt'

# DATA_CSV — внешний датасет. Ищем сначала в новой структуре, затем в legacy.
_DATA_CSV_CANDIDATES = [
    PROJECT_ROOT / 'data' / "Diana's folder" / 'ml_ozon_ounterfeit_train.csv',
    Path("/Users/diana/master-thesis-2025/Diana's folder/ml_ozon_ounterfeit_train.csv"),
]
DATA_CSV = next((p for p in _DATA_CSV_CANDIDATES if p.exists()), _DATA_CSV_CANDIDATES[0])

SEED = 42
TARGET = 'resolution'
VECTOR_SIZE = 200
WINDOW = 5
MIN_COUNT = 2
WORKERS = 4
EPOCHS = 20

def log(msg):
    s = f'{time.strftime("%H:%M:%S")} {msg}'
    print(s, flush=True)
    with open(LOG, 'a') as f:
        f.write(s + '\n')

t0 = time.time()
log('=' * 60)
log('Retrain Doc2Vec 200-dim для counterfeit_service')

# Загрузка данных
df = pd.read_csv(DATA_CSV, encoding='utf-8')
df['text'] = (df['name_rus'].fillna('') + ' ' +
              df['description'].fillna('') + ' ' +
              df['brand_name'].fillna(''))
log(f'data: {df.shape}')

# Тот же split, что в ноуте 02 и m2_fe_plus_individual_v2 — обучаем Doc2Vec на TRAIN-текстах
seller_targets = df.groupby('SellerID')[TARGET].max().reset_index()
train_sellers, _ = train_test_split(
    seller_targets['SellerID'], test_size=0.30,
    random_state=SEED, stratify=seller_targets[TARGET]
)
train_mask = df['SellerID'].isin(set(train_sellers))
train_texts = df.loc[train_mask, 'text'].values
log(f'train texts: {len(train_texts)}')

# Та же токенизация, что в predictor.py: text.lower().split()
log('Токенизация (lower().split())...')
docs = [TaggedDocument(words=t.lower().split(), tags=[i]) for i, t in enumerate(train_texts)]
log(f'documents: {len(docs)}')

# Обучение Doc2Vec
log(f'Обучение Doc2Vec (vector_size={VECTOR_SIZE}, window={WINDOW}, min_count={MIN_COUNT}, '
    f'workers={WORKERS}, epochs={EPOCHS})...')
t_t = time.time()
model = Doc2Vec(
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    workers=WORKERS,
    epochs=EPOCHS,
    dm=1,
    seed=SEED,
)
model.build_vocab(docs)
log(f'  vocabulary built: {len(model.wv)} токенов')
model.train(docs, total_examples=model.corpus_count, epochs=EPOCHS)
log(f'  обучен за {(time.time()-t_t)/60:.1f} мин')

# Контрольная проверка: вектор для произвольного текста и его размер
sample_text = "Кроссовки Nike Air Max оригинал размер 42 черные"
sample_vec = model.infer_vector(sample_text.lower().split(), epochs=50)
log(f'sample infer_vector shape: {sample_vec.shape}, mean={sample_vec.mean():.4f}, std={sample_vec.std():.4f}')
assert sample_vec.shape == (VECTOR_SIZE,), 'размерность не 200!'

# Сохранение через joblib (как в predictor.py: joblib.load(d2v_path))
SERVICE_ARTIFACTS.mkdir(parents=True, exist_ok=True)
out_path = SERVICE_ARTIFACTS / 'd2v_model.pkl'

# Резервная копия старого файла (если есть)
old_path = SERVICE_ARTIFACTS / 'd2v_model.pkl.legacy_backup'
if out_path.exists() and not old_path.exists():
    import shutil
    shutil.move(str(out_path), str(old_path))
    log(f'старая модель → {old_path.name}')

joblib.dump(model, out_path)
size_mb = out_path.stat().st_size / 1024**2
log(f'Saved: {out_path} ({size_mb:.1f} MB)')

# Контрольная загрузка через joblib (sanity check)
loaded = joblib.load(out_path)
test_vec = loaded.infer_vector(sample_text.lower().split(), epochs=50)
log(f'control reload: shape={test_vec.shape}, совпадает с обученной = {np.allclose(sample_vec, test_vec, atol=1e-3)}')

log(f'Всего: {(time.time()-t0)/60:.1f} мин')
log('=' * 60)
