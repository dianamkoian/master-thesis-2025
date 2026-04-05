import streamlit as st

st.set_page_config(page_title="Детектор контрафакта — Ozon", page_icon="🛡️")

st.title("Система детекции контрафактных товаров")
st.subheader("Кросс-доменная мультимодальная система на основе методов из FinTech, Social Networks, Real Estate и Mobile Apps")

st.divider()

st.markdown("""
### О проекте
Мы адаптируем методы борьбы с мошенничеством из смежных доменов для детекции контрафакта на маркетплейсе Ozon.

### Домены
- 💳 **FinTech** — AML-скоринг, behavioral analytics, anomaly detection
- 🌐 **Social Networks** — bot detection, graph analysis  
- 🏠 **Real Estate** — document verification, CV-анализ
- 📱 **Mobile Apps** — мультимодальный анализ контента

### Baseline метрики
| Модальность | ROC-AUC | PR-AUC | Recall@P≥0.9 |
|---|---|---|---|
| Текст (TF-IDF + LR) | 0.905 | 0.590 | ~0.0 |
| Табличные (CatBoost) | 0.905 | 0.623 | 0.015 |
| Изображения (LR) | 0.853 | 0.462 | 0.004 |

### Команда
Магистратура ВШЭ, 2025
""")

st.info("Полная демо-версия сервиса скоро будет доступна")