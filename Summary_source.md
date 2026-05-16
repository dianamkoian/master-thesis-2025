# Список источников

Библиография по теме мультимодальной детекции контрафакта и фейковых объявлений на маркетплейсах: методы машинного обучения, vision-language модели, борьба с дисбалансом классов, MLOps и инженерная инфраструктура.

## Содержание

- [Мультимодальная детекция фейков и контрафакта](#мультимодальная-детекция-фейков-и-контрафакта)
- [Vision-Language модели (VLM/LMM)](#vision-language-модели-vlmlmm)
- [Архитектуры компьютерного зрения](#архитектуры-компьютерного-зрения)
- [NLP, эмбеддинги и трансформеры](#nlp-эмбеддинги-и-трансформеры)
- [Градиентный бустинг и табличные модели](#градиентный-бустинг-и-табличные-модели)
- [Дисбаланс классов и метрики](#дисбаланс-классов-и-метрики)
- [Интерпретируемость и калибровка](#интерпретируемость-и-калибровка)
- [Transfer learning и доменная адаптация](#transfer-learning-и-доменная-адаптация)
- [Снижение размерности и классические методы](#снижение-размерности-и-классические-методы)
- [MLOps и инженерия ML-систем](#mlops-и-инженерия-ml-систем)
- [Программная архитектура и инструменты](#программная-архитектура-и-инструменты)
- [Отраслевые отчёты и рыночные данные](#отраслевые-отчёты-и-рыночные-данные)
- [Нормативные документы](#нормативные-документы)

---

## Мультимодальная детекция фейков и контрафакта

- Singhal S., Shah R. R., Chakraborty T., Kumaraguru P., Satoh S. **SpotFake: A multi-modal framework for fake news detection** // 2019 IEEE Fifth International Conference on Multimedia Big Data (BigMM). — Singapore: IEEE, 2019. — P. 39–47. DOI: [10.1109/BigMM.2019.00-44](https://doi.org/10.1109/BigMM.2019.00-44).
- Zhou X., Wu J., Zafarani R. **SAFE: Similarity-Aware Multi-modal Fake News Detection** // PAKDD 2020. LNCS, vol. 12085. — Cham: Springer, 2020. — P. 354–367. DOI: [10.1007/978-3-030-47436-2_27](https://doi.org/10.1007/978-3-030-47436-2_27). arXiv:[2003.04981](https://arxiv.org/abs/2003.04981).
- Chen Y., Li D., Zhang P., Sui J., Lv Q., Lu T., Shang L. **Cross-modal Ambiguity Learning for Multimodal Fake News Detection (CAFE)** // ACM Web Conference 2022 (WWW). — ACM, 2022. — P. 2897–2905. DOI: [10.1145/3485447.3511968](https://doi.org/10.1145/3485447.3511968).
- Pramanick S., Sharma S., Dimitrov D., Akhtar M. S., Nakov P., Chakraborty T. **MOMENTA: A Multimodal Framework for Detecting Harmful Memes and Their Targets** // Findings of ACL: EMNLP 2021. — ACL, 2021. — P. 4439–4455. DOI: [10.18653/v1/2021.findings-emnlp.379](https://doi.org/10.18653/v1/2021.findings-emnlp.379). arXiv:[2109.05184](https://arxiv.org/abs/2109.05184).
- Wu J., Fu Y. **MMD-Thinker: Adaptive Multi-Dimensional Thinking for Multimodal Misinformation Detection** // arXiv preprint, 2025. arXiv:[2511.13242](https://arxiv.org/abs/2511.13242).
- Karunanayake N., Rajasegaran J., Gunathillake A., Seneviratne S., Jourjon G. **A Multi-modal Neural Embeddings Approach for Detecting Mobile Counterfeit Apps: A Case Study on Google Play Store** // IEEE Transactions on Mobile Computing. — 2022. — Vol. 21, № 1. — P. 16–30. DOI: [10.1109/TMC.2020.3007260](https://doi.org/10.1109/TMC.2020.3007260). arXiv:[2006.02231](https://arxiv.org/abs/2006.02231).
- Rajasegaran J., Karunanayake N., Gunathillake A., Seneviratne S., Jourjon G. **A Neural Embeddings Approach for Detecting Mobile Counterfeit Apps** // WWW 2019. — ACM, 2019. — P. 3165–3171. DOI: [10.1145/3308558.3313427](https://doi.org/10.1145/3308558.3313427). arXiv:[1804.09882](https://arxiv.org/abs/1804.09882).
- Deng X., Zhang M., Dong X., Hu X. **Detect Counterfeit Mini-apps: A Case Study on WeChat** // ACM Workshop on Secure and Trustworthy Superapps (SaTS '24). — ACM, 2024. DOI: [10.1145/3689941.3695773](https://doi.org/10.1145/3689941.3695773).
- Mohsen F., Karastoyanova D., Azzopardi G. **Early Detection of Violating Mobile Apps: A Data-Driven Predictive Model Approach** // Systems and Soft Computing. — Elsevier, 2022. — Vol. 4, Article 200045. DOI: [10.1016/j.sasc.2022.200045](https://doi.org/10.1016/j.sasc.2022.200045).
- Nguyen D., Nguyen T. T., Nguyen C. V. **FADAML: Fake Advertisements Detection Using Automated Multimodal Learning** // Applied Intelligence. — Springer, 2025. DOI: [10.1007/s10489-025-06238-2](https://doi.org/10.1007/s10489-025-06238-2). arXiv:[2501.10848](https://arxiv.org/abs/2501.10848).
- Mohd Amin N. A., Sulaiman J., Mohd Tahir N. **Multimodal classification of fake property listings using cluster-based preprocessing and Dempster–Shafer fusion** // PeerJ Computer Science. — 2024. — Vol. 10, e2197. DOI: [10.7717/peerj-cs.2197](https://doi.org/10.7717/peerj-cs.2197).
- Mohd Amin M., Sani N. S., Nasrudin M. F., Abdullah S. **Clustering Analysis for Classifying Fake Real Estate Listings** // PeerJ Computer Science. — 2024. — Vol. 10, e2019. DOI: [10.7717/peerj-cs.2019](https://doi.org/10.7717/peerj-cs.2019).
- Mohd Amin M., Sani N. S., Nasrudin M. F. **Class-weighted Dempster–Shafer in dual-level fusion for multimodal fake real estate listings detection** // PeerJ Computer Science. — 2025. — Vol. 11, e2797. DOI: [10.7717/peerj-cs.2797](https://doi.org/10.7717/peerj-cs.2797).
- Sulistio B., Suparta W., Trisetyarso A., Abbas B. S., Kang C. H. **Multimodal CatBoost framework for online real estate fraud detection** // IEEE IAICT 2025. — IEEE, 2025. — P. 112–117.
- Sulistio F. A., Suakanto S., Ambarsari N., Rijadi S. C. R. **Detecting Fake Boarding House Listings Using Multimodal Deep Learning** // IEEE IAICT 2025. — IEEE, 2025. DOI: [10.1109/IAICT65714.2025.11101391](https://doi.org/10.1109/IAICT65714.2025.11101391).
- Shehu A.-S., Pinto A., Correia M. E. **A Decentralised Real Estate Transfer Verification Based on Self-Sovereign Identity and Smart Contracts** // arXiv preprint, 2022. arXiv:[2207.04459](https://arxiv.org/abs/2207.04459).
- Mutemi A., Bacao F. **A numeric-based machine learning design for detecting organized retail fraud in digital marketplaces** // Scientific Reports. — 2023. — Vol. 13, Article 12499. DOI: [10.1038/s41598-023-38304-5](https://doi.org/10.1038/s41598-023-38304-5).
- Garcia-Cotte H., Mellouli D., Rehman A., Wang L., Stork D. G. **Deep neural network-based detection of counterfeit products from smartphone images** // arXiv preprint, 2024. arXiv:[2410.05969](https://arxiv.org/abs/2410.05969).
- Peng J., Zou B., Zhu C. **A two-stage deep learning framework for counterfeit luxury handbag detection in logo images** // Signal, Image and Video Processing. — 2023. — Vol. 17, № 4. — P. 1439–1448. DOI: [10.1007/s11760-022-02352-7](https://doi.org/10.1007/s11760-022-02352-7).
- Deng R., Chen J., Niu Y., Yang M. **Understanding promotion-as-a-service on GitHub** // 29th USENIX Security Symposium. — 2020. — P. 2065–2082.
- Wei W., Li J., Cao L., Ou Y., Chen J. **Effective detection of sophisticated online banking fraud on extremely imbalanced data** // World Wide Web. — 2013. — Vol. 16, № 4. — P. 449–475. DOI: [10.1007/s11280-012-0178-0](https://doi.org/10.1007/s11280-012-0178-0).
- Boulieris P., Pavlopoulos J., Xenos A., Vassalos V. **Fraud detection with natural language processing** // Machine Learning. — 2024. — Vol. 113. — P. 5087–5108. DOI: [10.1007/s10994-023-06354-5](https://doi.org/10.1007/s10994-023-06354-5).
- Qiao Y. et al. **Fraud detection and risk assessment of online payment transactions on e-commerce platforms based on LLM and GCN frameworks** // arXiv preprint, 2025. arXiv:[2509.09928](https://arxiv.org/abs/2509.09928).
- Weber M. et al. **Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics** // arXiv preprint, 2019. arXiv:[1908.02591](https://arxiv.org/abs/1908.02591).
- Correa Bahnsen A., Aouada D., Stojanovic A., Ottersten B. **Feature engineering strategies for credit card fraud detection** // Expert Systems with Applications. — 2016. — Vol. 51. — P. 134–142. DOI: [10.1016/j.eswa.2015.12.030](https://doi.org/10.1016/j.eswa.2015.12.030).
- Dal Pozzolo A., Caelen O., Le Borgne Y.-A., Waterschoot S., Bontempi G. **Learned lessons in credit card fraud detection from a practitioner perspective** // Expert Systems with Applications. — 2014. — Vol. 41, № 10. — P. 4915–4928.
- Tax N., de Vries K. J., de Jong M. et al. **Machine Learning for Fraud Detection in E-Commerce: A Research Agenda** // arXiv preprint, 2021 (MLHat 2021). arXiv:[2107.01979](https://arxiv.org/abs/2107.01979).
- Baltrušaitis T., Ahuja C., Morency L.-P. **Multimodal Machine Learning: A Survey and Taxonomy** // IEEE TPAMI. — 2019. — Vol. 41, № 2. — P. 423–443. DOI: [10.1109/TPAMI.2018.2798607](https://doi.org/10.1109/TPAMI.2018.2798607).

## Vision-Language модели (VLM/LMM)

- Radford A., Kim J. W., Hallacy C. et al. **Learning Transferable Visual Models From Natural Language Supervision (CLIP)** // ICML 2021, PMLR. — Vol. 139. — P. 8748–8763. arXiv:[2103.00020](https://arxiv.org/abs/2103.00020).
- Li J., Li D., Savarese S., Hoi S. **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models** // ICML 2023, PMLR. — Vol. 202. — P. 19730–19742. arXiv:[2301.12597](https://arxiv.org/abs/2301.12597).
- Liu H., Li C., Wu Q., Lee Y. J. **Visual Instruction Tuning (LLaVA)** // NeurIPS 2023. arXiv:[2304.08485](https://arxiv.org/abs/2304.08485).
- Li B., Zhang Y., Guo D. et al. **LLaVA-OneVision: Easy Visual Task Transfer** // arXiv preprint, 2024. arXiv:[2408.03326](https://arxiv.org/abs/2408.03326).
- Bai J., Bai S., Yang S. et al. **Qwen-VL: A Versatile Vision-Language Model** // arXiv preprint, 2023. arXiv:[2308.12966](https://arxiv.org/abs/2308.12966).
- Wang P., Bai S., Tan S. et al. **Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution** // arXiv preprint, 2024. arXiv:[2409.12191](https://arxiv.org/abs/2409.12191).
- Qwen Team. **Qwen2.5-VL Technical Report** // arXiv preprint, 2025. arXiv:[2502.13923](https://arxiv.org/abs/2502.13923).
- Qwen Team (Alibaba). **Qwen2.5 Technical Report** // arXiv preprint, 2024. arXiv:[2412.15115](https://arxiv.org/abs/2412.15115).
- Xu J., Lo S.-Y., Safaei B., Patel V. M., Dwivedi I. **Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models** // CVPR 2025.
- Keita M., Hamidouche W., Bougueffa Eutamene H., Hadid A., Taleb-Ahmed A. **FIDAVL: Fake Image Detection and Attribution using Vision-Language Model** // arXiv preprint, 2024. arXiv:[2409.03109](https://arxiv.org/abs/2409.03109).
- Lee J., Lim Y., Lee S. **Multimodal Large Language Models for Phishing Webpage Detection and Identification** // arXiv preprint, 2024. arXiv:[2408.05941](https://arxiv.org/abs/2408.05941).
- Hu Y., Wang L., Liu B. et al. **Unlocking the Capabilities of Large Vision-Language Models for Generalizable and Explainable Deepfake Detection** // arXiv preprint, 2025. arXiv:[2503.14853](https://arxiv.org/abs/2503.14853).
- Hu Z., Yin Y., Wu Y. **Visual Language Models as Zero-Shot Deepfake Detectors** // arXiv preprint, 2025. arXiv:[2507.22469](https://arxiv.org/abs/2507.22469).
- Vajda P. et al. **Vision-Language Models for E-commerce: Detecting Non-Compliant Product Images in Online Catalogs** // ICAART 2025. — SCITEPRESS, 2025. URL: <https://www.scitepress.org/Papers/2025/132650/132650.pdf>.
- Vajda P. **Transforming Product Discovery and Interpretation Using Vision-Language Models** // J. of Theoretical and Applied Electronic Commerce Research. — MDPI, 2025. — Vol. 20, № 3, Article 191.
- Asai R., Saito Y., Yamada S. **Improving Visual Recommendation on E-commerce Platforms Using Vision-Language Models** // RecSys '25. — 2025. arXiv:[2510.13359](https://arxiv.org/abs/2510.13359).
- eBay Engineering. **Scaling Large Language Models for e-Commerce: The Development of a Llama-Based Customized LLM** // eBay Tech Blog. — 2024. URL: <https://innovation.ebayinc.com/stories/scaling-large-language-models-for-e-commerce-the-development-of-a-llama-based-customized-llm-for-e-commerce/>.

## Архитектуры компьютерного зрения

- Simonyan K., Zisserman A. **Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)** // ICLR 2015. arXiv:[1409.1556](https://arxiv.org/abs/1409.1556).
- He K., Zhang X., Ren S., Sun J. **Deep Residual Learning for Image Recognition (ResNet)** // CVPR 2016. — IEEE, 2016. — P. 770–778. DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90).
- Tan M., Le Q. V. **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks** // ICML 2019, PMLR. — Vol. 97. — P. 6105–6114. arXiv:[1905.11946](https://arxiv.org/abs/1905.11946).
- Dosovitskiy A., Beyer L., Kolesnikov A. et al. **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale (ViT)** // ICLR 2021. arXiv:[2010.11929](https://arxiv.org/abs/2010.11929).

## NLP, эмбеддинги и трансформеры

- Vaswani A., Shazeer N., Parmar N. et al. **Attention Is All You Need** // NeurIPS 2017. — Vol. 30. — P. 5998–6008. arXiv:[1706.03762](https://arxiv.org/abs/1706.03762).
- Devlin J., Chang M.-W., Lee K., Toutanova K. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** // NAACL-HLT 2019. — ACL, 2019. — P. 4171–4186. DOI: [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423). arXiv:[1810.04805](https://arxiv.org/abs/1810.04805).
- Kuratov Y., Arkhipov M. **Adaptation of Deep Bidirectional Multilingual Transformers for Russian Language (RuBERT)** // Dialogue 2019. arXiv:[1905.07213](https://arxiv.org/abs/1905.07213).
- Reimers N., Gurevych I. **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** // EMNLP-IJCNLP 2019. — P. 3982–3992.
- Le Q. V., Mikolov T. **Distributed Representations of Sentences and Documents** // ICML 2014, PMLR. — Vol. 32. — P. 1188–1196. arXiv:[1405.4053](https://arxiv.org/abs/1405.4053).
- Wang L., Yang N., Huang X. et al. **Multilingual E5 Text Embeddings: A Technical Report** // arXiv preprint, 2024. arXiv:[2402.05672](https://arxiv.org/abs/2402.05672).
- Wang L. et al. **Text Embeddings by Weakly-Supervised Contrastive Pre-training** // arXiv preprint, 2022. arXiv:[2212.03533](https://arxiv.org/abs/2212.03533).
- Sparck Jones K. **A statistical interpretation of term specificity and its application in retrieval** // Journal of Documentation. — 1972. — Vol. 28, № 1. — P. 11–21. DOI: [10.1108/eb026526](https://doi.org/10.1108/eb026526).
- Robertson S. **Understanding Inverse Document Frequency: On Theoretical Arguments for IDF** // Journal of Documentation. — 2004. — Vol. 60, № 5. — P. 503–520. DOI: [10.1108/00220410410560582](https://doi.org/10.1108/00220410410560582).
- Levenshtein V. I. **Binary codes capable of correcting deletions, insertions, and reversals** // Soviet Physics Doklady. — 1966. — Vol. 10, № 8. — P. 707–710.

## Градиентный бустинг и табличные модели

- Prokhorenkova L., Gusev G., Vorobev A., Dorogush A. V., Gulin A. **CatBoost: Unbiased Boosting with Categorical Features** // NeurIPS 2018. — Vol. 31. — P. 6638–6648. arXiv:[1706.09516](https://arxiv.org/abs/1706.09516).
- Chen T., Guestrin C. **XGBoost: A Scalable Tree Boosting System** // ACM SIGKDD 2016. — P. 785–794. DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).
- Ke G., Meng Q., Finley T. et al. **LightGBM: A Highly Efficient Gradient Boosting Decision Tree** // NeurIPS 2017. — Vol. 30. — P. 3146–3154.
- Friedman J. H. **Greedy Function Approximation: A Gradient Boosting Machine** // Annals of Statistics. — 2001. — Vol. 29, № 5. — P. 1189–1232.
- Breiman L. **Random Forests** // Machine Learning. — 2001. — Vol. 45, № 1. — P. 5–32.
- Wolpert D. H. **Stacked Generalization** // Neural Networks. — 1992. — Vol. 5, № 2. — P. 241–259. DOI: [10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1).
- Vapnik V. **The Nature of Statistical Learning Theory**. — Springer-Verlag, 1995. — 188 p.
- Grinsztajn L., Oyallon E., Varoquaux G. **Why do tree-based models still outperform deep learning on typical tabular data?** // NeurIPS 2022, Datasets and Benchmarks Track. — Vol. 35. — P. 507–520. arXiv:[2207.08815](https://arxiv.org/abs/2207.08815).
- Hollmann N., Müller S., Eggensperger K., Hutter F. **TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second** // Nature, 2025; ICLR 2023. arXiv:[2207.01848](https://arxiv.org/abs/2207.01848).
- Arik S. Ö., Pfister T. **TabNet: Attentive Interpretable Tabular Learning** // AAAI 2021. — Vol. 35, № 8. — P. 6679–6687.

## Дисбаланс классов и метрики

- Chawla N. V., Bowyer K. W., Hall L. O., Kegelmeyer W. P. **SMOTE: Synthetic Minority Over-sampling Technique** // JAIR. — 2002. — Vol. 16. — P. 321–357. DOI: [10.1613/jair.953](https://doi.org/10.1613/jair.953).
- He H., Garcia E. A. **Learning from Imbalanced Data** // IEEE TKDE. — 2009. — Vol. 21, № 9. — P. 1263–1284. DOI: [10.1109/TKDE.2008.239](https://doi.org/10.1109/TKDE.2008.239).
- Dal Pozzolo A., Caelen O., Johnson R. A., Bontempi G. **Calibrating Probability with Undersampling for Unbalanced Classification** // IEEE SSCI 2015. — P. 159–166. DOI: [10.1109/SSCI.2015.33](https://doi.org/10.1109/SSCI.2015.33).
- Lin T.-Y., Goyal P., Girshick R., He K., Dollár P. **Focal Loss for Dense Object Detection** // ICCV 2017. — IEEE, 2017. — P. 2980–2988. DOI: [10.1109/ICCV.2017.324](https://doi.org/10.1109/ICCV.2017.324). arXiv:[1708.02002](https://arxiv.org/abs/1708.02002).
- Liu F. T., Ting K. M., Zhou Z.-H. **Isolation Forest** // ICDM 2008. — IEEE, 2008. — P. 413–422. DOI: [10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17).
- Sheng V. S., Ling C. X. **Thresholding for Making Classifiers Cost-Sensitive** // AAAI 2006. — P. 476–481.
- Davis J., Goadrich M. **The Relationship Between Precision-Recall and ROC Curves** // ICML 2006. — ACM, 2006. — P. 233–240. DOI: [10.1145/1143844.1143874](https://doi.org/10.1145/1143844.1143874).
- Saito T., Rehmsmeier M. **The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets** // PLoS ONE. — 2015. — Vol. 10, № 3, Article e0118432. DOI: [10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432).
- Boyd K., Eng K. H., Page C. D. **Area under the Precision-Recall Curve: Point Estimates and Confidence Intervals** // ECML PKDD 2013, LNCS, vol. 8190. — Springer, 2013. — P. 451–466. DOI: [10.1007/978-3-642-40994-3_29](https://doi.org/10.1007/978-3-642-40994-3_29).
- Shang H., Langlois J.-M., Tsioutsiouliklis K., Kang C. **Precision/Recall on Imbalanced Test Data** // AISTATS 2023, PMLR. — Vol. 206. — P. 9879–9891.
- MDPI Technologies. **Why ROC-AUC Is Misleading for Highly Imbalanced Data: In-Depth Evaluation of MCC, F2-Score, H-Measure, and AUC-Based Metrics** // MDPI Technologies. — 2025. — Vol. 14, № 1, Article 54.
- Mann H. B., Whitney D. R. **On a Test of Whether One of Two Random Variables is Stochastically Larger than the Other** // Annals of Mathematical Statistics. — 1947. — Vol. 18, № 1. — P. 50–60. DOI: [10.1214/aoms/1177730491](https://doi.org/10.1214/aoms/1177730491).

## Интерпретируемость и калибровка

- Lundberg S. M., Lee S.-I. **A Unified Approach to Interpreting Model Predictions (SHAP)** // NeurIPS 2017. — Vol. 30. — P. 4765–4774. arXiv:[1705.07874](https://arxiv.org/abs/1705.07874).
- Platt J. C. **Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods** // Advances in Large Margin Classifiers. — MIT Press, 1999. — P. 61–74.
- Niculescu-Mizil A., Caruana R. **Predicting good probabilities with supervised learning** // ICML 2005. — P. 625–632.
- Shafer G., Vovk V. **A Tutorial on Conformal Prediction** // JMLR. — 2008. — Vol. 9. — P. 371–421.
- Romano Y., Sesia M., Candès E. **Classification with Valid and Adaptive Coverage** // NeurIPS 2020. — Vol. 33.

## Transfer learning и доменная адаптация

- Pan S. J., Yang Q. **A Survey on Transfer Learning** // IEEE TKDE. — 2010. — Vol. 22, № 10. — P. 1345–1359. DOI: [10.1109/TKDE.2009.191](https://doi.org/10.1109/TKDE.2009.191).
- Ganin Y., Lempitsky V. **Unsupervised Domain Adaptation by Backpropagation** // ICML 2015. — P. 1180–1189.
- Zhang W., Deng L. F., Zhang L., Wu D. R. **A survey on negative transfer** // IEEE/CAA Journal of Automatica Sinica. — 2023. — Vol. 10, № 2. — P. 305–329. DOI: [10.1109/JAS.2022.106004](https://doi.org/10.1109/JAS.2022.106004).
- Loshchilov I., Hutter F. **Decoupled Weight Decay Regularization (AdamW)** // ICLR 2019. arXiv:[1711.05101](https://arxiv.org/abs/1711.05101).
- Loshchilov I., Hutter F. **SGDR: Stochastic Gradient Descent with Warm Restarts** // ICLR 2017. arXiv:[1608.03983](https://arxiv.org/abs/1608.03983).

## Снижение размерности и классические методы

- Pearson K. **On lines and planes of closest fit to systems of points in space** // Philosophical Magazine. — 1901. — Vol. 2, № 11. — P. 559–572.
- Jolliffe I. T. **Principal Component Analysis**. 2nd ed. — New York: Springer, 2002. — 487 p.
- Halko N., Martinsson P. G., Tropp J. A. **Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions** // SIAM Review. — 2011. — Vol. 53, № 2. — P. 217–288. DOI: [10.1137/090771806](https://doi.org/10.1137/090771806).
- Cover T. M., Thomas J. A. **Elements of Information Theory**. 2nd ed. — Hoboken, NJ: Wiley-Interscience, 2006. — 776 p. ISBN: 978-0-471-24195-9.
- Koechlin E., Summerfield C. **An information theoretical approach to prefrontal executive function** // Trends in Cognitive Sciences. — 2007. — Vol. 11, № 6. — P. 229–235. DOI: [10.1016/j.tics.2007.04.005](https://doi.org/10.1016/j.tics.2007.04.005).

## MLOps и инженерия ML-систем

- Sculley D., Holt G., Golovin D. et al. **Hidden Technical Debt in Machine Learning Systems** // NeurIPS 2015. — Vol. 28. — P. 2503–2511.
- Kreuzberger D., Kühl N., Hirschl S. **Machine Learning Operations (MLOps): Overview, Definition, and Architecture** // IEEE Access. — 2023. — Vol. 11. — P. 31866–31879. DOI: [10.1109/ACCESS.2023.3262138](https://doi.org/10.1109/ACCESS.2023.3262138).
- Eken B., Pallewatta S., Tran N. K., Tosun A., Babar M. A. **A Multivocal Review of MLOps Practices, Challenges and Open Issues** // arXiv preprint, 2024. arXiv:[2406.09737](https://arxiv.org/abs/2406.09737).

## Программная архитектура и инструменты

- Fielding R. T. **Architectural Styles and the Design of Network-based Software Architectures**: PhD dissertation. — University of California, Irvine, 2000. — 162 p. URL: <https://ics.uci.edu/~fielding/pubs/dissertation/top.htm>.
- Parnas D. L. **On the Criteria To Be Used in Decomposing Systems into Modules** // Communications of the ACM. — 1972. — Vol. 15, № 12. — P. 1053–1058. DOI: [10.1145/361598.361623](https://doi.org/10.1145/361598.361623).
- Nygard M. T. **Release It! Design and Deploy Production-Ready Software**. 2nd ed. — Raleigh, NC: The Pragmatic Bookshelf, 2018. — 360 p.
- Boettiger C. **An introduction to Docker for reproducible research** // ACM SIGOPS OSR. — 2015. — Vol. 49, № 1. — P. 71–79. DOI: [10.1145/2723872.2723882](https://doi.org/10.1145/2723872.2723882).
- Ramírez S. **FastAPI: Modern, Fast Web Framework for Building APIs with Python**. URL: <https://fastapi.tiangolo.com/>.
- Colvin S., Montague A., Gallin K. et al. **Pydantic: Data Validation Using Python Type Hints**. URL: <https://docs.pydantic.dev>.
- **PostgreSQL 16 Documentation** / The PostgreSQL Global Development Group. URL: <https://www.postgresql.org/docs/16/>.
- **SQLAlchemy 2.0 Documentation**. URL: <https://docs.sqlalchemy.org/en/20/>.
- **Alembic Documentation**. URL: <https://alembic.sqlalchemy.org/en/latest/>.
- **RabbitMQ Server Documentation** / VMware. URL: <https://www.rabbitmq.com/documentation.html>.
- Pedregosa F., Varoquaux G., Gramfort A. et al. **Scikit-learn: Machine Learning in Python** // JMLR. — 2011. — Vol. 12. — P. 2825–2830.
- Paszke A., Gross S., Massa F. et al. **PyTorch: An Imperative Style, High-Performance Deep Learning Library** // NeurIPS 2019. — Vol. 32. — P. 8024–8035.
- Harris C. R., Millman K. J., van der Walt S. J. et al. **Array Programming with NumPy** // Nature. — 2020. — Vol. 585. — P. 357–362. DOI: [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
- McKinney W. **Data Structures for Statistical Computing in Python** // SciPy 2010. — P. 56–61. DOI: [10.25080/Majora-92bf1922-00a](https://doi.org/10.25080/Majora-92bf1922-00a).
- Clark A. et al. **Pillow (PIL Fork) Documentation**. URL: <https://pillow.readthedocs.io>.
- **DeepPavlov/rubert-base-cased: Russian BERT Model** // Hugging Face Hub. URL: <https://huggingface.co/DeepPavlov/rubert-base-cased>.

## Отраслевые отчёты и рыночные данные

### Международные

- **OECD/EUIPO. Mapping Global Trade in Fakes 2025: Global trends and enforcement challenges**. — Paris/Alicante, 2025. URL: <https://www.oecd.org/publications/global-trade-in-fakes-2025>.
- **Capital One Shopping. E-commerce fraud statistics 2024–2025**. URL: <https://capitaloneshopping.com/research/ecommerce-fraud-statistics/>.
- **Capital One Shopping Research. Counterfeit Goods Statistics 2024**. URL: <https://capitaloneshopping.com/research/counterfeit-statistics>.
- **Cybersource. Global Fraud Report 2024**. — Cybersource by Visa, 2024.
- **Statista. Global e-commerce retail sales 2024–2027**. URL: <https://www.statista.com/statistics/379046/>.
- **Amazon. 2024 Brand Protection Report: Trustworthy Shopping at Amazon**. URL: <https://trustworthyshopping.aboutamazon.com/2024-brand-protection-report>.
- **National Retail Federation. 2025 Consumer Returns in the Retail Industry**. — 2026. URL: <https://nrf.com/research/2025-consumer-returns-retail-industry>.
- **McKinsey & Company. From cost center to competitive advantage: Modernizing reverse logistics with AI**. — McKinsey Logistics Practice, 2025.
- **DC Velocity. Study: Over 15 % of all retail returns in 2024 were fraudulent**. — 2025.

### Российский рынок

- **Data Insight. Интернет-торговля в России 2026: отчёт**. — Москва, 2026. URL: <https://datainsight.ru/DI_eCommerce_2025>.
- **Объём заказов на российских маркетплейсах в 2025 году достиг 9–10 млрд единиц** // Forbes.ru, 2025.
- **Рост рынка e-commerce в России замедлился в 2024 году** // Forbes.ru, 18.05.2025.
- **Forbes Russia. Контрафакт на российских маркетплейсах: масштаб проблемы и автоматизация модерации в 2026 году**. — 2026.
- **Forbes Russia. Wildberries, Ozon и «Яндекс Маркет» создали единую систему для борьбы с контрафактом**. — Август 2022, дополнения 2024.
- **Forbes Russia. Дорожная карта между «Честным ЗНАКом», Wildberries и Ozon**. — Январь 2026.
- **ФАС России: маркетплейсы заблокировали свыше 2 млн контрафактных товаров в I квартале 2024 года** // CNews.ru, 2024.
- **ФАС России. 2,4 млн карточек товаров было заблокировано на маркетплейсах по жалобам правообладателей**: офиц. сообщение, 19.04.2024. URL: <https://fas.gov.ru/news/33041>.
- **Электроника контрафактически не растёт** // ComNews, 17.06.2025.
- **Sidorin Lab. Жалобы покупателей на подделки на Ozon и Wildberries за 2024 год превысило 264 тыс.** // NEWS.ru, 2024.
- **Sidorin Lab. Подделай меня, если сможешь! Анализ и оценка распространённости подделок на маркетплейсах**: исследовательский кейс (август 2023 – июль 2024).
- **The Moscow Times. ФАС уличил Ozon в бесконтрольной раздаче знака «оригинал» товарам**. — Апрель 2026.
- **Коммерсантъ. Доля контрафактной электроники на маркетплейсах в 2024 г.**. — 2024.
- **Честный ЗНАК. Обязательная маркировка товаров в 2025 году**. Официальный сайт системы маркировки.
- **Yandex Market. Press Release: ML-Accelerated Counterfeit Detection**. — 2024.
- **Ozon Holdings PLC. Финансовые результаты за четвёртый квартал 2025 года и 2025 год: годовой отчёт по МСФО**. — 2026. URL: <https://ir.ozon.com/ru/sth/ozon-obyavlyaet-finansovye-rezultaty-za-chetvertyy-kvartal-2025-goda-i-2025-god-18833bbf>.
- **ECDB. Ozon retailer profile: GMV, category breakdown, market structure 2025**. URL: <https://ecdb.com/resources/sample-data/retailer/ozon>.
- **TAdviser. Финансовые показатели Ozon**. — 2025.
- **Ozon eCup 2025: соревнование по детекции контрафактных товаров** // Ozon Tech. URL: <https://ecup.ozon.ru/>.

## Нормативные документы

- **ГОСТ Р 7.0.5–2008. Система стандартов по информации, библиотечному и издательскому делу. Библиографическая ссылка. Общие требования и правила составления**. — Введ. 2009-01-01. — Москва: Стандартинформ, 2008. — 23 с.
