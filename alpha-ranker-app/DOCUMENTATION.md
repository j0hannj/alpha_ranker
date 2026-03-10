# Alpha Ranker — Documentation complète

Documentation de l’application : tous les écrans, le fonctionnement et les concepts (stabilité, modèles, données).

---

## 1. Vue d’ensemble

**Alpha Ranker** (MyFinancialAdvisor) est une application desktop Windows pour le stock picking et le suivi de portefeuille, avec un moteur quantitatif (ML) et un assistant IA.

- **Onglets principaux** : Portfolio | Rankings | Build | Projections | Backtest | Universe | Settings  
- **Panneau droit** : **AI Advisor** (chat partagé sur tous les onglets).  
- **Données** : Yahoo Finance (prix, fondamentaux), FRED (macro), FMP optionnel (fondamentaux, ISIN).  
- **Modèle** : ensemble de régressions (LightGBM, XGBoost, Ridge, RandomForest, etc.) qui prédit le rendement futur ; classement des titres par alpha prédit.  
- **Persistance** : SQLite (portefeuille, cache API, historique des classements, config).

---

## 2. Panneau AI Advisor (chat)

- **Position** : à droite de la fenêtre, visible sur tous les onglets.  
- **Contenu** : zone de chat + raccourcis (« Analyze portfolio », « Risks? », « Top picks? », « Macro? ») + champ « Ask anything... » + bouton Send.  
- **Fonction** : poser des questions en langage naturel. L’IA a accès au portefeuille, aux résultats du modèle, au régime macro et peut proposer des **actions** (ex. refresh_prices, run_model, fetch_large_cap_isins, search_news).  
- **Moteurs** : Anthropic (clé API) ou Ollama (local). Le libellé en haut indique le moteur utilisé.  
- **Briefing** : au démarrage, si une IA est configurée, un court « market briefing » peut s’afficher dans le chat.  
- **Boutons** : ▶ afficher/masquer le panneau ; + / − pour élargir/réduire la largeur.

---

## 3. Onglet Portfolio

**Rôle** : afficher et gérer les positions (holdings), exposition sectorielle, signaux de vente et projection DCA.

- **Cartes du haut** : Total Value, Total Cost, P&L, Performance (%).  
- **Tableau Holdings** : ISIN, Ticker, Type (stock/etf), Qty, Cost (PRU), Price, Last update, Value, P&L, P&L%.  
- **Boutons** : **Refresh** (mise à jour des prix), **+ Add** (nouvelle position), **Delete** (supprimer la sélection). Double-clic sur une ligne = **éditer** (quantité, coût, stratégie).  
- **Ajout** : dialogue avec Ticker ou ISIN, quantité, coût, type (stock/etf), devise. Si vous entrez un ISIN, il est résolu vers un ticker.  
- **Colonne droite** :  
  - **Sector Exposure** : répartition sectorielle (texte + barres).  
  - **Sell Signals** : signaux de vente (Signal, Urgency, Strategy, Pred%, Rank, etc.) avec **Accept Sell** et **History**.  
  - **DCA Projection** : courbe de projection (capitalisation DCA).  
- **En bas** : **Portfolio Value (1Y)** : courbe de la valeur du portefeuille sur un an.

Les prix sont mis à jour au démarrage et peuvent être rafraîchis manuellement ou via une action IA.

---

## 4. Onglet Rankings

**Rôle** : afficher le classement des titres par alpha prédit (résultat du modèle) et l’état du modèle.

- **Bouton « Run Model »** : lance le pipeline complet (fetch données, entraînement, prédictions). Long (plusieurs minutes). Barre de progression et message de statut.  
- **Horizon** : boutons 3M, 6M, 12M, 24M, 10Y. Le classement affiché correspond à l’horizon sélectionné (meilleurs performeurs sur cet horizon).  
- **Tableau** : #, ISIN, Ticker, Name, Sector, Change (évolution du rang), **Stability** (stabilité du signal), Predicted (rendement prédit), Conv (conviction), Analyst, Sent (sentiment), P/E, Grwth, FCF, Mom.  
- **Infos sous le tableau** :  
  - **Model health** : verdict (ex. OK / Warning), Mean IC, Hit rate, ICIR.  
  - **Model comparison** : IC par modèle (LightGBM, XGBoost, etc.) quand plusieurs modèles sont utilisés.  
- **Data last updated** : date de dernière mise à jour des données marché.  
- Double-clic sur une ligne = popup détaillée sur le titre (explication, features, etc.).  
- Survol d’une cellule = tooltip (ex. détail pour Stability : « Signal stability: stable (rank std: 2.50) »).

Le classement est enrichi avec **rank_delta**, **stability_index** et **movement_classification** à partir de l’historique des runs (voir section Stabilité plus bas).

---

## 5. Onglet Build

**Rôle** : construire une proposition de portefeuille (liste d’achats/ventes) à partir du budget et du classement du modèle.

- **Contrôles** : Budget (EUR), Fees (EUR/trade), Max positions, Confidence (Low / Medium / High).  
- **Slider** : part ETF vs Stock Picks (ex. 30 % ETF / 70 % Stocks).  
- **Weighting** : Equal Weight, Risk Parity, Alpha Weight.  
- **Strategy override** : Auto, LONG_TERM, MEDIUM_TERM, SHORT_TERM, DONT_SELL.  
- **Bouton « Generate »** : lance le moteur quant (filtrage par conviction, allocation sous contrainte de budget, coûts de transaction, horizon). Utilise le classement de l’onglet Rankings (et l’horizon actuellement sélectionné).  
- **Tableau** : Src (ETF/Stock/SELL), ISIN, Ticker, Name, Sector, Alpha, Conf, Price, Invested, Qty, Horizon, Target, Stop, Consensus, Reason.  
- **Boutons** : **Add All to Portfolio** (ajouter toutes les lignes au portefeuille), **Remove Selected** (retirer la sélection du tableau).  
- **Ask AI to Adjust** : envoie la proposition au LLM pour qu’il suggère des ajustements (texte dans le chat / statut).

Il faut avoir lancé le modèle (Rankings → Run Model) avant de générer une proposition.

---

## 6. Onglet Projections

**Rôle** : projections de prix à 3M, 6M, 12M, 24M et 10Y pour chaque position du portefeuille, basées sur le modèle.

- **Bouton « Refresh »** : recalcule les projections à partir du portefeuille actuel et des résultats du modèle.  
- **Tableau** : ISIN, Ticker, Name, Price, Qty, Value, colonnes 3M, 6M, 12M, 24M, 10Y (prix projeté + gain %), Model ret.  
- **Légende** : résumé 12M (valeur projetée, %), valeur actuelle, horizon du modèle utilisé.

Nécessite un run du modèle et des positions dans le portefeuille.

---

## 7. Onglet Backtest

**Rôle** : backtester la stratégie (walk-forward ensemble) sur des données passées.

- **Paramètres** : Horizon (3, 6, 12, 24, 120 mois), From (année de début : 2019, 2020, 2021, 2022).  
- **Bouton « Run Backtest »** : fetch des données, entraînement walk-forward sur la période, calcul des métriques. Le log s’affiche en bas.  
- **Résultats** : cartes (IC, IR, L/S return, Hit rate) et graphiques (courbe d’équité L/S, IC par période).  
- **Log** : messages de progression (fetch, modélisation, nombre de titres avec ISIN, etc.).

Seuls les titres avec **ticker et ISIN** sont utilisés pour la modélisation (comme pour le run principal).

---

## 8. Onglet Universe

**Rôle** : gérer la liste des titres utilisés comme univers (données et modèle).

- **Haut** : titre « Universe — Tickers connus », **Actualiser la liste**, compteur **X titres — Y ISIN**.  
- **Auto-fill** :  
  - **Cible** : nombre de titres cible (ex. 500).  
  - **Région** : Global, US uniquement, Europe uniquement, Asie uniquement.  
  - **Bouton « Remplir automatiquement »** : remplit l’univers avec des large caps (Wikipedia, FMP, etc.) jusqu’à la cible, en résolvant les ISIN. Barre de progression et **Annuler**. À la fin : dialogue récap (ajoutés, échecs ISIN, taille univers, sources utilisées).  
- **Ajout manuel** : champ **ISIN** + bouton **Ajouter** (ajoute l’ISIN à l’univers ; le ticker est résolu si possible).  
- **Liste gauche** : arbre ISIN / Ticker (tous les titres en base). Clic sur une ligne = graphique de prix (5 ans) à droite.  
- **Compteur** : « X titres — Y ISIN » = nombre de titres stockés et nombre de ceux qui ont un ISIN résolu.

La modélisation (Rankings, Build, Backtest) n’utilise que les titres pour lesquels **ticker et ISIN** sont connus.

---

## 9. Onglet Settings

**Rôle** : configuration des clés API, du moteur de décision, du modèle et du profil d’investissement.

- **Data freshness** : dates de dernière mise à jour (prix, fondamentaux, macro).  
- **API Keys** : Anthropic, FRED, FMP, Alpha Vantage (champs masqués).  
- **Portfolio decision engine** : frais de courtier, spread (bps), slippage (bps), horizon de détention par défaut.  
- **Investment horizon (per strategy)** : horizons SHORT_TERM, MEDIUM_TERM, LONG_TERM, max holding, review frequency.  
- **Sell Signal Configuration** : mode par stratégie (disabled / passive / active).  
- **Risk & regime** : max sector weight, max turnover, max position, seuils bull/bear (6m return), seuils de volatilité.  
- **Engine / Model configuration** :  
  - Modèles activés (cases à cocher : LightGBM, XGBoost, RandomForest, Ridge, ElasticNet, TCN, LSTM).  
  - Execution mode : single / multi / all.  
  - Single model (si mode = single).  
  - Ensemble method : simple_average, ic_weighted_average, stacked_meta_model.  
  - Prediction horizon (mois), winsorization, rank normalization, sector neutralization, feature decorrelation, GPU pour TCN/LSTM.  
  - Poids des horizons (term structure), lookback, training window, n_estimators, max_depth, learning_rate, ridge_alpha, etc.  
  - Feature groups (momentum, volatility, value, quality, trend, macro, sentiment, size).  
- **Investment Profile** : DCA mensuel, devise, risk profile, target amount, horizon d’investissement.  
- **Sauvegarde** : les réglages sont persistés (DB / config).

---

## 10. Workflow typique

1. **Premier lancement** : configurer les clés API (Settings), notamment FMP pour un univers et des ISIN complets.  
2. **Universe** : définir une cible (ex. 500), lancer « Remplir automatiquement » ou ajouter des ISIN à la main. Vérifier « X titres — Y ISIN ».  
3. **Rankings** : cliquer sur **Run Model**. Attendre la fin (données + entraînement multi-horizons). Consulter le classement par horizon (3M, 6M, 12M, etc.).  
4. **Portfolio** : ajouter des positions (+ Add) ou importer. Rafraîchir les prix (Refresh). Consulter les signaux de vente et la projection DCA.  
5. **Build** : saisir budget et options, Generate. Ajuster éventuellement avec « Ask AI to Adjust », puis « Add All to Portfolio » si souhaité.  
6. **Projections** : Refresh pour voir les projections de prix par position.  
7. **Backtest** : optionnel ; choisir horizon et année, Run Backtest pour évaluer la stratégie sur le passé.  
8. **AI Advisor** : à tout moment, poser des questions (portefeuille, risques, top picks, macro) ou déclencher des actions (refresh, run model, etc.).

---

## 11. Stabilité (Stability) et classement

### Qu’est-ce que la stabilité ?

La **stabilité** mesure à quel point le **rang** d’un titre dans le classement varie d’un run à l’autre.

- **Stability index** : écart-type des rangs sur les N derniers runs (ex. 5). **Faible** = rang stable ; **élevé** = rang volatile.  
- **Rank delta** : rang actuel − rang au run précédent (négatif = monte, positif = descend).  
- **Colonne Stability** (Rankings) : affiche la **classification du mouvement** (stable, moderate, volatile, etc.) et éventuellement l’indice (rank std) en tooltip.

### Classifications (movement_classification)

| Valeur | Signification |
|--------|----------------|
| **new** | Jamais classé auparavant. |
| **returning** | Déjà vu, mais absent des N derniers runs. |
| **stable** | Rang peu variable. |
| **moderate** | Variation modérée. |
| **volatile** | Forte variation de rang. |
| **improving** / **degrading** | Tendance à monter/descendre sur les derniers runs. |
| **rapidly improving** / **rapidly declining** | Forte variation récente. |
| **insufficient_data** | Pas assez d’historique. |

L’historique des rangs est enregistré en base à chaque run ; les indicateurs sont calculés côté app (voir `ranking_insights`).

---

## 12. Types de modèles

| Modèle | Type | Rôle |
|--------|------|------|
| **LightGBM** | Gradient Boosting | Interactions non linéaires, rapide, SHAP natif. |
| **XGBoost** | Gradient Boosting | Complémentaire à LightGBM, régularisation. |
| **RandomForest** | Bagging | Plus stable, moins précis. |
| **Ridge** | Régression linéaire L2 | Relations monotones, coefficients interprétables. |
| **ElasticNet** | Régression L1+L2 | Sélection de variables. |
| **TCN** | Réseau temporel (PyTorch) | Optionnel, motifs temporels. |
| **LSTM** | Réseau récurrent (PyTorch) | Optionnel, séquences. |

- **Execution mode** : single (un seul modèle), multi (plusieurs), all (tous).  
- **Ensemble** : simple_average, ic_weighted_average, stacked_meta_model.  
- **Horizons** : 3, 6, 12, 24, 120 mois ; le classement principal utilise en général l’horizon 12M.

Détails et paramètres : voir `engine_config` et onglet Settings → Engine / Model configuration.

---

## 13. Données et cache

- **Prix** : téléchargés **mois par mois** (cache par mois). Si l’univers s’agrandit, seuls les **nouveaux** tickers sont téléchargés pour chaque mois puis fusionnés au cache.  
- **Universe** : liste stockée en base + ISIN map (ticker → ISIN). Seuls les titres avec **ticker et ISIN** sont utilisés pour l’entraînement et les prédictions.  
- **Résolution ISIN** : chaîne cache local → fichier statique → FMP (si clé) → yfinance.  
- **Fundamentals** : Yahoo (et FMP si clé), cache par ticker.

---

## 14. Fichiers utiles

- **ARCHITECTURE.md** : vision, stack, structure des dossiers.  
- **DOCUMENTATION.md** (ce fichier) : tous les écrans, workflow, stabilité, modèles, données.

Pour toute question sur un écran ou une action précise, se référer à la section correspondante ci-dessus.
