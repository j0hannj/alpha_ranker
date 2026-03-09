# ALPHA RANKER — Desktop App Architecture
# ==========================================

## Vision
App desktop Windows native. Double-clic = ca s'ouvre.
Pas de terminal, pas de navigateur, pas de batch.
Un outil de stock picking + suivi de portefeuille + agent IA.

## Stack technique

### GUI: CustomTkinter (Python)
- Fenetres natives Windows, look moderne (dark mode natif)
- Zero HTML/CSS/JS/Node — tout en Python
- Tabs, graphiques, tableaux, inputs
- matplotlib integre pour les charts

### Data & ML
- LightGBM pour le modele de stock picking
- yfinance pour les prix live + fondamentaux
- FRED pour les indicateurs macro
- SQLite pour persister le portefeuille (plus de JSON a editer)
- pandas/numpy pour le processing

### Agent IA
- API Anthropic (Claude Sonnet) pour l'analyse
- L'agent a acces a: ton portefeuille, les resultats du modele, le regime macro
- Il genere des suggestions personnalisees avec raisonnement
- Il peut repondre a des questions ("pourquoi tu me recommandes XOM?")
- Historique des conversations sauvegarde en local

### Packaging
- PyInstaller pour generer un .exe standalone
- Un seul fichier a double-cliquer
- Premiere execution: installe les dependances automatiquement

## Structure de l'app

```
alpha-ranker/
  main.pyw              <- Point d'entree (double-clic, pas de console)
  setup.bat             <- Installation one-time
  requirements.txt
  
  src/
    app.py              <- Fenetre principale + navigation
    ui/
      portfolio_tab.py  <- Onglet Portefeuille (tableau, PnL, sectors)
      rankings_tab.py   <- Onglet Rankings (resultats du modele)
      invest_tab.py     <- Onglet Investir (suggestions + allocation)
      agent_tab.py      <- Onglet Agent IA (chat + analyse)
      model_tab.py      <- Onglet Modele (features, diagnostics)
      settings_tab.py   <- Parametres (API keys, preferences)
    
    core/
      model.py          <- LightGBM training + prediction
      data.py           <- Fetch yfinance + FRED + AV + FMP
      portfolio.py      <- SQLite CRUD + calcul PnL
      agent.py          <- Agent IA (Anthropic API)
      macro.py          <- Regime macro classification
    
    db/
      portfolio.db      <- SQLite (cree automatiquement)
      model_cache.pkl   <- Modele entraine (cache)
      history.json      <- Historique agent IA
```

## Onglets de l'app

### 1. Portefeuille
- Tableau avec: Ticker | Type | Qty | PRU | Prix act. | Valeur | P&L | P&L%
- Boutons: Ajouter | Modifier | Supprimer | Rafraichir prix
- Pie chart exposition sectorielle (ETFs decomposes)
- Courbe valeur portefeuille dans le temps
- Sauvegarde automatique en SQLite

### 2. Rankings
- Resultats du modele LightGBM
- Horizon selectionnable (3/6/12/24 mois)
- Filtres: secteur, conviction min, market cap
- Tri: score, conviction, P/E, growth, FCF yield
- Details expandables par action
- Bouton "Lancer le modele" (tourne en background thread)

### 3. Investir
- Input: montant a investir
- 3 strategies: DCA Pur | Equilibre | Alpha
- Propositions generees par le moteur + l'agent IA
- Chaque suggestion avec raisons detaillees
- Bouton "Demander a l'agent" pour creuser

### 4. Agent IA
- Interface chat type ChatGPT
- L'agent a le contexte complet: portefeuille, modele, macro
- Questions possibles:
  "Analyse mon portefeuille"
  "Pourquoi XOM est #1?"
  "Je veux investir 500 EUR, quoi faire?"
  "Compare NVDA vs AVGO"
  "Quel est mon risque si le petrole baisse?"
- Historique sauvegarde

### 5. Modele
- Feature importances (bar chart)
- Diagnostics (R2, n_features, etc.)
- Bouton "Re-entrainer"
- Log du dernier training

### 6. Settings
- API keys (FRED, Alpha Vantage, FMP, Anthropic)
- Preferences (devise, DCA mensuel, risk profile)
- Exclusions (secteurs, tickers)
- Theme (dark/light)

## Agent IA — Detail

L'agent utilise Claude Sonnet via l'API Anthropic.
A chaque requete, on lui injecte dans le system prompt:
- Le portefeuille complet avec PnL
- Les top 20 picks du modele
- Le regime macro actuel
- Les feature importances
- L'historique recent des suggestions

L'agent peut:
- Analyser une position specifique
- Comparer deux actions
- Generer un plan d'investissement
- Expliquer pourquoi le modele surpondere un secteur
- Alerter sur des risques (concentration, drawdown, macro)

## Workflow utilisateur

1. Double-clic sur l'app
2. L'app s'ouvre sur l'onglet Portefeuille
3. Si premiere fois: popup "Ajouter vos positions"
4. Les prix se mettent a jour automatiquement (toutes les 15 min)
5. Onglet Rankings: clic "Lancer le modele" (tourne en background, ~15 min)
6. Onglet Investir: entre un montant, recoit des suggestions
7. Onglet Agent: pose des questions, recoit des analyses
