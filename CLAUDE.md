# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Rôle
Tu es un Data Engineer et Développeur Python Senior, expert en web scraping, analyse de données financières et intégration de LLM (LangChain/OpenAI).

## Objectif
Développer un outil en Python orienté objet (OOP) permettant de générer un benchmark financier automatisé. L'outil doit identifier des entreprises comparables, récupérer leurs rapports, extraire des KPIs via l'IA, et générer un fichier Excel propre.

## Règles de développement (Strict) :

Utiliser une architecture modulaire (un fichier par composant logique).
Utiliser le typage strict (Type hints).
Intégrer une gestion robuste des erreurs (try/except) et un logger (logging) au lieu de simples print().
Commenter le code sous format Docstring (Google style).

## Stack Technique Imposée :

Sourcing financier : yfinance ou appels API REST simples.
Scraping/Requêtes : requests, beautifulsoup4.
Extraction PDF : pdfplumber.
Extraction IA : langchain, couplé à pydantic pour forcer un output JSON structuré.
Données et Export : pandas, openpyxl.

## Architecture du Pipeline :

Module 1 : PeerGroupFinder
Input : Un ticker (ex: "AAPL" ou "SUD EST BETON" si API Pappers), un secteur cible, et des fourchettes (CA, effectifs).
Traitement : Interroge l'API cible pour retourner une liste d'entreprises correspondantes.
Output : Une liste de dictionnaires [{"nom": "...", "ticker": "...", "url_investisseur": "..."}].

Module 2 : ReportFetcher
Input : La liste générée par le Module 1.
Traitement : Pointe vers une base de données de rapports (ex: SEC EDGAR via API) ou utilise une méthode de fallback (scraping web de base) pour trouver l'URL du PDF du dernier rapport annuel. Télécharge le PDF dans un dossier local /data/reports/.
Output : Met à jour la liste avec le chemin local du PDF {"chemin_pdf": "./data/reports/nom_rapport.pdf"}.

Module 3 : KpiExtractorLLM
Input : Le chemin d'un PDF.
Traitement :
Utilise pdfplumber pour extraire le texte des 10 premières pages (ou de la section financière).
Envoie ce texte à un LLM via LangChain.
Crucial : Utilise Pydantic pour définir un schéma de sortie strict contenant : capex (float), marge_brute (float), ratio_endettement (float).
Output : Un objet JSON/Dictionnaire validé contenant les KPIs.

Module 4 : ExcelExporter
Input : Une liste de tous les KPIs extraits pour toutes les entreprises.
Traitement : Charge les données dans un pandas.DataFrame. Calcule d'éventuels ratios manquants.
Output : Exporte un fichier Excel .xlsx formaté (largeur de colonnes ajustée, format pourcentage/monétaire appliqué via openpyxl).

## État actuel du projet et Architecture

Architecture du module 1

  BasePeerFinder (ABC)
  ├── YFinancePeerFinder     ← sociétés cotées (yfinance >= 0.2.44)
  │   ├── _get_industry_candidates()   yf.Industry / yf.Sector
  │   └── _enrich_and_filter()         appels individuels throttlés (0.25s)
  └── PappersPeerFinder      ← sociétés françaises (cotées ou non)
      └── codes tranches CA + effectifs auto-calculés
  
  PeerGroupFinder            ← point d'entrée, route vers la bonne stratégie

Architecture du module 2

  Flux de décision pour chaque société :

  fetch_one(company)
  │
  ├── ticker US pur (ex: "AAPL", sans ".") ?
  │   └── EdgarReportFetcher
  │       ├── company_tickers.json → CIK
  │       ├── submissions/{CIK}.json → dernier 10-K
  │       └── Téléchargement du document principal
  │           ✓ trouvé → retourne Path
  │
  └── fallback → WebScrapingReportFetcher
      ├── yfinance.info["website"] → _find_ir_page()
      │   (teste /investor-relations, /investors, etc. via HEAD)
      ├── sinon : url_investisseur du Module 1
      └── _scrape_for_pdf() : score chaque lien <a>
          (mots-clés + bonus .pdf) → télécharge le meilleur

  Note : _download_file() déduit l'extension réelle depuis Content-Type (PDF vs HTM/XBRL).

### Architecture du module 3

  Stratégie par type de fichier :

  Fichier .htm / .html / .xbrl -> XbrlKpiExtractor en premier (sans LLM)
    Si >= 2 KPIs trouvés -> retourne directement
    Sinon -> bascule sur TextExtractor (HtmlTextExtractor) + LlmKpiExtractor
  Fichier .pdf -> TextExtractor (PdfTextExtractor) + LlmKpiExtractor directement

  Classes :

  KpiResult
    Schéma Pydantic : capex, marge_brute, ratio_endettement, devise, annee_exercice, source

  PdfTextExtractor
    Extraction pdfplumber — pages financières en priorité (mots-clés), fallback 10 premières pages

  HtmlTextExtractor
    Extraction BeautifulSoup sur HTML/iXBRL — supprime le bruit (script/style),
    sélectionne les lignes proches des mots-clés financiers

  TextExtractor
    Routeur — délègue à PdfTextExtractor ou HtmlTextExtractor selon l'extension du fichier

  XbrlKpiExtractor
    Extraction directe iXBRL sans LLM — parse ix:nonFraction, résout les contextes
    annuels/instantanés, couvre CAPEX / GrossProfit / Revenues / Debt / Equity
    via concepts US-GAAP. is_sufficient() vérifie qu'au moins 2 KPIs sont trouvés.

  LlmKpiExtractor
    Chaîne LangChain — ChatOpenAI.with_structured_output(KpiResult) garantit un JSON validé

  KpiExtractorLLM
    Orchestrateur (point d'entrée) — extract_all(companies), extract_one_company(company),
    _extract_one() + _try_xbrl()

### Architecture du module 4

  export() accepte un paramètre optionnel `initial_company` (société de référence) affichée
  en tête de tableau avec un style doré distinct, séparée du peer group par une ligne
  visuelle "--- Groupe de comparables ---".

  Méthodes :

  export(companies, filename, initial_company)
    Point d'entrée — orchestre toutes les étapes

  _build_dataframe(companies)
    Sélectionne et renomme les colonnes via _COLUMN_MAP, convertit les types

  _compute_missing_ratios(df)
    Point d'extension pour dériver des ratios depuis données brutes ; logue les NaN

  _compute_summary_stats(df)
    Calcule médiane, moyenne, min, max sur les 3 colonnes numériques

  _write_excel(df, stats, path, initial_df)
    Coordonne l'écriture openpyxl : société initiale -> séparateur -> peers -> synthèse

  _write_initial_row()
    Ligne société de référence (fond jaune pâle, police gras doré)

  _write_peers_separator()
    Ligne "--- Groupe de comparables ---" (fond vert pâle)

  _write_separator_row()
    Ligne "--- Synthèse du peer group ---" (fond bleu pâle)

  _auto_fit_columns(ws, df, cols, initial_df)
    Ajuste chaque largeur colonne au contenu le plus long (plafond 60 car.)
    Prend en compte initial_df, les données du peer group et les libellés de synthèse
