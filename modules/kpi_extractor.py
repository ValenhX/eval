"""Module 3 : KpiExtractorLLM

Extrait les KPIs financiers clés d'un rapport annuel PDF via LangChain et un LLM.

Pipeline :
1. Extraction du texte brut avec pdfplumber (sections financières en priorité,
   fallback sur les N premières pages).
2. Envoi du texte au LLM (OpenAI via LangChain) avec un prompt structuré.
3. Validation de la réponse via Pydantic → retourne un objet KpiResult typé.
"""

import logging
import re
from datetime import date
from pathlib import Path
from typing import Optional

import pdfplumber
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MAX_PAGES = 10
# Limite de caractères envoyés au LLM pour rester dans la fenêtre de contexte
_MAX_CHARS = 12_000

# Mots-clés identifiant les pages de sections financières dans les rapports
_FINANCIAL_KEYWORDS = [
    "consolidated statements",
    "états financiers",
    "financial statements",
    "balance sheet",
    "bilan",
    "income statement",
    "compte de résultat",
    "cash flow",
    "flux de trésorerie",
    "capital expenditure",
    "capex",
    "gross profit",
    "marge brute",
]


# ---------------------------------------------------------------------------
# Schéma de sortie Pydantic
# ---------------------------------------------------------------------------


class KpiResult(BaseModel):
    """KPIs financiers extraits d'un rapport annuel.

    Attributes:
        capex: Dépenses d'investissement (CAPEX) en millions d'unité monétaire.
        marge_brute: Marge brute en ratio décimal (ex: 0.42 pour 42 %).
        ratio_endettement: Ratio dette nette / EBITDA ou dette / capitaux propres.
        devise: Devise des montants monétaires (ex: "EUR", "USD").
        annee_exercice: Année fiscale des données (ex: 2023).
        source: Section ou tableau d'où proviennent les données extraites.
    """

    capex: Optional[float] = Field(
        None, description="CAPEX en millions d'unité monétaire"
    )
    marge_brute: Optional[float] = Field(
        None, ge=0.0, le=10.0, description="Marge brute (ratio décimal, ex: 0.42)"
    )
    ratio_endettement: Optional[float] = Field(
        None, description="Ratio d'endettement (dette nette / EBITDA ou dettes / CP)"
    )
    devise: Optional[str] = Field(None, description="Devise des montants (EUR, USD…)")
    annee_exercice: Optional[int] = Field(None, description="Année fiscale (ex: 2023)")
    source: Optional[str] = Field(
        None, description="Section ou tableau d'où proviennent les données"
    )


# ---------------------------------------------------------------------------
# Extraction du texte PDF
# ---------------------------------------------------------------------------


class PdfTextExtractor:
    """Extrait le texte d'un rapport PDF avec pdfplumber.

    Stratégie :
    1. Pages contenant des sections financières identifiées par mots-clés.
    2. Fallback sur les N premières pages si aucune section détectée.
    """

    def __init__(
        self, max_pages: int = _MAX_PAGES, max_chars: int = _MAX_CHARS
    ) -> None:
        """Args:
            max_pages: Nombre maximal de pages à lire en fallback.
            max_chars: Nombre maximal de caractères retournés.
        """
        self._max_pages = max_pages
        self._max_chars = max_chars

    def extract(self, pdf_path: Path) -> str:
        """Extrait le texte pertinent d'un PDF.

        Args:
            pdf_path: Chemin local du fichier PDF.

        Returns:
            Texte extrait, tronqué à max_chars.

        Raises:
            FileNotFoundError: Si le PDF n'existe pas.
            ValueError: Si le PDF est vide ou illisible.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = pdf.pages
                if not pages:
                    raise ValueError(f"PDF sans pages : {pdf_path}")

                text = self._extract_financial_sections(pages)
                if not text.strip():
                    text = self._extract_first_pages(pages)
        except (FileNotFoundError, ValueError):
            raise
        except Exception as exc:
            raise ValueError(
                f"Impossible de lire le PDF '{pdf_path}' : {exc}"
            ) from exc

        if not text.strip():
            raise ValueError(f"Aucun texte extractible dans '{pdf_path}'")

        truncated = text[: self._max_chars]
        logger.debug(
            "Texte extrait de '%s' : %d car. → tronqué à %d",
            pdf_path.name,
            len(text),
            len(truncated),
        )
        return truncated

    def _extract_financial_sections(self, pages: list) -> str:
        """Retourne le texte des pages contenant des sections financières.

        Args:
            pages: Pages du PDF (objets pdfplumber).

        Returns:
            Texte concaténé des pages financières détectées.
        """
        selected: list[str] = []
        for page in pages:
            page_text = page.extract_text() or ""
            lower = page_text.lower()
            if any(kw in lower for kw in _FINANCIAL_KEYWORDS):
                selected.append(page_text)
                if sum(len(t) for t in selected) >= self._max_chars:
                    break
        return "\n\n".join(selected)

    def _extract_first_pages(self, pages: list) -> str:
        """Extrait le texte des N premières pages.

        Args:
            pages: Pages du PDF.

        Returns:
            Texte concaténé des premières pages.
        """
        texts: list[str] = []
        for page in pages[: self._max_pages]:
            texts.append(page.extract_text() or "")
        return "\n\n".join(texts)


# ---------------------------------------------------------------------------
# Extraction du texte HTML / inline XBRL
# ---------------------------------------------------------------------------

_HTML_EXTENSIONS = {".htm", ".html", ".xbrl", ".xhtml"}

# Balises à supprimer avant extraction du texte
_HTML_NOISE_TAGS = ["script", "style", "head", "meta", "link"]


class HtmlTextExtractor:
    """Extrait le texte d'un rapport au format HTML ou inline XBRL (iXBRL).

    Stratégie :
    1. Suppression des balises de bruit (scripts, styles, en-têtes).
    2. Sélection des lignes proches des mots-clés financiers.
    3. Fallback sur les premières lignes si aucun mot-clé détecté.
    """

    _CONTEXT_LINES_AFTER = 25

    def __init__(self, max_chars: int = _MAX_CHARS) -> None:
        """Args:
            max_chars: Nombre maximal de caractères retournés.
        """
        self._max_chars = max_chars

    def extract(self, path: Path) -> str:
        """Extrait le texte pertinent d'un fichier HTML/XBRL.

        Args:
            path: Chemin local du fichier.

        Returns:
            Texte extrait, tronqué à max_chars.

        Raises:
            FileNotFoundError: Si le fichier n'existe pas.
            ValueError: Si aucun texte extractible.
        """
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            soup = BeautifulSoup(raw, "html.parser")
            for tag in soup(_HTML_NOISE_TAGS):
                tag.decompose()
            full_text = soup.get_text(separator="\n", strip=True)
        except Exception as exc:
            raise ValueError(
                f"Impossible de lire le fichier HTML '{path}' : {exc}"
            ) from exc

        if not full_text.strip():
            raise ValueError(f"Aucun texte extractible dans '{path}'")

        selected = self._extract_financial_sections(full_text)
        result = selected if selected.strip() else full_text
        truncated = result[: self._max_chars]
        logger.debug(
            "Texte HTML extrait de '%s' : %d car. → tronqué à %d",
            path.name, len(result), len(truncated),
        )
        return truncated

    def _extract_financial_sections(self, text: str) -> str:
        """Retourne les lignes situées autour des mots-clés financiers.

        Args:
            text: Texte brut complet extrait du HTML.

        Returns:
            Sous-ensemble de lignes pertinentes, ou chaîne vide si aucun match.
        """
        lines = text.splitlines()
        collected: list[str] = []
        budget = self._max_chars

        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in _FINANCIAL_KEYWORDS):
                end = min(len(lines), i + self._CONTEXT_LINES_AFTER)
                chunk = "\n".join(lines[i:end])
                collected.append(chunk)
                budget -= len(chunk)
                if budget <= 0:
                    break

        return "\n\n".join(collected)


# ---------------------------------------------------------------------------
# Extraction directe iXBRL (sans LLM)
# ---------------------------------------------------------------------------

# Concepts US-GAAP ordonnés par priorité pour chaque KPI
_XBRL_CAPEX = [
    "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
    "us-gaap:PaymentsToAcquireProductiveAssets",
    "us-gaap:CapitalExpendituresIncurredButNotYetPaid",
]
_XBRL_REVENUE = [
    "us-gaap:Revenues",
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
    "us-gaap:SalesRevenueNet",
]
_XBRL_GROSS_PROFIT = ["us-gaap:GrossProfit"]
_XBRL_COST_OF_REVENUE = [
    "us-gaap:CostOfRevenue",
    "us-gaap:CostOfGoodsSold",
    "us-gaap:CostOfGoodsAndServicesSold",
]
_XBRL_LONG_TERM_DEBT = [
    "us-gaap:LongTermDebt",
    "us-gaap:LongTermDebtNoncurrent",
    "us-gaap:LongTermDebtAndCapitalLeaseObligations",
]
_XBRL_SHORT_TERM_DEBT = [
    "us-gaap:ShortTermBorrowings",
    "us-gaap:LongTermDebtCurrent",
    "us-gaap:NotesPayableCurrent",
]
_XBRL_EQUITY = [
    "us-gaap:StockholdersEquity",
    "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]

# Durée d'un exercice annuel en jours (tolérance ±35 j pour les exercices décalés)
_ANNUAL_DAYS_MIN = 330
_ANNUAL_DAYS_MAX = 400


def _tags_by_localname(soup: BeautifulSoup, local_name: str) -> list:
    """Retourne tous les tags dont le nom local (après ':') correspond, insensible à la casse.

    Args:
        soup: Document BeautifulSoup.
        local_name: Nom local à chercher (ex : "context", "nonFraction").

    Returns:
        Liste de tags correspondants.
    """
    pattern = re.compile(rf"(?:[\w-]+:)?{re.escape(local_name)}$", re.IGNORECASE)
    return soup.find_all(pattern)


class XbrlKpiExtractor:
    """Extrait les KPIs directement depuis les balises iXBRL d'un rapport EDGAR.

    Ne fait aucun appel LLM — exploite les valeurs numériques structurées
    (``ix:nonFraction``) et les métadonnées de contexte (``xbrli:context``)
    présentes dans les fichiers inline XBRL (.htm / .xbrl) fournis par la SEC.

    Concept coverage :
    - CAPEX            → PaymentsToAcquirePropertyPlantAndEquipment (et variantes)
    - Marge brute      → GrossProfit / Revenues (calculé si GrossProfit absent)
    - Ratio endettement → (LongTermDebt + ShortTermBorrowings) / StockholdersEquity
    """

    def extract(self, path: Path) -> KpiResult:
        """Parse le fichier iXBRL et retourne les KPIs structurés.

        Args:
            path: Chemin local du fichier HTML ou XBRL.

        Returns:
            KpiResult — champs non trouvés laissés à None.

        Raises:
            FileNotFoundError: Si le fichier est absent.
            ValueError: Si le document est illisible ou ne contient aucun contexte XBRL.
        """
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

        try:
            html = path.read_text(encoding="utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            raise ValueError(f"Impossible de lire '{path}' : {exc}") from exc

        contexts = self._build_context_map(soup)
        if not contexts:
            raise ValueError(f"Aucun contexte XBRL trouvé dans '{path.name}'")

        annual_ctxs, fiscal_end = self._find_annual_contexts(contexts)
        instant_ctxs = self._find_instant_contexts(contexts, fiscal_end)

        values = self._build_value_map(soup)

        # CAPEX (flux de trésorerie — durée)
        capex = self._pick(values, _XBRL_CAPEX, annual_ctxs)
        if capex is not None:
            capex = abs(capex)  # convention : positif

        # Marge brute
        gross_profit = self._pick(values, _XBRL_GROSS_PROFIT, annual_ctxs)
        revenue = self._pick(values, _XBRL_REVENUE, annual_ctxs)
        marge_brute: Optional[float] = None
        if gross_profit is not None and revenue:
            marge_brute = gross_profit / revenue
        elif revenue:
            cost = self._pick(values, _XBRL_COST_OF_REVENUE, annual_ctxs)
            if cost is not None:
                marge_brute = (revenue - cost) / revenue

        # Ratio endettement = (dette LT + dette CT) / capitaux propres (bilan — instant)
        lt_debt = self._pick(values, _XBRL_LONG_TERM_DEBT, instant_ctxs)
        st_debt = self._pick(values, _XBRL_SHORT_TERM_DEBT, instant_ctxs) or 0.0
        equity = self._pick(values, _XBRL_EQUITY, instant_ctxs)
        ratio_endettement: Optional[float] = None
        if lt_debt is not None and equity:
            ratio_endettement = (lt_debt + st_debt) / equity

        devise = self._infer_currency(soup)
        annee_exercice = fiscal_end.year if fiscal_end else None

        found = [
            k for k, v in {
                "capex": capex,
                "marge_brute": marge_brute,
                "ratio_endettement": ratio_endettement,
            }.items()
            if v is not None
        ]
        source = f"iXBRL EDGAR ({', '.join(found)})" if found else "iXBRL EDGAR (vide)"
        logger.debug(
            "XBRL — capex=%s  marge_brute=%s  ratio_endettement=%s",
            capex, marge_brute, ratio_endettement,
        )
        return KpiResult(
            capex=capex,
            marge_brute=marge_brute,
            ratio_endettement=ratio_endettement,
            devise=devise,
            annee_exercice=annee_exercice,
            source=source,
        )

    # ------------------------------------------------------------------
    # Construction des tables internes
    # ------------------------------------------------------------------

    def _build_context_map(self, soup: BeautifulSoup) -> dict[str, dict]:
        """Construit id → {type, start, end, has_segment} pour chaque xbrli:context.

        Args:
            soup: Document parsé.

        Returns:
            Dictionnaire de contextes indexés par id.
        """
        ctx_map: dict[str, dict] = {}
        for tag in _tags_by_localname(soup, "context"):
            ctx_id = tag.get("id")
            if not ctx_id:
                continue

            period = next(iter(_tags_by_localname(tag, "period")), None)
            if not period:
                continue

            instant_tag = next(iter(_tags_by_localname(period, "instant")), None)
            start_tag = next(iter(_tags_by_localname(period, "startDate")), None)
            end_tag = next(iter(_tags_by_localname(period, "endDate")), None)
            has_segment = bool(_tags_by_localname(tag, "segment"))

            try:
                if instant_tag and instant_tag.get_text(strip=True):
                    ctx_map[ctx_id] = {
                        "type": "instant",
                        "start": None,
                        "end": date.fromisoformat(instant_tag.get_text(strip=True)),
                        "has_segment": has_segment,
                    }
                elif start_tag and end_tag:
                    ctx_map[ctx_id] = {
                        "type": "duration",
                        "start": date.fromisoformat(start_tag.get_text(strip=True)),
                        "end": date.fromisoformat(end_tag.get_text(strip=True)),
                        "has_segment": has_segment,
                    }
            except ValueError:
                continue

        return ctx_map

    def _build_value_map(self, soup: BeautifulSoup) -> dict[str, list[dict]]:
        """Indexe toutes les valeurs ix:nonFraction par concept (en minuscules).

        Les valeurs sont converties en millions dans l'unité de base du document
        via l'attribut ``scale`` (actual = displayed × 10^scale, puis ÷ 10^6).

        Args:
            soup: Document parsé.

        Returns:
            {concept_lower: [{"contextRef": str, "value": float}]}
        """
        val_map: dict[str, list[dict]] = {}
        for tag in _tags_by_localname(soup, "nonFraction"):
            concept = tag.get("name", "")
            ctx_ref = tag.get("contextref") or tag.get("contextRef", "")
            if not concept or not ctx_ref:
                continue

            raw = tag.get_text(strip=True).replace(",", "").replace("\xa0", "").replace(" ", "")
            if not raw or raw in ("-", "—"):
                continue

            try:
                numeric = float(raw)
            except ValueError:
                continue

            scale = 0
            try:
                scale = int(tag.get("scale", 0))
            except (ValueError, TypeError):
                pass

            value_millions = numeric * (10 ** scale) / 1_000_000

            if tag.get("sign") == "-":
                value_millions = -value_millions

            val_map.setdefault(concept.lower(), []).append(
                {"contextRef": ctx_ref, "value": value_millions}
            )

        return val_map

    # ------------------------------------------------------------------
    # Sélection des contextes de référence
    # ------------------------------------------------------------------

    def _find_annual_contexts(
        self, contexts: dict[str, dict]
    ) -> tuple[Optional[set[str]], Optional[date]]:
        """Retourne les ids des contextes de durée annuelle consolidés les plus récents.

        Args:
            contexts: Table des contextes issue de _build_context_map.

        Returns:
            (set d'ids valides, date de fin d'exercice) ou (None, None).
        """
        candidates = [
            (ctx_id, info)
            for ctx_id, info in contexts.items()
            if info["type"] == "duration"
            and not info["has_segment"]
            and info["start"] is not None
            and _ANNUAL_DAYS_MIN <= (info["end"] - info["start"]).days <= _ANNUAL_DAYS_MAX
        ]
        if not candidates:
            return None, None

        latest_end = max(info["end"] for _, info in candidates)
        ctx_ids = {cid for cid, info in candidates if info["end"] == latest_end}
        return ctx_ids, latest_end

    def _find_instant_contexts(
        self, contexts: dict[str, dict], fiscal_end: Optional[date]
    ) -> Optional[set[str]]:
        """Retourne les ids des contextes instantanés à la date de clôture.

        Args:
            contexts: Table des contextes.
            fiscal_end: Date de fin d'exercice (issue de _find_annual_contexts).

        Returns:
            Set d'ids ou None si fiscal_end est inconnu / aucun contexte trouvé.
        """
        if not fiscal_end:
            return None
        ids = {
            cid
            for cid, info in contexts.items()
            if info["type"] == "instant"
            and info["end"] == fiscal_end
            and not info["has_segment"]
        }
        return ids or None

    # ------------------------------------------------------------------
    # Helpers de lecture de valeur
    # ------------------------------------------------------------------

    def _pick(
        self,
        values: dict[str, list[dict]],
        concepts: list[str],
        valid_ctxs: Optional[set[str]],
    ) -> Optional[float]:
        """Retourne la première valeur trouvée pour la liste de concepts dans un contexte valide.

        Args:
            values: Table de valeurs issue de _build_value_map.
            concepts: Concepts à tester par ordre de priorité.
            valid_ctxs: Ensemble d'ids de contextes autorisés (None = tout accepter).

        Returns:
            Valeur en millions, ou None si aucun concept trouvé.
        """
        for concept in concepts:
            for entry in values.get(concept.lower(), []):
                if valid_ctxs is None or entry["contextRef"] in valid_ctxs:
                    return entry["value"]
        return None

    def _infer_currency(self, soup: BeautifulSoup) -> Optional[str]:
        """Déduit la devise depuis les balises xbrli:unit du document.

        Args:
            soup: Document parsé.

        Returns:
            Code ISO de devise (ex : "USD") ou None.
        """
        for tag in _tags_by_localname(soup, "unit"):
            measure = next(iter(_tags_by_localname(tag, "measure")), None)
            if measure:
                text = measure.get_text(strip=True).upper()
                for iso in ("USD", "EUR", "GBP", "CHF", "JPY", "CAD", "AUD"):
                    if iso in text:
                        return iso
        return None

    def is_sufficient(self, result: KpiResult) -> bool:
        """Vérifie si le résultat XBRL contient au moins deux KPIs renseignés.

        Args:
            result: Résultat de l'extraction XBRL.

        Returns:
            True si au moins 2 des 3 KPIs principaux sont non-None.
        """
        filled = sum(
            v is not None
            for v in (result.capex, result.marge_brute, result.ratio_endettement)
        )
        return filled >= 2


# ---------------------------------------------------------------------------
# Routeur — sélectionne l'extracteur selon l'extension du fichier
# ---------------------------------------------------------------------------


class TextExtractor:
    """Délègue l'extraction à PdfTextExtractor ou HtmlTextExtractor selon l'extension.

    Args:
        max_pages: Nombre maximal de pages PDF (fallback).
        max_chars: Nombre maximal de caractères retournés.
    """

    def __init__(
        self, max_pages: int = _MAX_PAGES, max_chars: int = _MAX_CHARS
    ) -> None:
        self._pdf = PdfTextExtractor(max_pages=max_pages, max_chars=max_chars)
        self._html = HtmlTextExtractor(max_chars=max_chars)

    def extract(self, path: Path) -> str:
        """Extrait le texte du fichier, quel que soit son format.

        Args:
            path: Chemin local du fichier (PDF, HTM, HTML, XBRL…).

        Returns:
            Texte extrait et tronqué.
        """
        if path.suffix.lower() in _HTML_EXTENSIONS:
            return self._html.extract(path)
        return self._pdf.extract(path)


# ---------------------------------------------------------------------------
# Extraction LLM via LangChain
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "Tu es un analyste financier expert. Analyse le texte extrait d'un rapport annuel "
    "et extrais les KPIs financiers suivants :\n\n"
    "1. **capex** : Dépenses d'investissement (CAPEX / Capital Expenditures / Investissements). "
    "Exprime en millions dans la devise du rapport. Si les données sont en milliers, convertis.\n"
    "2. **marge_brute** : (Chiffre d'affaires − Coût des ventes) / Chiffre d'affaires. "
    "Retourne un ratio décimal entre 0 et 1 (ex : 42 % → 0.42).\n"
    "3. **ratio_endettement** : Ratio dette nette / EBITDA ; ou dette totale / capitaux propres "
    "si l'autre est indisponible. Retourne un float (ex : 2.3).\n"
    "4. **devise** : Devise des montants (EUR, USD, GBP…).\n"
    "5. **annee_exercice** : Année de l'exercice fiscal (ex : 2023).\n"
    "6. **source** : Brève description de la section ou du tableau d'où proviennent les données.\n\n"
    "Si une donnée est absente ou impossible à calculer avec certitude, retourne null. "
    "Ne jamais inventer ni estimer une valeur absente du texte."
)

_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "Voici le texte extrait du rapport annuel :\n\n{text}"),
    ]
)


class LlmKpiExtractor:
    """Extrait les KPIs via OpenAI avec output structuré Pydantic.

    Utilise `ChatOpenAI.with_structured_output(KpiResult)` pour garantir
    un JSON validé contre le schéma sans post-traitement manuel.
    """

    def __init__(
        self, model: str = "gpt-4o-mini", temperature: float = 0.0
    ) -> None:
        """Args:
            model: Identifiant du modèle OpenAI (ex : "gpt-4o-mini", "gpt-4o").
            temperature: Température de sampling (0.0 = déterministe).
        """
        llm = ChatOpenAI(model=model, temperature=temperature)
        self._chain = _EXTRACTION_PROMPT | llm.with_structured_output(KpiResult)

    def extract(self, text: str) -> KpiResult:
        """Envoie le texte au LLM et retourne les KPIs validés.

        Args:
            text: Texte brut extrait du PDF.

        Returns:
            Objet KpiResult avec les champs renseignés ou None.

        Raises:
            ValueError: Si la réponse LLM est inattendue ou non parseable.
        """
        try:
            result = self._chain.invoke({"text": text})
        except Exception as exc:
            raise ValueError(f"Erreur lors de l'appel LLM : {exc}") from exc

        if not isinstance(result, KpiResult):
            raise ValueError(
                f"Réponse LLM de type inattendu (reçu : {type(result).__name__})"
            )

        logger.debug(
            "KPIs — capex=%s, marge_brute=%s, ratio_endettement=%s",
            result.capex,
            result.marge_brute,
            result.ratio_endettement,
        )
        return result


# ---------------------------------------------------------------------------
# Orchestrateur — point d'entrée du module
# ---------------------------------------------------------------------------

_EMPTY_KPIS: dict = {
    "capex": None,
    "marge_brute": None,
    "ratio_endettement": None,
    "devise": None,
    "annee_exercice": None,
    "kpi_source": None,
}


class KpiExtractorLLM:
    """Module 3 — Extrait les KPIs financiers de rapports annuels (PDF ou iXBRL).

    Stratégie par type de fichier :
    - Fichier HTML/XBRL → XbrlKpiExtractor en premier (sans LLM).
      Si le résultat est insuffisant (<2 KPIs), bascule sur le LLM.
    - Fichier PDF → TextExtractor + LLM directement.

    Example:
        >>> extractor = KpiExtractorLLM()
        >>> enriched = extractor.extract_all([
        ...     {"nom": "RTX Corporation", "chemin_pdf": "data/reports/RTX_rapport.htm"},
        ...     {"nom": "LVMH", "chemin_pdf": "data/reports/LVMH_rapport.pdf"},
        ... ])
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_pages: int = _MAX_PAGES,
        max_chars: int = _MAX_CHARS,
    ) -> None:
        """Args:
            model: Modèle OpenAI à utiliser (fallback LLM uniquement).
            temperature: Température du LLM (0.0 = déterministe).
            max_pages: Nombre maximal de pages PDF à lire en fallback.
            max_chars: Limite de caractères envoyés au LLM.
        """
        self._xbrl_extractor = XbrlKpiExtractor()
        self._text_extractor = TextExtractor(max_pages=max_pages, max_chars=max_chars)
        self._llm_extractor = LlmKpiExtractor(model=model, temperature=temperature)

    def extract_all(self, companies: list[dict]) -> list[dict]:
        """Extrait les KPIs pour chaque société disposant d'un chemin PDF.

        Args:
            companies: Sortie du Module 2 —
                [{"nom": str, "chemin_pdf": str | None, ...}]

        Returns:
            La même liste enrichie avec :
            capex, marge_brute, ratio_endettement, devise, annee_exercice, kpi_source.
            Les sociétés sans PDF ou en erreur conservent ces champs à None.
        """
        results: list[dict] = []
        total = len(companies)

        for i, company in enumerate(companies, start=1):
            name = company.get("nom", "inconnu")
            logger.info("[%d/%d] Extraction KPIs : %s", i, total, name)
            kpis = self._extract_one(name, company.get("chemin_pdf"))
            results.append({**company, **kpis})

        n_success = sum(
            1
            for c in results
            if c.get("capex") is not None or c.get("marge_brute") is not None
        )
        logger.info(
            "Bilan Module 3 : %d/%d sociétés avec KPIs extraits", n_success, total
        )
        return results

    def extract_one_company(self, company: dict) -> dict:
        """Extrait les KPIs pour une seule société.

        Args:
            company: Dictionnaire avec au moins 'nom' et 'chemin_pdf'.

        Returns:
            Le dictionnaire enrichi avec les KPIs.
        """
        name = company.get("nom", "inconnu")
        kpis = self._extract_one(name, company.get("chemin_pdf"))
        return {**company, **kpis}

    def _extract_one(self, name: str, pdf_path_str: Optional[str]) -> dict:
        """Pipeline d'extraction complet pour une société.

        Pour les fichiers iXBRL (HTML/XBRL), tente d'abord XbrlKpiExtractor.
        Bascule sur le LLM si le résultat est insuffisant ou en cas d'erreur.

        Args:
            name: Nom de la société (pour les logs).
            pdf_path_str: Chemin local du rapport (PDF, HTM, XBRL…), ou None.

        Returns:
            Dict avec les clés KPI (valeurs None en cas d'échec ou rapport absent).
        """
        if not pdf_path_str:
            logger.warning("Pas de rapport pour '%s', extraction ignorée", name)
            return dict(_EMPTY_KPIS)

        report_path = Path(pdf_path_str)

        # Tentative XBRL directe pour les fichiers HTML
        if report_path.suffix.lower() in _HTML_EXTENSIONS:
            kpi = self._try_xbrl(name, report_path)
            if kpi is not None:
                return self._kpi_to_dict(kpi)
            logger.info(
                "XBRL insuffisant pour '%s', passage au LLM", name
            )

        # Extraction texte + LLM
        try:
            text = self._text_extractor.extract(report_path)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Extraction texte échouée pour '%s' : %s", name, exc)
            return dict(_EMPTY_KPIS)

        try:
            kpi = self._llm_extractor.extract(text)
        except ValueError as exc:
            logger.warning("Extraction LLM échouée pour '%s' : %s", name, exc)
            return dict(_EMPTY_KPIS)

        return self._kpi_to_dict(kpi)

    def _try_xbrl(self, name: str, path: Path) -> Optional[KpiResult]:
        """Tente l'extraction XBRL et retourne None si insuffisante ou en erreur.

        Args:
            name: Nom de la société (pour les logs).
            path: Chemin du fichier HTML/XBRL.

        Returns:
            KpiResult si suffisant (≥2 KPIs), None sinon.
        """
        try:
            result = self._xbrl_extractor.extract(path)
        except (FileNotFoundError, ValueError) as exc:
            logger.debug("XBRL inutilisable pour '%s' : %s", name, exc)
            return None

        if self._xbrl_extractor.is_sufficient(result):
            logger.info("KPIs extraits via iXBRL pour '%s'", name)
            return result
        return None

    @staticmethod
    def _kpi_to_dict(kpi: KpiResult) -> dict:
        """Convertit un KpiResult en dictionnaire de sortie standard."""
        return {
            "capex": kpi.capex,
            "marge_brute": kpi.marge_brute,
            "ratio_endettement": kpi.ratio_endettement,
            "devise": kpi.devise,
            "annee_exercice": kpi.annee_exercice,
            "kpi_source": kpi.source,
        }
