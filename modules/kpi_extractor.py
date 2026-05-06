"""Module 3 : KpiExtractorLLM

Extrait les KPIs financiers clés d'un rapport annuel PDF via LangChain et un LLM.

Pipeline :
1. Extraction du texte brut avec pdfplumber (sections financières en priorité,
   fallback sur les N premières pages).
2. Envoi du texte au LLM (OpenAI via LangChain) avec un prompt structuré.
3. Validation de la réponse via Pydantic → retourne un objet KpiResult typé.
"""

import logging
from pathlib import Path
from typing import Optional

import pdfplumber
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
    """Module 3 — Extrait les KPIs financiers de rapports PDF via LLM.

    Pour chaque société disposant d'un PDF local, le module :
    1. Extrait le texte via pdfplumber (sections financières en priorité).
    2. Envoie le texte au LLM et récupère les KPIs structurés.
    3. Enrichit le dictionnaire société avec les KPIs.

    Example:
        >>> extractor = KpiExtractorLLM()
        >>> enriched = extractor.extract_all([
        ...     {"nom": "Apple Inc.", "chemin_pdf": "data/reports/Apple_rapport.pdf"},
        ... ])
        >>> # [{"nom": "Apple Inc.", "capex": 11455.0, "marge_brute": 0.43, ...}]
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_pages: int = _MAX_PAGES,
        max_chars: int = _MAX_CHARS,
    ) -> None:
        """Args:
            model: Modèle OpenAI à utiliser.
            temperature: Température du LLM (0.0 = déterministe).
            max_pages: Nombre maximal de pages PDF à lire en fallback.
            max_chars: Limite de caractères envoyés au LLM.
        """
        self._pdf_extractor = PdfTextExtractor(
            max_pages=max_pages, max_chars=max_chars
        )
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

        Args:
            name: Nom de la société (pour les logs d'erreur).
            pdf_path_str: Chemin local du PDF, ou None.

        Returns:
            Dict avec les clés KPI (valeurs None en cas d'échec ou PDF absent).
        """
        if not pdf_path_str:
            logger.warning("Pas de PDF pour '%s', extraction ignorée", name)
            return dict(_EMPTY_KPIS)

        pdf_path = Path(pdf_path_str)

        try:
            text = self._pdf_extractor.extract(pdf_path)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Extraction texte échouée pour '%s' : %s", name, exc)
            return dict(_EMPTY_KPIS)

        try:
            kpi: KpiResult = self._llm_extractor.extract(text)
        except ValueError as exc:
            logger.warning("Extraction LLM échouée pour '%s' : %s", name, exc)
            return dict(_EMPTY_KPIS)

        return {
            "capex": kpi.capex,
            "marge_brute": kpi.marge_brute,
            "ratio_endettement": kpi.ratio_endettement,
            "devise": kpi.devise,
            "annee_exercice": kpi.annee_exercice,
            "kpi_source": kpi.source,
        }
