"""Module 1 : PeerGroupFinder.

Identifie des entreprises comparables à partir d'un ticker ou d'un nom
d'entreprise, en filtrant par secteur et fourchettes de taille.

Deux sources supportées :
- Yahoo Finance (yfinance) — sociétés cotées internationales.
- Pappers API — sociétés françaises (cotées ou non).
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

import requests
import yfinance as yf

from models.peer_models import CompanyFilter, PeerCompany

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Correspondance pays → continent (marchés financiers principaux)
# ---------------------------------------------------------------------------

_COUNTRY_TO_CONTINENT: dict[str, str] = {
    # Amérique du Nord
    "United States": "North America",
    "Canada": "North America",
    "Mexico": "North America",
    # Amérique du Sud
    "Brazil": "South America",
    "Argentina": "South America",
    "Chile": "South America",
    "Colombia": "South America",
    "Peru": "South America",
    # Europe
    "France": "Europe",
    "Germany": "Europe",
    "United Kingdom": "Europe",
    "Spain": "Europe",
    "Italy": "Europe",
    "Netherlands": "Europe",
    "Belgium": "Europe",
    "Switzerland": "Europe",
    "Sweden": "Europe",
    "Norway": "Europe",
    "Denmark": "Europe",
    "Finland": "Europe",
    "Portugal": "Europe",
    "Austria": "Europe",
    "Ireland": "Europe",
    "Luxembourg": "Europe",
    "Poland": "Europe",
    "Czech Republic": "Europe",
    "Hungary": "Europe",
    "Greece": "Europe",
    "Romania": "Europe",
    "Turkey": "Europe",
    "Russia": "Europe",
    # Asie
    "China": "Asia",
    "Japan": "Asia",
    "India": "Asia",
    "South Korea": "Asia",
    "Taiwan": "Asia",
    "Hong Kong": "Asia",
    "Singapore": "Asia",
    "Indonesia": "Asia",
    "Malaysia": "Asia",
    "Thailand": "Asia",
    "Vietnam": "Asia",
    "Philippines": "Asia",
    "Pakistan": "Asia",
    "Bangladesh": "Asia",
    "Israel": "Asia",
    "Saudi Arabia": "Asia",
    "United Arab Emirates": "Asia",
    "Qatar": "Asia",
    "Kuwait": "Asia",
    # Océanie
    "Australia": "Oceania",
    "New Zealand": "Oceania",
    # Afrique
    "South Africa": "Africa",
    "Nigeria": "Africa",
    "Egypt": "Africa",
    "Kenya": "Africa",
    "Morocco": "Africa",
}


def _get_continent(country: str) -> Optional[str]:
    """Retourne le continent d'un pays, ou None si inconnu.

    Args:
        country: Nom du pays tel que retourné par yfinance (ex: "France").

    Returns:
        Nom du continent (ex: "Europe") ou None.
    """
    return _COUNTRY_TO_CONTINENT.get(country)


# ---------------------------------------------------------------------------
# Tables de conversion des tranches Pappers
# ---------------------------------------------------------------------------

# (code_tranche, borne_basse_incluse, borne_haute_incluse) en EUR
_PAPPERS_CA_TRANCHES: list[tuple[str, float, float]] = [
    ("A", 0, 899_999),
    ("B", 900_000, 3_999_999),
    ("C", 4_000_000, 9_999_999),
    ("D", 10_000_000, 19_999_999),
    ("E", 20_000_000, 49_999_999),
    ("F", 50_000_000, 99_999_999),
    ("G", 100_000_000, 199_999_999),
    ("H", 200_000_000, 1_499_999_999),
    ("I", 1_500_000_000, float("inf")),
]

# Codes INSEE des tranches d'effectif
_PAPPERS_EFFECTIF_TRANCHES: list[tuple[str, int, int]] = [
    ("0", 0, 0),
    ("1", 1, 2),
    ("2", 3, 5),
    ("3", 6, 9),
    ("11", 10, 19),
    ("12", 20, 49),
    ("21", 50, 99),
    ("22", 100, 199),
    ("31", 200, 249),
    ("32", 250, 499),
    ("41", 500, 999),
    ("42", 1_000, 1_999),
    ("51", 2_000, 4_999),
    ("52", 5_000, 9_999),
    ("53", 10_000, 999_999_999),
]


def _find_pappers_tranche(value: float, tranches: list[tuple]) -> str:
    """Retourne le code de tranche Pappers correspondant à une valeur.

    Args:
        value: Valeur numérique (CA en EUR ou effectif).
        tranches: Table de conversion [(code, borne_basse, borne_haute), ...].

    Returns:
        Code de tranche (ex: "E" pour un CA entre 20M et 50M€).
    """
    for code, low, high in tranches:
        if low <= value <= high:
            return code
    return tranches[-1][0]


# ---------------------------------------------------------------------------
# Classe de base
# ---------------------------------------------------------------------------


class BasePeerFinder(ABC):
    """Interface abstraite pour les stratégies de recherche de comparables."""

    @abstractmethod
    def find_peers(
        self, identifier: str, filters: Optional[CompanyFilter] = None
    ) -> list[PeerCompany]:
        """Retourne une liste d'entreprises comparables.

        Args:
            identifier: Ticker boursier ou nom d'entreprise selon la stratégie.
            filters: Critères de secteur et de taille. Si None, aucun filtre appliqué.

        Returns:
            Liste d'objets PeerCompany.
        """


# ---------------------------------------------------------------------------
# Stratégie Yahoo Finance
# ---------------------------------------------------------------------------


class YFinancePeerFinder(BasePeerFinder):
    """Recherche de comparables cotés via les données sectorielles Yahoo Finance.

    Nécessite yfinance >= 0.2.44 pour l'accès aux objets yf.Industry / yf.Sector.
    """

    _MAX_RESULTS: int = 30
    _THROTTLE_SEC: float = 0.25  # Délai entre appels yfinance individuels

    def find_peers(
        self, ticker: str, filters: Optional[CompanyFilter] = None
    ) -> list[PeerCompany]:
        """Identifie les pairs d'une société cotée.

        Args:
            ticker: Symbole boursier de référence (ex: "MC.PA", "AAPL").
            filters: Secteur cible et fourchettes optionnelles de CA / effectifs / market cap.
                Si None, aucun filtre de taille ou de zone géographique n'est appliqué.

        Returns:
            Liste de PeerCompany, sans le ticker de référence.

        Raises:
            ValueError: Si le ticker est invalide ou si yfinance ne retourne
                pas de clé sectorielle (industryKey / sectorKey).
        """
        filters = filters or CompanyFilter()
        logger.info("Récupération des informations pour le ticker '%s'", ticker)
        try:
            ref_info: dict = yf.Ticker(ticker).info
        except Exception as exc:
            logger.error("Échec de la requête yfinance pour '%s' : %s", ticker, exc)
            raise

        industry_key: Optional[str] = ref_info.get("industryKey")
        sector_key: Optional[str] = ref_info.get("sectorKey")

        if not industry_key and not sector_key:
            raise ValueError(
                f"Impossible de déterminer le secteur pour '{ticker}'. "
                "Vérifiez le ticker et que yfinance >= 0.2.44 est installé."
            )

        logger.info(
            "Référence — secteur : '%s', industrie : '%s'",
            ref_info.get("sector"),
            ref_info.get("industry"),
        )

        candidates = self._get_industry_candidates(industry_key, sector_key)
        logger.info("%d candidats trouvés dans le même secteur / industrie", len(candidates))

        needs_enrichment = any(
            [
                filters.min_ca,
                filters.max_ca,
                filters.min_effectifs,
                filters.max_effectifs,
                filters.pays,
                filters.continent,
                filters.min_market_cap,
                filters.max_market_cap,
            ]
        )
        if needs_enrichment:
            candidates = self._enrich_and_filter(
                candidates, filters, exclude_ticker=ticker.upper()
            )
        else:
            candidates = [
                c for c in candidates
                if (c.get("symbol") or "").upper() != ticker.upper()
            ]

        candidates = candidates[: self._MAX_RESULTS]
        logger.info("%d comparables retenus après filtrage", len(candidates))

        return [
            PeerCompany(
                nom=c.get("displayName") or c.get("shortName") or c.get("symbol", ""),
                ticker=c.get("symbol"),
                url_investisseur=f"https://finance.yahoo.com/quote/{c['symbol']}"
                if c.get("symbol")
                else None,
            )
            for c in candidates
        ]

    def _get_industry_candidates(
        self,
        industry_key: Optional[str],
        sector_key: Optional[str],
    ) -> list[dict]:
        """Retourne les top sociétés d'une industrie ou d'un secteur.

        Tente d'abord le niveau industrie (plus précis), puis le secteur en fallback.

        Args:
            industry_key: Clé industrie yfinance (ex: "software-application").
            sector_key: Clé secteur yfinance (ex: "technology").

        Returns:
            Liste de dicts avec au minimum la clé 'symbol'.
        """
        import pandas as pd

        if industry_key:
            try:
                df: pd.DataFrame = yf.Industry(industry_key).top_companies
                df = pd.concat([df, yf.Industry(industry_key).top_growth_companies], )
                df = pd.concat([df, yf.Industry(industry_key).top_performing_companies])
                df.drop_duplicates(inplace=True)
                if df is not None and not df.empty:
                    return df.reset_index().to_dict("records")
            except Exception as exc:
                logger.warning(
                    "Échec du lookup industrie '%s' : %s", industry_key, exc
                )

        if sector_key:
            try:
                df = yf.Sector(sector_key).top_companies
                if df is not None and not df.empty:
                    return df.reset_index().to_dict("records")
            except Exception as exc:
                logger.warning(
                    "Échec du lookup secteur '%s' : %s", sector_key, exc
                )

        logger.error("Aucune donnée sectorielle disponible via yfinance.")
        return []

    def _enrich_and_filter(
        self,
        candidates: list[dict],
        filters: CompanyFilter,
        exclude_ticker: str,
    ) -> list[dict]:
        """Enrichit chaque candidat avec son CA et ses effectifs, puis filtre.

        Effectue un appel yfinance individuel par société — limité en débit
        via _THROTTLE_SEC pour éviter les blocages Yahoo Finance.

        Args:
            candidates: Sociétés issues de _get_industry_candidates.
            filters: Fourchettes de CA et d'effectifs à appliquer.
            exclude_ticker: Ticker de référence à exclure du résultat.

        Returns:
            Candidats enrichis passant tous les filtres de taille.
        """
        filtered: list[dict] = []
        for candidate in candidates:
            symbol: str = (candidate.get("symbol") or candidate.get("Symbol") or "").strip()
            if not symbol or symbol.upper() == exclude_ticker:
                continue

            try:
                info: dict = yf.Ticker(symbol).info
            except Exception as exc:
                logger.warning("Impossible d'enrichir '%s' : %s", symbol, exc)
                continue

            revenue: Optional[float] = info.get("totalRevenue")
            employees: Optional[int] = info.get("fullTimeEmployees")
            country: Optional[str] = info.get("country")
            marketCap: Optional[float] = info.get("marketCap")

            if filters.min_ca is not None and revenue is not None:
                if revenue < filters.min_ca:
                    continue
            if filters.max_ca is not None and revenue is not None:
                if revenue > filters.max_ca:
                    continue
            if filters.min_effectifs is not None and employees is not None:
                if employees < filters.min_effectifs:
                    continue
            if filters.max_effectifs is not None and employees is not None:
                if employees > filters.max_effectifs:
                    continue
            if filters.min_market_cap is not None and marketCap is not None:
                if marketCap < filters.min_market_cap:
                    continue
            if filters.max_market_cap is not None and marketCap is not None:
                if marketCap > filters.max_market_cap:
                    continue

            if filters.pays is not None:
                if country is None or country.lower() != filters.pays.lower():
                    logger.debug(
                        "Exclure '%s' — pays '%s' ≠ '%s'", symbol, country, filters.pays
                    )
                    continue

            if filters.continent is not None:
                candidate_continent = _get_continent(country) if country else None
                if (
                    candidate_continent is None
                    or candidate_continent.lower() != filters.continent.lower()
                ):
                    logger.debug(
                        "Exclure '%s' — continent '%s' ≠ '%s'",
                        symbol,
                        candidate_continent,
                        filters.continent,
                    )
                    continue

            filtered.append(
                {
                    **candidate,
                    "shortName": info.get("shortName", symbol),
                    "symbol": symbol,
                    "country": country,
                }
            )
            time.sleep(self._THROTTLE_SEC)

        return filtered


# ---------------------------------------------------------------------------
# Stratégie Pappers (sociétés françaises)
# ---------------------------------------------------------------------------


class PappersPeerFinder(BasePeerFinder):
    """Recherche de comparables français via l'API Pappers.

    Référence API : https://www.pappers.fr/api/documentation
    """

    _BASE_URL = "https://api.pappers.fr/v2"
    _DEFAULT_PAGE_SIZE = 50

    def __init__(self, api_key: str) -> None:
        """Initialise le client Pappers.

        Args:
            api_key: Jeton d'authentification Pappers.
        """
        self._api_key = api_key

    def find_peers(
        self, company_name: str, filters: Optional[CompanyFilter] = None
    ) -> list[PeerCompany]:
        """Recherche des entreprises françaises comparables via Pappers.

        Args:
            company_name: Nom de la société de référence (pour le log uniquement).
            filters: `secteur` doit être un code NAF/APE (ex: "4120A").
                Les fourchettes de CA et d'effectifs sont converties en
                codes de tranches Pappers automatiquement. Si None, aucun filtre appliqué.

        Returns:
            Liste de PeerCompany avec des URLs de profil Pappers.

        Raises:
            requests.HTTPError: En cas d'erreur HTTP de l'API Pappers.
            requests.RequestException: En cas d'échec réseau.
        """
        filters = filters or CompanyFilter()
        if filters.pays is not None and filters.pays.lower() not in ("france", "fr"):
            logger.warning(
                "PappersPeerFinder ne couvre que la France ; "
                "le filtre pays='%s' sera ignoré.",
                filters.pays,
            )
        if filters.continent is not None and filters.continent.lower() != "europe":
            logger.warning(
                "PappersPeerFinder ne couvre que la France (Europe) ; "
                "le filtre continent='%s' sera ignoré.",
                filters.continent,
            )

        logger.info(
            "Recherche Pappers pour '%s' (code_naf='%s')",
            company_name,
            filters.secteur,
        )

        params: dict[str, str | int] = {
            "api_token": self._api_key,
            "code_naf": filters.secteur,
            "par_page": self._DEFAULT_PAGE_SIZE,
            "precision": "standard",
        }

        if filters.min_ca is not None:
            params["tranche_ca_min"] = _find_pappers_tranche(
                filters.min_ca, _PAPPERS_CA_TRANCHES
            )
        if filters.max_ca is not None:
            params["tranche_ca_max"] = _find_pappers_tranche(
                filters.max_ca, _PAPPERS_CA_TRANCHES
            )
        if filters.min_effectifs is not None:
            params["tranche_effectif_min"] = _find_pappers_tranche(
                filters.min_effectifs, _PAPPERS_EFFECTIF_TRANCHES
            )
        if filters.max_effectifs is not None:
            params["tranche_effectif_max"] = _find_pappers_tranche(
                filters.max_effectifs, _PAPPERS_EFFECTIF_TRANCHES
            )

        try:
            response = requests.get(
                f"{self._BASE_URL}/recherche",
                params=params,
                timeout=15,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            logger.error(
                "Pappers API — erreur HTTP %s : %s",
                exc.response.status_code,
                exc,
            )
            raise
        except requests.RequestException as exc:
            logger.error("Pappers API — échec réseau : %s", exc)
            raise

        data = response.json()
        companies: list[dict] = data.get("resultats", [])
        logger.info(
            "Pappers : %d résultats pour code_naf='%s'",
            len(companies),
            filters.secteur,
        )

        return [
            PeerCompany(
                nom=c.get("nom_entreprise", ""),
                ticker=None,
                url_investisseur=(
                    f"https://www.pappers.fr/entreprise/{c['siren']}"
                    if c.get("siren")
                    else None
                ),
            )
            for c in companies
            if c.get("nom_entreprise")
        ]


# ---------------------------------------------------------------------------
# Orchestrateur principal
# ---------------------------------------------------------------------------


class PeerGroupFinder:
    """Point d'entrée du Module 1 — route la recherche vers la bonne source.

    Example:
        >>> finder = PeerGroupFinder(pappers_api_key="votre_clé")

        >>> # Société cotée (Yahoo Finance), restreint à l'Europe
        >>> peers = finder.find_peers(
        ...     "MC.PA",
        ...     CompanyFilter(secteur="Consumer Cyclical", continent="Europe"),
        ... )

        >>> # Restreindre à un pays précis
        >>> peers = finder.find_peers(
        ...     "AAPL",
        ...     CompanyFilter(secteur="Technology", pays="United States"),
        ... )

        >>> # Société française non cotée (Pappers)
        >>> peers = finder.find_peers(
        ...     "SUD EST BETON",
        ...     CompanyFilter(secteur="4120A", min_ca=5_000_000, max_ca=50_000_000),
        ...     use_pappers=True,
        ... )
        >>> # [{"nom": "...", "ticker": None, "url_investisseur": "https://..."}]
    """

    def __init__(self, pappers_api_key: Optional[str] = None) -> None:
        """Initialise l'orchestrateur.

        Args:
            pappers_api_key: Clé API Pappers optionnelle. Requise si
                `use_pappers=True` est passé à find_peers().
        """
        self._yfinance_finder = YFinancePeerFinder()
        self._pappers_finder = (
            PappersPeerFinder(pappers_api_key) if pappers_api_key else None
        )

    def find_peers(
        self,
        identifier: str,
        filters: Optional[CompanyFilter] = None,
        use_pappers: bool = False,
    ) -> list[dict]:
        """Identifie les entreprises comparables et retourne un résultat sérialisable.

        Args:
            identifier: Ticker boursier (Yahoo Finance) ou nom d'entreprise (Pappers).
            filters: CompanyFilter avec secteur et fourchettes de taille optionnelles.
                Si None, aucun filtre n'est appliqué (retourne tous les pairs du secteur).
            use_pappers: Si True, route vers l'API Pappers (sociétés françaises).

        Returns:
            Liste de dicts :
            [{"nom": str, "ticker": str | None, "url_investisseur": str | None}]

        Raises:
            ValueError: Si use_pappers=True mais aucune clé API fournie à l'init.
        """
        if use_pappers:
            if self._pappers_finder is None:
                raise ValueError(
                    "Clé API Pappers requise. "
                    "Passez pappers_api_key= au constructeur PeerGroupFinder()."
                )
            peers = self._pappers_finder.find_peers(identifier, filters)
        else:
            peers = self._yfinance_finder.find_peers(identifier, filters)

        return [p.model_dump() for p in peers]
