"""Module 2 : ReportFetcher

Localise et télécharge le dernier rapport annuel de chaque entreprise comparable.

Stratégie de récupération (par ordre de priorité) :
1. SEC EDGAR API  — tickers US purs (sans suffixe de place, ex : "AAPL").
2. Scraping IR    — page investisseurs ou site officiel pour toutes les autres sociétés.
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
import yfinance as yf
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_DEFAULT_REPORTS_DIR = Path("data/reports")

_EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_EDGAR_FILING_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
)
# La SEC exige un User-Agent identifiant l'application et un email de contact
_EDGAR_HEADERS = {
    "User-Agent": "benchmark-tool research@example.com",
    "Accept-Encoding": "gzip, deflate",
}
_SCRAPING_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; benchmark-tool/1.0)"}

_ANNUAL_REPORT_KEYWORDS = [
    "annual report", "rapport annuel", "10-k", "annual-report",
    "rapport-annuel", "annual_report", "yearly report",
    "résultats annuels", "annual results",
]

# Chemins courants des sections investisseurs sur les sites d'entreprises
_IR_URL_PATHS = [
    "/investors/being-a-lvmh-shareholder","/investor-relations", "/investors", "/investisseurs",
    "/ir", "/finance", "/investor-centre", "/relations-investisseurs",
    "/annual-report", "/rapport-annuel",
]


class BaseReportFetcher(ABC):
    """Interface abstraite pour les stratégies de récupération de rapports."""

    def __init__(self, reports_dir: Path) -> None:
        """Args:
            reports_dir: Dossier de destination des PDF téléchargés.
        """
        self._reports_dir = reports_dir

    @abstractmethod
    def fetch_report(self, company: dict) -> Optional[Path]:
        """Télécharge le rapport annuel d'une société.

        Args:
            company: Dictionnaire avec les clés 'nom', 'ticker', 'url_investisseur'.

        Returns:
            Chemin local du PDF téléchargé, ou None si non trouvé.
        """


class EdgarReportFetcher(BaseReportFetcher):
    """Récupère les dépôts 10-K officiels depuis l'API SEC EDGAR.

    Respecte la limite de 10 requêtes/s recommandée par la SEC via _THROTTLE_SEC.
    La table ticker → CIK est chargée une seule fois depuis company_tickers.json.
    """

    _THROTTLE_SEC = 0.12

    def __init__(self, reports_dir: Path) -> None:
        super().__init__(reports_dir)
        self._cik_map: dict[str, str] = {}  # ticker.upper() → CIK 10 chiffres

    def fetch_report(self, company: dict) -> Optional[Path]:
        """Récupère le dernier 10-K d'une société cotée US depuis EDGAR.

        Args:
            company: Doit contenir 'ticker' sans suffixe de place (ex: "AAPL").

        Returns:
            Chemin local du PDF, ou None si absent d'EDGAR ou erreur réseau.
        """
        ticker = company.get("ticker", "")
        if not ticker:
            return None

        try:
            cik = self._resolve_cik(ticker)
        except (ValueError, requests.RequestException) as exc:
            logger.warning("CIK introuvable pour '%s' : %s", ticker, exc)
            return None

        try:
            filing = self._get_latest_10k(cik)
        except requests.RequestException as exc:
            logger.warning("Erreur EDGAR pour '%s' (CIK=%s) : %s", ticker, cik, exc)
            return None

        if not filing:
            logger.info("Aucun 10-K dans EDGAR pour '%s'", ticker)
            return None

        url = _EDGAR_FILING_URL.format(
            cik=str(int(cik)),
            accession=filing["accessionNumber"],
            document=filing["primaryDocument"],
        )
        try:
            path = _download_file(
                url=url,
                company_name=company.get("nom", ticker),
                dest_dir=self._reports_dir,
                headers=_EDGAR_HEADERS,
            )
            logger.info("10-K téléchargé (%s) → %s", filing["filingDate"], path)
            return path
        except (requests.RequestException, OSError) as exc:
            logger.warning("Échec du téléchargement EDGAR pour '%s' : %s", ticker, exc)
            return None

    def _resolve_cik(self, ticker: str) -> str:
        """Retourne le CIK EDGAR (10 chiffres) pour un ticker donné.

        Charge la table de correspondance depuis EDGAR au premier appel.

        Args:
            ticker: Symbole boursier (ex: "AAPL").

        Returns:
            CIK formaté sur 10 chiffres (ex: "0000320193").

        Raises:
            ValueError: Si le ticker est absent du registre EDGAR.
            requests.RequestException: En cas d'échec réseau.
        """
        key = ticker.upper()
        if key not in self._cik_map:
            self._load_cik_map()
        if key not in self._cik_map:
            raise ValueError(f"Ticker '{ticker}' absent du registre EDGAR")
        return self._cik_map[key]

    def _load_cik_map(self) -> None:
        """Charge la table ticker → CIK depuis company_tickers.json (EDGAR)."""
        logger.debug("Chargement de la table CIK depuis EDGAR")
        resp = requests.get(_EDGAR_TICKERS_URL, headers=_EDGAR_HEADERS, timeout=20)
        resp.raise_for_status()
        for entry in resp.json().values():
            self._cik_map[entry["ticker"].upper()] = str(entry["cik_str"]).zfill(10)
        time.sleep(self._THROTTLE_SEC)

    def _get_latest_10k(self, cik: str) -> Optional[dict]:
        """Retourne les métadonnées du dépôt 10-K le plus récent pour un CIK.

        Args:
            cik: CIK EDGAR formaté sur 10 chiffres.

        Returns:
            Dict avec 'accessionNumber' (sans tirets), 'primaryDocument',
            'filingDate', ou None si aucun 10-K trouvé.
        """
        resp = requests.get(
            _EDGAR_SUBMISSIONS_URL.format(cik=cik),
            headers=_EDGAR_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        time.sleep(self._THROTTLE_SEC)

        recent = resp.json().get("filings", {}).get("recent", {})
        for idx, form in enumerate(recent.get("form", [])):
            if form == "10-K":
                return {
                    "accessionNumber": recent["accessionNumber"][idx].replace("-", ""),
                    "primaryDocument": recent["primaryDocument"][idx],
                    "filingDate": recent["filingDate"][idx],
                }
        return None


class WebScrapingReportFetcher(BaseReportFetcher):
    """Fallback : scrape la page investisseurs pour localiser le PDF du rapport annuel.

    Pour les sociétés avec un ticker, tente d'abord de récupérer l'URL du site
    officiel via yfinance, puis cherche une page /investor-relations courante.
    """

    def fetch_report(self, company: dict) -> Optional[Path]:
        """Scrape une page web pour trouver et télécharger le rapport annuel.

        Args:
            company: Dict avec 'url_investisseur' et/ou 'ticker'.

        Returns:
            Chemin local du PDF, ou None si non trouvé.
        """
        ir_url = self._resolve_ir_url(company)
        if not ir_url:
            logger.warning(
                "Aucune URL disponible pour scraper '%s'", company.get("nom")
            )
            return None

        pdf_url = self._scrape_for_pdf(ir_url)
        if not pdf_url:
            logger.warning("Aucun lien PDF trouvé sur '%s'", ir_url)
            return None

        try:
            return _download_file(
                url=pdf_url,
                company_name=company.get("nom", ""),
                dest_dir=self._reports_dir,
                headers=_SCRAPING_HEADERS,
            )
        except (requests.RequestException, OSError) as exc:
            logger.warning("Échec du téléchargement depuis '%s' : %s", pdf_url, exc)
            return None

    def _resolve_ir_url(self, company: dict) -> Optional[str]:
        """Détermine l'URL investisseurs à scraper.

        Ordre de priorité :
        1. Site officiel yfinance → page /investor-relations si trouvée.
        2. url_investisseur fournie par le Module 1.

        Args:
            company: Dict avec 'ticker' et/ou 'url_investisseur'.

        Returns:
            URL la plus pertinente à scraper, ou None.
        """
        ticker = company.get("ticker")
        if ticker:
            try:
                website = yf.Ticker(ticker).info.get("website")
                if website:
                    ir_page = self._find_ir_page(website)
                    return ir_page or website
            except Exception as exc:
                logger.debug(
                    "Impossible de récupérer le site de '%s' via yfinance : %s",
                    ticker, exc,
                )

        return company.get("url_investisseur")

    def _find_ir_page(self, website: str) -> Optional[str]:
        """Cherche la section investisseurs sur le site officiel d'une société.

        Vérifie l'existence des chemins courants (_IR_URL_PATHS) via HEAD.

        Args:
            website: URL racine du site (ex: "https://www.apple.com").

        Returns:
            URL de la première page IR accessible (HTTP 200), ou None.
        """
        base = website.rstrip("/")
        for path in _IR_URL_PATHS:
            url = f"{base}{path}"
            try:
                resp = requests.head(
                    url,
                    headers=_SCRAPING_HEADERS,
                    timeout=8,
                    allow_redirects=True,
                )
                if resp.status_code == 200:
                    logger.debug("Page IR trouvée : %s", url)
                    return url
            except requests.RequestException:
                continue
        return None

    def _scrape_for_pdf(self, url: str) -> Optional[str]:
        """Analyse une page HTML et retourne le lien le plus pertinent vers un rapport annuel.

        Chaque lien <a> reçoit un score basé sur les mots-clés présents dans
        son href et son texte ancre, avec un bonus de 5 points pour les .pdf directs.

        Args:
            url: URL de la page à analyser.

        Returns:
            URL absolue du lien avec le score le plus élevé, ou None.
        """
        try:
            resp = requests.get(
                url, headers=_SCRAPING_HEADERS, timeout=15, allow_redirects=True
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Impossible de scraper '%s' : %s", url, exc)
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        scored: list[tuple[int, str]] = []
        for tag in soup.find_all("a", href=True):
            href: str = tag["href"].strip()
            anchor: str = tag.get_text(" ", strip=True).lower()
            href_lower = href.lower()

            score = 0
            for kw in _ANNUAL_REPORT_KEYWORDS:
                if kw in href_lower:
                    score += 2
                if kw in anchor:
                    score += 1

            # if score == 0:
            #     continue

            if href_lower.endswith(".pdf"):
                score += 5

            scored.append((score, urljoin(url, href)))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_url = scored[0]
        logger.debug("Lien rapport retenu (score=%d) : %s", best_score, best_url)
        return best_url


class ReportFetcher:
    """Orchestrateur du Module 2 — récupère les rapports pour toute la liste de comparables.

    Pour chaque société :
    - Ticker US pur (sans ".") → EDGAR en premier, puis scraping en fallback.
    - Ticker étranger ou pas de ticker → scraping directement.

    Example:
        >>> fetcher = ReportFetcher()
        >>> enriched = fetcher.fetch_all([
        ...     {"nom": "Apple Inc.", "ticker": "AAPL", "url_investisseur": "..."},
        ...     {"nom": "SUD EST BETON", "ticker": None, "url_investisseur": "https://pappers.fr/..."},
        ... ])
        >>> # [{"nom": "...", ..., "chemin_pdf": "data/reports/Apple_Inc._rapport.pdf"}]
    """

    def __init__(self, reports_dir: Path = _DEFAULT_REPORTS_DIR) -> None:
        """Initialise les stratégies et crée le dossier de destination si nécessaire.

        Args:
            reports_dir: Chemin du dossier local où sauvegarder les PDF.
        """
        self._reports_dir = reports_dir
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        self._edgar = EdgarReportFetcher(reports_dir)
        self._scraping = WebScrapingReportFetcher(reports_dir)

    def fetch_all(self, companies: list[dict]) -> list[dict]:
        """Télécharge les rapports et enrichit la liste avec le chemin PDF local.

        Args:
            companies: Sortie du Module 1 —
                [{"nom": str, "ticker": str | None, "url_investisseur": str | None}]

        Returns:
            La même liste enrichie avec "chemin_pdf" (str) ou None :
            [{"nom": "...", "ticker": "...", "url_investisseur": "...",
              "chemin_pdf": "data/reports/Apple_Inc._rapport.pdf"}]
        """
        enriched: list[dict] = []
        total = len(companies)

        for i, company in enumerate(companies, start=1):
            name = company.get("nom", "inconnu")
            logger.info("[%d/%d] Récupération rapport : %s", i, total, name)

            pdf_path = self._fetch_one(company)
            enriched.append({**company, "chemin_pdf": str(pdf_path) if pdf_path else None})

        found = sum(1 for c in enriched if c["chemin_pdf"])
        logger.info(
            "Bilan Module 2 : %d/%d rapports téléchargés dans '%s'",
            found, total, self._reports_dir,
        )
        return enriched

    def _fetch_one(self, company: dict) -> Optional[Path]:
        """Tente EDGAR puis scraping web pour une société.

        EDGAR est réservé aux tickers US purs (sans suffixe de place boursière).

        Args:
            company: Dictionnaire société issu du Module 1.

        Returns:
            Chemin local du PDF, ou None.
        """
        ticker = company.get("ticker") or ""
        is_us_ticker = bool(ticker) and "." not in ticker

        if is_us_ticker:
            path = self._edgar.fetch_report(company)
            if path:
                return path
            logger.info(
                "EDGAR sans résultat pour '%s', passage au scraping", ticker
            )

        return self._scraping.fetch_report(company)


# ---------------------------------------------------------------------------
# Utilitaire de téléchargement partagé
# ---------------------------------------------------------------------------


def _sanitize_filename(name: str) -> str:
    """Normalise une chaîne en nom de fichier valide (alphanum, tirets, underscores)."""
    name = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:80]


def _download_file(
    url: str,
    company_name: str,
    dest_dir: Path,
    headers: Optional[dict] = None,
    chunk_size: int = 8192,
) -> Path:
    """Télécharge un fichier en streaming et le sauvegarde dans dest_dir.

    Le nom de fichier est déduit du header Content-Disposition ou du chemin
    de l'URL, puis préfixé par le nom normalisé de la société.

    Args:
        url: URL source du fichier.
        company_name: Nom de la société (préfixe du fichier local).
        dest_dir: Dossier de destination (doit exister).
        headers: Headers HTTP additionnels.
        chunk_size: Taille des blocs de lecture pour le streaming.

    Returns:
        Chemin absolu du fichier sauvegardé.

    Raises:
        requests.HTTPError: Si la réponse HTTP est une erreur (4xx/5xx).
        OSError: Si l'écriture sur disque échoue.
    """
    resp = requests.get(url, headers=headers, timeout=30, stream=True)
    resp.raise_for_status()

    # Déduction du nom de fichier
    content_disposition = resp.headers.get("Content-Disposition", "")
    cd_match = re.search(r'filename[^;=\n]*=["\']?([^;\n"\']+)', content_disposition)
    if cd_match:
        base_filename = cd_match.group(1).strip()
    else:
        url_path = urlparse(url).path
        base_filename = url_path.rstrip("/").split("/")[-1] or "rapport.pdf"

    safe_prefix = _sanitize_filename(company_name)

    # Déduire l'extension réelle depuis le Content-Type de la réponse
    content_type = resp.headers.get("Content-Type", "").lower()
    if "pdf" in content_type:
        real_ext = ".pdf"
    elif "html" in content_type or "xhtml" in content_type:
        real_ext = ".htm"
    else:
        url_ext = Path(urlparse(url).path).suffix.lower()
        real_ext = url_ext if url_ext in (".pdf", ".htm", ".html", ".xbrl") else ".htm"

    base_stem = Path(base_filename).stem
    if safe_prefix:
        base_filename = f"{safe_prefix}_rapport{real_ext}"
    else:
        base_filename = f"{base_stem}{real_ext}"

    dest = dest_dir / base_filename
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)

    logger.debug("Fichier écrit : %s (%d octets)", dest, dest.stat().st_size)
    return dest
