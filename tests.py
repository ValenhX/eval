from modules.peer_group_finder import PeerGroupFinder
from models.peer_models import CompanyFilter
from modules.report_fetcher import ReportFetcher
from modules.kpi_extractor import KpiExtractorLLM
from modules.excel_exporter import ExcelExporter

def test_module_1_yahoo():

    finder = PeerGroupFinder(pappers_api_key="votre_clé")  # clé optionnelle

    # Coté (Yahoo Finance)
    peers = finder.find_peers("HO.PA", CompanyFilter(secteur="Technology", continent="Europe"))

    print(peers[:5])
    return peers

def test_module_1_pappers():

    finder = PeerGroupFinder(pappers_api_key="votre_clé")  # clé optionnelle

    # Français non coté (Pappers) — secteur = code NAF
    peers = finder.find_peers(
        "SUD EST BETON",
        CompanyFilter(secteur="4120A", min_ca=5_000_000, max_ca=50_000_000),
        use_pappers=True,
    )
    print(peers)
    return peers

def test_module_2(peers):

    fetcher = ReportFetcher()
    enriched = fetcher.fetch_all(peers)
    print(enriched)


def test_module_3(enriched_companies):
    extractor = KpiExtractorLLM(model="gpt-4o-mini")
    kpi_data = extractor.extract_all(enriched_companies)
    for company in kpi_data:
        print(
            f"{company['nom']} — capex={company.get('capex')}, "
            f"marge_brute={company.get('marge_brute')}, "
            f"ratio_endettement={company.get('ratio_endettement')}"
        )
    return kpi_data


def test_module_4(kpi_data):
    exporter = ExcelExporter()
    path = exporter.export(kpi_data)
    print(f"Excel généré : {path}")
    return path


test_data=[{'nom': 'TPR', 'ticker': 'TPR', 'url_investisseur': 'https://finance.yahoo.com/quote/TPR'}, {'nom': 'LVMH', 'ticker': 'MC.PA', 'url_investisseur': 'https://finance.yahoo.com/quote/MC.PA'}]

test_module_1_yahoo()
#test_module_2(test_data)