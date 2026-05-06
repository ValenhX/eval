from modules.peer_group_finder import PeerGroupFinder
from models.peer_models import CompanyFilter
from modules.report_fetcher import ReportFetcher
from modules.kpi_extractor import KpiExtractorLLM
from modules.excel_exporter import ExcelExporter

def test_module_1_yahoo():
    finder = PeerGroupFinder()  # clé optionnelle
    # Coté (Yahoo Finance)
    peers = finder.find_peers("NVTS")
    return peers

def test_module_1_pappers():
    finder = PeerGroupFinder(pappers_api_key="votre_clé")  # clé optionnelle
    # Français non coté (Pappers) — secteur = code NAF
    peers = finder.find_peers(
        "SUD EST BETON",
        CompanyFilter(secteur="4120A", min_ca=5_000_000, max_ca=50_000_000),
        use_pappers=True,
    )
    return peers

def test_module_2(peers):
    fetcher = ReportFetcher()
    enriched = fetcher.fetch_all(peers)
    return enriched


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


def test_module_4(kpi_data, initial_company=None):
    exporter = ExcelExporter()
    path = exporter.export(kpi_data, initial_company=initial_company)
    print(f"Excel généré : {path}")
    return path


def stack1():
    company_init={'nom': 'Thales', 'ticker': 'HO.PA', 'url_investisseur': 'https://finance.yahoo.com/quote/HO.PA'}

    #peers = test_module_1_yahoo()
    #enriched_companies = test_module_2(peers)
    enriched_companies=[{'nom': 'BWX Technologies, Inc.', 'ticker': 'BWXT', 'url_investisseur': 'https://finance.yahoo.com/quote/BWXT', 'chemin_pdf': 'data\\reports\\BWX_Technologies_Inc_rapport.htm'}, {'nom': 'Textron Inc.', 'ticker': 'TXT', 'url_investisseur': 'https://finance.yahoo.com/quote/TXT', 'chemin_pdf': 'data\\reports\\Textron_Inc_rapport.htm'}, {'nom': 'Arxis, Inc.', 'ticker': 'ARXS', 'url_investisseur': 'https://finance.yahoo.com/quote/ARXS', 'chemin_pdf': 'data\\reports\\Arxis_Inc_rapport.htm'}, {'nom': 'Planet Labs PBC', 'ticker': 'PL', 'url_investisseur': 'https://finance.yahoo.com/quote/PL', 'chemin_pdf': 'data\\reports\\Planet_Labs_PBC_rapport.htm'}, {'nom': 'Huntington Ingalls Industries, ', 'ticker': 'HII', 'url_investisseur': 'https://finance.yahoo.com/quote/HII', 'chemin_pdf': 'data\\reports\\Huntington_Ingalls_Industries_rapport.htm'}]

    # Récupération du rapport et extraction des KPIs de la société initiale
    enriched_init = test_module_2([company_init])
    extractor_init = KpiExtractorLLM(model="gpt-4o-mini")
    company_init_with_kpis = extractor_init.extract_one_company(enriched_init[0])

    kpi_data=test_module_3(enriched_companies)
    test_module_4(kpi_data, initial_company=company_init_with_kpis)

def stack2():
    peers = test_module_1_yahoo()
    enriched_companies = test_module_2(peers)
    kpi_data=test_module_3(enriched_companies)
    test_module_4(kpi_data)

stack2()