"""Pydantic models partagés à travers le pipeline de benchmark."""

from typing import Optional

from pydantic import BaseModel, Field


class CompanyFilter(BaseModel):
    """Critères de sélection des entreprises comparables.

    Attributes:
        secteur: Secteur cible — label Yahoo Finance (ex: "Technology")
            ou code NAF/APE pour Pappers (ex: "4120A").
        min_ca: Chiffre d'affaires annuel minimum (EUR ou USD).
        max_ca: Chiffre d'affaires annuel maximum (EUR ou USD).
        min_effectifs: Nombre minimum de salariés.
        max_effectifs: Nombre maximum de salariés.
        pays: Filtrer par pays (ex: "France", "United States"). Insensible
            à la casse. Non applicable à Pappers (France implicite).
        continent: Filtrer par continent (ex: "Europe", "North America").
            Insensible à la casse. Non applicable à Pappers.
    """

    secteur: Optional[str] = Field(None, description="Secteur ou code NAF cible")
    min_ca: Optional[float] = Field(None, ge=0, description="CA minimum")
    max_ca: Optional[float] = Field(None, ge=0, description="CA maximum")
    min_effectifs: Optional[int] = Field(None, ge=0, description="Effectif minimum")
    max_effectifs: Optional[int] = Field(None, ge=0, description="Effectif maximum")
    pays: Optional[str] = Field(None, description="Pays cible (ex: 'France', 'United States')")
    continent: Optional[str] = Field(None, description="Continent cible (ex: 'Europe', 'North America')")
    min_market_cap: Optional[float] = Field(None, ge=0, description="Market Cap minimum")
    max_market_cap: Optional[float] = Field(None, ge=0, description="Market Cap maximum")
    

class PeerCompany(BaseModel):
    """Entreprise comparable identifiée par le PeerGroupFinder.

    Attributes:
        nom: Raison sociale de l'entreprise.
        ticker: Symbole boursier (None pour les sociétés non cotées).
        url_investisseur: URL de la page investisseurs ou du profil Pappers.
    """

    nom: str
    ticker: Optional[str] = None
    url_investisseur: Optional[str] = None
