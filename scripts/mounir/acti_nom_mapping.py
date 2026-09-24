"""Classify every ACTI_NOM label into the outcome used by the research question.

Each label gets:

- group      : condition | service | administrative
    condition      = a neighbourhood condition a resident would report
                     (counts toward the outcome)
    service        = a resident-facing request for a municipal service that is
                     not itself a neighbourhood condition (bin delivery, water
                     shut-off, housing inspection, new signage ...)
    administrative = municipal administration, payments, permits, taxes,
                     information, internal work orders and staff/elected-office
                     business
- theme      : one of the five outcome themes (condition labels only)
- subtheme   : finer description, so a theme can be narrowed or dropped
- confidence : clear | ambiguous (ambiguous labels mix conditions with
               non-condition requests; used for the sensitivity range)
- rule       : the regex that matched, so every assignment can be audited

Rules are applied in order to an accent-free, lowercase copy of the label with
any leading '*' (the City's marker for retired labels) removed. The first
matching rule wins. Labels matched by no rule are reported as 'unmapped'.

Run after build_dataset.py. Writes data/mounir/processed/acti_nom_mapping.csv.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from common import OUTPUT_DIR, WINDOW_YEARS


ROAD = "road_sidewalk"
WASTE = "waste_cleanliness"
GRAFFITI = "graffiti_vandalism"
NUISANCE = "noise_nuisance"
GREEN = "trees_greenspace"

C, S, A = "condition", "service", "administrative"
CLEAR, AMBIG = "clear", "ambiguous"

# (pattern, group, theme, subtheme, confidence) - first match wins.
RULES: list[tuple[str, str, str, str, str]] = [
    # --- complaints about unpermitted activity (before the generic permit rule) ---
    (r"coupe d'arbre prive sans permis|abattage illegal prive|arbre prive dangereux", C, GREEN, "private-tree hazards", CLEAR),
    (r"(chien|chat) sans permis", S, "", "animal services", CLEAR),
    (r"(entrave|travaux) sans permis", C, ROAD, "obstruction and works nuisances", AMBIG),
    (r"^permis - bruit", C, NUISANCE, "noise, light, odour, air", CLEAR),
    (r"^permis - logement", S, "", "housing and private buildings", CLEAR),

    # --- administrative: internal codes, money, permits, information, staff ---
    (r"^[zy] ?-", A, "", "internal work-order code (Z-/Y- prefix)", CLEAR),
    (r"^\d+ - ", A, "", "coded borough-internal vocabulary", CLEAR),
    (r"test-analyse|favorable - implantation|creation d'un simu|z-nouveau secteur", A, "", "internal technical", CLEAR),
    (r"encaissement|recettes diverses|cheque|remboursement|reclamation|cour municipale|plaidoyer",
     A, "", "payments, claims, court", CLEAR),
    (r"tax|evaluation|changement d'adresse|numero municipal|avis occupant", A, "", "taxation and civic address", CLEAR),
    (r"^info|info-|\(info\)|avis-alertes|carte info-neige|soutien carte info|internet - portail|mtlwifi"
     r"|communications|distribution des publications|acces a l'information|demande info|demande d'information"
     r"|demande de renseignement", A, "", "information and communications", CLEAR),
    (r"retour (de )?dds|^dds - |centre de services 311|bureau d'arrondissement|bureau du maire|maires et elus|\belu"
     r"|conseil d'arrondissement|greffe|direction|personnel|archives|budget|appel d'offre|election|assermentation"
     r"|document exige|langue francaise|code de langue|correspondance|prise de rendez-vous|statistique|suivi requis"
     r"|toponymie|geomatique|finances|rapport annuel|mesures d'urgence - gestion",
     A, "", "borough administration, elected officials, staff", CLEAR),
    (r"carte (d')?acces|adhesion|renouvellement|inscription|liste d'attente|vignette|certificat de vie|srrr"
     r"|tolerance de stationnement|revision du stationnement", A, "", "cards, registrations, parking permits", CLEAR),
    (r"permis|certificat|zonage|urbanisme|derogation|piia|^ccu$|enseigne commerciale|enseigne temporaire|autorisation"
     r"|^affaires - |cafe-terrasse|occupation du domaine public|occupation temporaire|occupation periodique"
     r"|occupation permanente|occupation batiment|acquisition de|subvention|^de - |demolition|entree charretiere"
     r"|abattage d'arbre prive|developpement (social|economique|communautaire|culturel)",
     A, "", "permits, zoning, grants", CLEAR),
    (r"^irp - ", A, "", "rapid-intervention coordination (internal)", CLEAR),
    (r"planification|ingenierie|surveillance des travaux|services techniques|soutien aux operations|demande interne"
     r"|entretien d'edifices municipaux|immeubles municipaux|immeuble ville|electricite - edifices|electricite-evenement"
     r"|plomberie|^travaux ?-?(elagage|abattage|plantation)|execution$|releve cpte|construction reseau local|^cvp - "
     r"|prr-|inspection(s)? en arrondissement|inspection - matieres|inspection vignette|avis de courtoisie"
     r"|affiche proprete|marquage - contrat|reparation de gazon - contrat|parcours de balai|^at - "
     r"|plainte fonctionnement|plainte contre|^plainte$",
     A, "", "internal work, inspections, notices issued by the city", CLEAR),

    # --- winter maintenance ---
    (r"neige - remorquage|billet de depot a neige|avis d'infraction - neige|signalisation - neige|deneigeur prive",
     S, "", "snow-removal logistics", CLEAR),
    (r"gazon endommage-operation deneigement", C, GREEN, "lawns, horticulture, park upkeep", CLEAR),
    (r"glissant", C, ROAD, "snow and ice", CLEAR),
    (r"deneigement|deblaiement|depot illegal - neige|depot neige", C, ROAD, "snow and ice", AMBIG),

    # --- graffiti / vandalism / illegal postering ---
    (r"graffiti|vandalisme|affichage (- )?illegal|affichage sauvage|affiche (fixee|autoportante)",
     C, GRAFFITI, "graffiti, vandalism, illegal postering", CLEAR),

    # --- special waste collections and greening programmes (before tree rules) ---
    (r"collecte (d'ecran|d'arbres de noel|de branches|de feuilles)|arbres de noel", S, "",
     "bins, bags, drop-off and special collections", CLEAR),
    (r"plantation d'arbre|une naissance / un arbre|arbre - suivi de plantation|distribution de vegetaux"
     r"|maisons fleuries|ruelle verte", S, "", "tree planting and greening programmes", CLEAR),

    # --- private-property nuisances (before the lawn rule) ---
    (r"gazon ou herbes hautes - domaine prive|batiment vacant|cloture et haie|haie privee|cloture privee"
     r"|obstacle en marge|antenne satellite", C, NUISANCE, "private-property nuisances", CLEAR),

    # --- trees and green space ---
    (r"danger potentiel - arbre|branche tombee|arbre tombe|maladie ou insectes - arbre|racines d'arbres|puceron"
     r"|agrile|signalisation cachee par des vegetaux", C, GREEN, "tree hazards and pests", CLEAR),
    (r"elagage|abattage|essouchement|arbre", C, GREEN, "tree maintenance requests", AMBIG),
    (r"parc - proprete|piscine - proprete|refuge - proprete|panier de parc", C, GREEN, "park cleanliness", CLEAR),
    (r"gazon|mauvaises herbes|herbe longue|herbe a poux|horticulture|^parc - divers|^grand parc$|mobilier de parc"
     r"|haie hauteur|feuilles mortes", C, GREEN, "lawns, horticulture, park upkeep", AMBIG),

    # --- noise and nuisances ---
    (r"bruit|souffleur a feuilles|eclairage nuisible|voirie - vibrations|odeur d'egout|pollution de l'air"
     r"|ralenti inutile|poele et foyer|thermopompe", C, NUISANCE, "noise, light, odour, air", CLEAR),
    (r"extermination a l'exterieur|animaux sauvages - nuisance|nourr\w*ssage|nuisance animaux|guepes|seringue",
     C, NUISANCE, "pests and public-health nuisances", CLEAR),

    # --- housing (private dwellings: a service, not a neighbourhood condition) ---
    (r"logement|eviction|airbnb|punaises|insalubrite|immeuble - |chauffage|vermine|moisissure|piscine ou spa"
     r"|piscine non cloturee|balcon$|aide au relogement", S, "", "housing and private buildings", CLEAR),

    # --- waste and cleanliness ---
    (r"depot illegal|depot sauvage|dechets - illegal|terrain insalubre|entreposage non autorise|balcon malpropre"
     r"|nettoyage du domaine public|debris|animal mort|panier de rue a vider|poubelle|balai mecanique"
     r"|sortis? en temps prohibe|bicyclette abandonnee|auto abandonnee|abribus - nettoyage|crue des eaux - nettoyage"
     r"|salubrite|proprete|nettoyage de mail|nettoyage sites", C, WASTE, "illegal dumping, litter, street cleaning", CLEAR),
    (r"non ramass", C, WASTE, "missed collection", CLEAR),
    (r"^bac|bac (roulant|montrealais|de |brun|resid)|bac.*livraison|distribution (de sacs?|au comptoir|de compost)"
     r"|sacs d'emplettes|modification de volume|vente de|compost$|demande de bac|endommage, perdu, vole"
     r"|ramassage du bac|conteneurs|installation pour dechets|refrigerateur|climatiseur$|ecocentre"
     r"|centres de traitement|complexe environnemental|residus domestiques dangereux|plastiques a usage"
     r"|baril de recuperation|ecoquartier|objet vol\. a ramasser|panier de rue - installation", S, "", "bins, bags, drop-off and special collections", CLEAR),
    (r"collecte des encombrants", S, "", "bulky-item collection booking", AMBIG),
    (r"collecte|recyclage|residus (verts|alimentaires)|matieres organiques|encombrants|^mo - |dechets",
     C, WASTE, "regular collection problems", AMBIG),

    # --- roads, sidewalks and street infrastructure ---
    (r"marquage de la chaussee - nouveau|dos d'ane", S, "", "traffic, parking, cycling, lighting requests", CLEAR),
    (r"voirie ?- ?divers", C, ROAD, "pavement, sidewalks, road surface", AMBIG),
    (r"nid-de-poule|pavage|pave uni|asphalte|defoncage|chaussee|trottoir|bordure|voirie ?- ?(poubelle|banc|accumulation"
     r"|degagement|nettoyage des fosses|camion|mobilier)|pont et tunnel|boite de service",
     C, ROAD, "pavement, sidewalks, road surface", CLEAR),
    (r"eclairage existant|eclairage - exterieur|feux de circulation - entretien|feux pietons|signalisation ecrite - ?entretien"
     r"|panneau de signalisation entretien|signalisation (manquante|endommagee)|panneaux electroniques|signalisation - ligne",
     C, ROAD, "street lighting, signals, signage defects", CLEAR),
    (r"occupation non-autorisee|utilite publique - excavation|problemes a la suite de travaux",
     C, ROAD, "obstruction and works nuisances", AMBIG),
    (r"mobilier urbain|\bbanc|statues|cloture publique", C, ROAD, "street furniture", AMBIG),

    # --- other resident-facing services (not counted as conditions) ---
    (r"feux|signalisation|marquage|apaisement|eclairage|etudes|circulation|voie cyclable|piste cyclable|velo"
     r"|stationnement|parcometre|debarcadere|livraison|bornes de recharge|mobilite|remorquage|exces de vitesse"
     r"|intersection dangereuse|panneaux|modification secteur", S, "", "traffic, parking, cycling, lighting requests", CLEAR),
    (r"animal|animaux|chien|chat|micropucage|parc canin", S, "", "animal services", CLEAR),
    (r"sport|loisir|culture|camp de jour|evenement|chalet|patinoire|piscine|pataugeoire|jeux d'eau|modules de jeux"
     r"|amenagement de parc|parc|tennis|arena|terrain sportif|terrains exterieurs|jardins communautaires|installations"
     r"|bibliotheque|espace pour la vie|amuseurs|artistes|marches - courses|rampe de mise|menuiserie|article de jeux"
     r"|station reparation|oriflamme|ecole-ecolo", S, "", "parks, recreation, culture, events", CLEAR),
    (r"\beau\b|egout|puisard|borne|aqueduc|fuite|pression|refoulement|inondation|crue|plomb|compteur|regard"
     r"|trou d'homme|branchement|conduite|infiltration|raccordement|fluor|gout, odeur|\bbf\b",
     S, "", "water and sewer service", CLEAR),
    (r"itinerance|cohabitation|securite|spvm|police|pompier|incendie|sinistre|sante|inspection|developpement social"
     r"|logement social|vente-debarras|distribution de circulaire|transition ecologique|environnement|pesticide"
     r"|epandage|carrieres|entretien general|mesure d'urgence|abri temporaire|abri d'auto|electricite|varia|divers"
     r"|autre|travaux publics", S, "", "social, safety, environment and other services", AMBIG),
]

COMPILED = [(re.compile(pattern), group, theme, sub, conf, pattern) for pattern, group, theme, sub, conf in RULES]


def label_key(label: str) -> str:
    text = unicodedata.normalize("NFKD", str(label))
    text = "".join(char for char in text if not unicodedata.combining(char)).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text.lstrip("* ").strip()


def classify(label: str) -> dict:
    key = label_key(label)
    for regex, group, theme, sub, conf, pattern in COMPILED:
        if regex.search(key):
            return {"group": group, "theme": theme, "subtheme": sub, "confidence": conf, "rule": pattern}
    return {"group": "unmapped", "theme": "", "subtheme": "", "confidence": "", "rule": ""}


def main() -> None:
    counts = pd.read_csv(OUTPUT_DIR / "label_year_counts.csv")
    requests = counts[counts["NATURE"].isin(["Requete", "Plainte"])]
    window = requests[requests["year"].isin(WINDOW_YEARS)]
    table = pd.DataFrame({"ACTI_NOM": sorted(set(counts["ACTI_NOM"].dropna()))})
    table["retired_label"] = table["ACTI_NOM"].str.startswith("*")
    table = table.join(table["ACTI_NOM"].map(classify).apply(pd.Series))
    table["rows_2023_2025_requete_plainte"] = table["ACTI_NOM"].map(
        window.groupby("ACTI_NOM")["rows"].sum()).fillna(0).astype(int)
    table["rows_2014_present_requete_plainte"] = table["ACTI_NOM"].map(
        requests.groupby("ACTI_NOM")["rows"].sum()).fillna(0).astype(int)
    table = table.sort_values("rows_2023_2025_requete_plainte", ascending=False)
    table.to_csv(OUTPUT_DIR / "acti_nom_mapping.csv", index=False, encoding="utf-8")

    in_window = table[table["rows_2023_2025_requete_plainte"] > 0]
    print(in_window.groupby(["group", "theme"])["rows_2023_2025_requete_plainte"].agg(["count", "sum"]))
    unmapped = in_window[in_window["group"].eq("unmapped")]
    print(f"unmapped labels in window: {len(unmapped)} ({unmapped['rows_2023_2025_requete_plainte'].sum()} rows)")


if __name__ == "__main__":
    main()
