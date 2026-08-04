"""
common/util/classifiers/sic_sector_classifier.py
------------------------------------------------
Maps the SEC SIC code to two levels of in-house classification:

    sector_code    -> 11 GICS-style sectors (ENERGY, CONS_DISCR, HEALTH_CARE, ...)
    industry_code  -> fine-grained sub-industry (PHARMA, BIOTECH, SOFTWARE, BANKS, ...)

This is what lets you narrow the universe before running the Document Tag Indexer:
"pharma only" = industry_code IN ('PHARMA','BIOTECH').

Resolution goes from most specific to most generic:
    1) exact 4-digit SIC
    2) 3-digit prefix
    3) 2-digit major group
    4) no match -> UNKNOWN / UNKNOWN

The 4-digit level exists because the 2-digit one lies exactly where it matters
most: 28 is chemicals but 2834 is pharma, and 37 is autos but 3721 is aerospace.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Sectors
# ---------------------------------------------------------------------------
SECTORS: Dict[str, str] = {
    "ENERGY":       "Energy",
    "MATERIALS":    "Materials",
    "INDUSTRIALS":  "Industrials",
    "CONS_DISCR":   "Consumer Discretionary",
    "CONS_STAPLES": "Consumer Staples",
    "HEALTH_CARE":  "Health Care",
    "FINANCIALS":   "Financials",
    "INFO_TECH":    "Information Technology",
    "COMM_SVCS":    "Communication Services",
    "UTILITIES":    "Utilities",
    "REAL_ESTATE":  "Real Estate",
    "GOVT":         "Government / Agency",
    "UNKNOWN":      "Unclassified",
}

INDUSTRIES: Dict[str, str] = {
    "OIL_GAS_EP":     "Oil & Gas Exploration & Production",
    "OIL_GAS_SVCS":   "Oil & Gas Equipment & Services",
    "REFINING":       "Refining & Marketing",
    "PIPELINES":      "Pipelines & Midstream",
    "COAL":           "Coal",
    "CHEMICALS":      "Chemicals",
    "METALS_MINING":  "Metals & Mining",
    "STEEL":          "Steel",
    "PAPER_PACK":     "Paper & Packaging",
    "CONSTR_MAT":     "Construction Materials",
    "AEROSPACE_DEF":  "Aerospace & Defense",
    "MACHINERY":      "Machinery",
    "ELECTRICAL_EQ":  "Electrical Equipment",
    "BUILDING_PROD":  "Building Products",
    "CONSTR_ENG":     "Construction & Engineering",
    "AIRLINES":       "Airlines",
    "RAILROADS":      "Railroads",
    "TRUCKING":       "Trucking & Logistics",
    "MARINE":         "Marine Transport",
    "COMMERCIAL_SVC": "Commercial Services & Supplies",
    "PROF_SVCS":      "Professional Services",
    "AUTOS":          "Automobiles & Components",
    "HOMEBUILDERS":   "Homebuilders",
    "APPAREL":        "Apparel & Luxury Goods",
    "LEISURE_PROD":   "Leisure Products",
    "HOUSEHOLD_DUR":  "Household Durables",
    "RESTAURANTS":    "Restaurants",
    "HOTELS_LEISURE": "Hotels & Leisure",
    "RETAIL_SPEC":    "Specialty Retail",
    "RETAIL_BROAD":   "Broadline Retail",
    "EDUCATION":      "Education Services",
    "FOOD_PROD":      "Food Products",
    "BEVERAGES":      "Beverages",
    "TOBACCO":        "Tobacco",
    "HOUSEHOLD_PROD": "Household & Personal Products",
    "FOOD_RETAIL":    "Food & Staples Retail",
    "AGRICULTURE":    "Agriculture",
    "PHARMA":         "Pharmaceuticals",
    "BIOTECH":        "Biotechnology & Life Sciences Research",
    "MEDDEV":         "Medical Devices & Equipment",
    "HEALTH_SVCS":    "Health Care Providers & Services",
    "MANAGED_CARE":   "Managed Care",
    "HEALTH_TECH":    "Health Care Technology",
    "BANKS":          "Banks",
    "CONSUMER_FIN":   "Consumer Finance",
    "CAPITAL_MKTS":   "Capital Markets",
    "INSURANCE":      "Insurance",
    "DIVERS_FIN":     "Diversified Financials",
    "BLANK_CHECK":    "Blank Check / SPAC",
    "SOFTWARE":       "Software",
    "IT_SVCS":        "IT Services",
    "SEMIS":          "Semiconductors & Equipment",
    "HARDWARE":       "Technology Hardware",
    "COMM_EQUIP":     "Communications Equipment",
    "ELECTRONIC_CMP": "Electronic Components & Instruments",
    "TELECOM":        "Telecommunication Services",
    "MEDIA":          "Media & Entertainment",
    "INTERACTIVE":    "Interactive Media & Services",
    "PUBLISHING":     "Publishing",
    "UTIL_ELECTRIC":  "Electric Utilities",
    "UTIL_GAS":       "Gas Utilities",
    "UTIL_WATER":     "Water Utilities",
    "WASTE":          "Waste Management",
    "REITS":          "REITs",
    "REAL_ESTATE_OP": "Real Estate Operating & Services",
    "GOVT":           "Government / Agency",
    "UNKNOWN":        "Unclassified",
}

# ---------------------------------------------------------------------------
# 1) Exact SIC (4 digits) -> (sector, industry)
# ---------------------------------------------------------------------------
SIC4: Dict[str, Tuple[str, str]] = {
    # --- Health care / pharma: the most important overrides for universe slicing ---
    "2833": ("HEALTH_CARE", "PHARMA"),      # medicinal chemicals & botanicals
    "2834": ("HEALTH_CARE", "PHARMA"),      # pharmaceutical preparations
    "2835": ("HEALTH_CARE", "MEDDEV"),      # in vitro & in vivo diagnostics
    "2836": ("HEALTH_CARE", "BIOTECH"),     # biological products
    "8731": ("HEALTH_CARE", "BIOTECH"),     # commercial physical & biological research
    "5122": ("HEALTH_CARE", "HEALTH_SVCS"), # drugs & druggists sundries (wholesale)
    "5912": ("CONS_STAPLES", "FOOD_RETAIL"),# drug stores
    "6324": ("HEALTH_CARE", "MANAGED_CARE"),# hospital & medical service plans
    "7372": ("INFO_TECH", "SOFTWARE"),      # prepackaged software

    # --- Autos vs aerospace (both fall under SIC 37) ---
    "3711": ("CONS_DISCR", "AUTOS"),
    "3713": ("CONS_DISCR", "AUTOS"),
    "3714": ("CONS_DISCR", "AUTOS"),
    "3715": ("CONS_DISCR", "AUTOS"),
    "3716": ("CONS_DISCR", "AUTOS"),
    "3751": ("CONS_DISCR", "LEISURE_PROD"),
    "3721": ("INDUSTRIALS", "AEROSPACE_DEF"),
    "3724": ("INDUSTRIALS", "AEROSPACE_DEF"),
    "3728": ("INDUSTRIALS", "AEROSPACE_DEF"),
    "3760": ("INDUSTRIALS", "AEROSPACE_DEF"),
    "3761": ("INDUSTRIALS", "AEROSPACE_DEF"),
    "3764": ("INDUSTRIALS", "AEROSPACE_DEF"),
    "3769": ("INDUSTRIALS", "AEROSPACE_DEF"),
    "3730": ("INDUSTRIALS", "MARINE"),
    "3743": ("INDUSTRIALS", "RAILROADS"),
    "3795": ("INDUSTRIALS", "AEROSPACE_DEF"),
    "3812": ("INDUSTRIALS", "AEROSPACE_DEF"),  # search, detection, navigation

    # --- Construction: homebuilders are consumer discretionary ---
    "1531": ("CONS_DISCR", "HOMEBUILDERS"),

    # --- REITs and holding companies ---
    "6798": ("REAL_ESTATE", "REITS"),
    "6770": ("FINANCIALS", "BLANK_CHECK"),
    "6199": ("FINANCIALS", "DIVERS_FIN"),
    "6221": ("FINANCIALS", "CAPITAL_MKTS"),
    "6282": ("FINANCIALS", "CAPITAL_MKTS"),
    "6726": ("FINANCIALS", "CAPITAL_MKTS"),

    # --- Tech / comms ---
    "3663": ("INFO_TECH", "COMM_EQUIP"),
    "3661": ("INFO_TECH", "COMM_EQUIP"),
    "3669": ("INFO_TECH", "COMM_EQUIP"),
    "3674": ("INFO_TECH", "SEMIS"),
    "3559": ("INFO_TECH", "SEMIS"),
    "3672": ("INFO_TECH", "ELECTRONIC_CMP"),
    "3827": ("INFO_TECH", "ELECTRONIC_CMP"),
    "3829": ("INFO_TECH", "ELECTRONIC_CMP"),
    "7311": ("COMM_SVCS", "MEDIA"),
    "7370": ("INFO_TECH", "IT_SVCS"),
    "7371": ("INFO_TECH", "IT_SVCS"),
    "7373": ("INFO_TECH", "IT_SVCS"),
    "7374": ("INFO_TECH", "IT_SVCS"),
    "7375": ("COMM_SVCS", "INTERACTIVE"),
    "7379": ("INFO_TECH", "IT_SVCS"),
    "7385": ("COMM_SVCS", "TELECOM"),
    "7389": ("INDUSTRIALS", "COMMERCIAL_SVC"),
    "8742": ("INDUSTRIALS", "PROF_SVCS"),
    "8748": ("INDUSTRIALS", "PROF_SVCS"),
    "8721": ("INDUSTRIALS", "PROF_SVCS"),
    "8711": ("INDUSTRIALS", "CONSTR_ENG"),

    # --- Utilities / waste ---
    "4911": ("UTILITIES", "UTIL_ELECTRIC"),
    "4922": ("ENERGY", "PIPELINES"),
    "4923": ("UTILITIES", "UTIL_GAS"),
    "4924": ("UTILITIES", "UTIL_GAS"),
    "4931": ("UTILITIES", "UTIL_ELECTRIC"),
    "4932": ("UTILITIES", "UTIL_GAS"),
    "4941": ("UTILITIES", "UTIL_WATER"),
    "4953": ("INDUSTRIALS", "WASTE"),
    "4959": ("INDUSTRIALS", "WASTE"),

    # --- Energy ---
    "1311": ("ENERGY", "OIL_GAS_EP"),
    "1381": ("ENERGY", "OIL_GAS_SVCS"),
    "1382": ("ENERGY", "OIL_GAS_SVCS"),
    "1389": ("ENERGY", "OIL_GAS_SVCS"),
    "3533": ("ENERGY", "OIL_GAS_SVCS"),
    "2911": ("ENERGY", "REFINING"),
    "5172": ("ENERGY", "REFINING"),
}

# ---------------------------------------------------------------------------
# 2) 3-digit prefix -> (sector, industry)
# ---------------------------------------------------------------------------
SIC3: Dict[str, Tuple[str, str]] = {
    "283": ("HEALTH_CARE", "PHARMA"),
    "384": ("HEALTH_CARE", "MEDDEV"),
    "385": ("HEALTH_CARE", "MEDDEV"),
    "357": ("INFO_TECH",   "HARDWARE"),
    "367": ("INFO_TECH",   "SEMIS"),
    "366": ("INFO_TECH",   "COMM_EQUIP"),
    "737": ("INFO_TECH",   "IT_SVCS"),
    "481": ("COMM_SVCS",   "TELECOM"),
    "483": ("COMM_SVCS",   "MEDIA"),
    "484": ("COMM_SVCS",   "MEDIA"),
    "489": ("COMM_SVCS",   "TELECOM"),
    "602": ("FINANCIALS",  "BANKS"),
    "603": ("FINANCIALS",  "BANKS"),
    "601": ("FINANCIALS",  "BANKS"),
    "614": ("FINANCIALS",  "CONSUMER_FIN"),
    "615": ("FINANCIALS",  "CONSUMER_FIN"),
    "616": ("FINANCIALS",  "CONSUMER_FIN"),
    "631": ("FINANCIALS",  "INSURANCE"),
    "632": ("FINANCIALS",  "INSURANCE"),
    "633": ("FINANCIALS",  "INSURANCE"),
    "635": ("FINANCIALS",  "INSURANCE"),
    "641": ("FINANCIALS",  "INSURANCE"),
    "653": ("REAL_ESTATE", "REAL_ESTATE_OP"),
    "655": ("REAL_ESTATE", "REAL_ESTATE_OP"),
    "801": ("HEALTH_CARE", "HEALTH_SVCS"),
    "805": ("HEALTH_CARE", "HEALTH_SVCS"),
    "806": ("HEALTH_CARE", "HEALTH_SVCS"),
    "807": ("HEALTH_CARE", "HEALTH_SVCS"),
    "809": ("HEALTH_CARE", "HEALTH_SVCS"),
    "581": ("CONS_DISCR",  "RESTAURANTS"),
    "701": ("CONS_DISCR",  "HOTELS_LEISURE"),
    "541": ("CONS_STAPLES","FOOD_RETAIL"),
    "208": ("CONS_STAPLES","BEVERAGES"),
    "211": ("CONS_STAPLES","TOBACCO"),
    "284": ("CONS_STAPLES","HOUSEHOLD_PROD"),
    "331": ("MATERIALS",   "STEEL"),
    "451": ("INDUSTRIALS", "AIRLINES"),
    "421": ("INDUSTRIALS", "TRUCKING"),
    "401": ("INDUSTRIALS", "RAILROADS"),
    "441": ("INDUSTRIALS", "MARINE"),
    "442": ("INDUSTRIALS", "MARINE"),
    "444": ("INDUSTRIALS", "MARINE"),
    "873": ("INDUSTRIALS", "PROF_SVCS"),
}

# ---------------------------------------------------------------------------
# 3) 2-digit major group -> (sector, industry)
# ---------------------------------------------------------------------------
SIC2: Dict[str, Tuple[str, str]] = {
    "01": ("CONS_STAPLES", "AGRICULTURE"),
    "02": ("CONS_STAPLES", "AGRICULTURE"),
    "07": ("CONS_STAPLES", "AGRICULTURE"),
    "08": ("MATERIALS",    "PAPER_PACK"),
    "09": ("CONS_STAPLES", "FOOD_PROD"),
    "10": ("MATERIALS",    "METALS_MINING"),
    "12": ("ENERGY",       "COAL"),
    "13": ("ENERGY",       "OIL_GAS_EP"),
    "14": ("MATERIALS",    "CONSTR_MAT"),
    "15": ("INDUSTRIALS",  "CONSTR_ENG"),
    "16": ("INDUSTRIALS",  "CONSTR_ENG"),
    "17": ("INDUSTRIALS",  "CONSTR_ENG"),
    "20": ("CONS_STAPLES", "FOOD_PROD"),
    "21": ("CONS_STAPLES", "TOBACCO"),
    "22": ("CONS_DISCR",   "APPAREL"),
    "23": ("CONS_DISCR",   "APPAREL"),
    "24": ("MATERIALS",    "BUILDING_PROD"),
    "25": ("CONS_DISCR",   "HOUSEHOLD_DUR"),
    "26": ("MATERIALS",    "PAPER_PACK"),
    "27": ("COMM_SVCS",    "PUBLISHING"),
    "28": ("MATERIALS",    "CHEMICALS"),
    "29": ("ENERGY",       "REFINING"),
    "30": ("MATERIALS",    "CHEMICALS"),
    "31": ("CONS_DISCR",   "APPAREL"),
    "32": ("MATERIALS",    "CONSTR_MAT"),
    "33": ("MATERIALS",    "METALS_MINING"),
    "34": ("INDUSTRIALS",  "BUILDING_PROD"),
    "35": ("INDUSTRIALS",  "MACHINERY"),
    "36": ("INFO_TECH",    "ELECTRICAL_EQ"),
    "37": ("CONS_DISCR",   "AUTOS"),
    "38": ("INFO_TECH",    "ELECTRONIC_CMP"),
    "39": ("CONS_DISCR",   "LEISURE_PROD"),
    "40": ("INDUSTRIALS",  "RAILROADS"),
    "41": ("INDUSTRIALS",  "TRUCKING"),
    "42": ("INDUSTRIALS",  "TRUCKING"),
    "44": ("INDUSTRIALS",  "MARINE"),
    "45": ("INDUSTRIALS",  "AIRLINES"),
    "46": ("ENERGY",       "PIPELINES"),
    "47": ("INDUSTRIALS",  "TRUCKING"),
    "48": ("COMM_SVCS",    "TELECOM"),
    "49": ("UTILITIES",    "UTIL_ELECTRIC"),
    "50": ("INDUSTRIALS",  "COMMERCIAL_SVC"),
    "51": ("CONS_STAPLES", "FOOD_RETAIL"),
    "52": ("CONS_DISCR",   "RETAIL_SPEC"),
    "53": ("CONS_DISCR",   "RETAIL_BROAD"),
    "54": ("CONS_STAPLES", "FOOD_RETAIL"),
    "55": ("CONS_DISCR",   "RETAIL_SPEC"),
    "56": ("CONS_DISCR",   "RETAIL_SPEC"),
    "57": ("CONS_DISCR",   "RETAIL_SPEC"),
    "58": ("CONS_DISCR",   "RESTAURANTS"),
    "59": ("CONS_DISCR",   "RETAIL_SPEC"),
    "60": ("FINANCIALS",   "BANKS"),
    "61": ("FINANCIALS",   "CONSUMER_FIN"),
    "62": ("FINANCIALS",   "CAPITAL_MKTS"),
    "63": ("FINANCIALS",   "INSURANCE"),
    "64": ("FINANCIALS",   "INSURANCE"),
    "65": ("REAL_ESTATE",  "REAL_ESTATE_OP"),
    "67": ("FINANCIALS",   "DIVERS_FIN"),
    "70": ("CONS_DISCR",   "HOTELS_LEISURE"),
    "72": ("CONS_DISCR",   "RETAIL_SPEC"),
    "73": ("INFO_TECH",    "IT_SVCS"),
    "75": ("CONS_DISCR",   "AUTOS"),
    "76": ("INDUSTRIALS",  "COMMERCIAL_SVC"),
    "78": ("COMM_SVCS",    "MEDIA"),
    "79": ("CONS_DISCR",   "HOTELS_LEISURE"),
    "80": ("HEALTH_CARE",  "HEALTH_SVCS"),
    "81": ("INDUSTRIALS",  "PROF_SVCS"),
    "82": ("CONS_DISCR",   "EDUCATION"),
    "83": ("HEALTH_CARE",  "HEALTH_SVCS"),
    "86": ("INDUSTRIALS",  "COMMERCIAL_SVC"),
    "87": ("INDUSTRIALS",  "PROF_SVCS"),
    "89": ("INDUSTRIALS",  "PROF_SVCS"),
    "91": ("GOVT",         "GOVT"),
    "92": ("GOVT",         "GOVT"),
    "93": ("GOVT",         "GOVT"),
    "94": ("GOVT",         "GOVT"),
    "95": ("GOVT",         "GOVT"),
    "96": ("GOVT",         "GOVT"),
    "97": ("GOVT",         "GOVT"),
    "99": ("UNKNOWN",      "UNKNOWN"),
}


def normalize_sic(sic) -> Optional[str]:
    """Normalizes the SIC into a zero-padded 4-digit string."""
    if sic is None:
        return None
    raw = str(sic).strip()
    if not raw or not raw.isdigit():
        return None
    return raw.zfill(4)[-4:] if len(raw) > 4 else raw.zfill(4)


def classify_sic(sic) -> Tuple[str, str, str, str]:
    """
    Returns (sector_code, sector_name, industry_code, industry_name).
    Never raises: falls back to UNKNOWN when nothing matches.
    """
    code = normalize_sic(sic)
    if code is None:
        return ("UNKNOWN", SECTORS["UNKNOWN"], "UNKNOWN", INDUSTRIES["UNKNOWN"])

    for table, key in ((SIC4, code), (SIC3, code[:3]), (SIC2, code[:2])):
        hit = table.get(key)
        if hit:
            sector, industry = hit
            return (sector, SECTORS.get(sector, sector),
                    industry, INDUSTRIES.get(industry, industry))

    return ("UNKNOWN", SECTORS["UNKNOWN"], "UNKNOWN", INDUSTRIES["UNKNOWN"])


class SICSectorClassifier:
    """Class wrapper, to follow the convention used across common/util."""

    SECTORS = SECTORS
    INDUSTRIES = INDUSTRIES

    @staticmethod
    def normalize(sic) -> Optional[str]:
        return normalize_sic(sic)

    @staticmethod
    def classify(sic) -> Tuple[str, str, str, str]:
        """Returns (sector_code, sector_name, industry_code, industry_name)."""
        return classify_sic(sic)


if __name__ == "__main__":
    for probe in ("2834", "8731", 3571, "6798", "1311", "7372", "6022", None, "0000"):
        print(probe, "->", SICSectorClassifier.classify(probe))
