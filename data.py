"""Curated source catalogue and safe routing rules for EvidenceLens.

This module separates *where to look* from model prompting.  It deliberately
does not imply that a public webpage, a credentialed repository, or a
controlled patient-level database can be accessed automatically.  The app may
search only the domains returned by ``build_trusted_retrieval_buckets``;
controlled resources remain discovery-only entries in the catalogue.
"""

from __future__ import annotations

from collections.abc import Iterable
import re


OPEN = "Open / public"
CREDENTIALLED = "Credentialed research access"
CONTROLLED = "Controlled or approved-researcher access"
MIXED = "Mixed access; check source-specific terms"


# A single, editable catalogue.  Add a new source here rather than scattering
# domain names throughout app.py.  ``search_enabled`` means it may be offered
# to the web-retrieval router; it never grants an account or a data-use licence.
MEDICAL_SOURCE_CATALOG: list[dict[str, object]] = [
    # Clinical and biomedical datasets
    {
        "id": "physionet",
        "name": "PhysioNet",
        "category": "clinical_dataset",
        "url": "https://physionet.org/",
        "domains": ["physionet.org"],
        "coverage": "International contributions",
        "data_available": "Clinical datasets, ECG, physiological signals, imaging, and biomedical data",
        "access": MIXED,
        "search_enabled": False,
    },
    {
        "id": "mimic_iv",
        "name": "MIMIC-IV",
        "category": "clinical_dataset",
        "url": "https://physionet.org/content/mimiciv/",
        "domains": ["physionet.org"],
        "coverage": "United States",
        "data_available": "De-identified hospital and ICU records, diagnoses, laboratory results, medications, and outcomes",
        "access": CREDENTIALLED,
        "search_enabled": False,
    },
    {
        "id": "eicu",
        "name": "eICU Collaborative Research Database",
        "category": "clinical_dataset",
        "url": "https://physionet.org/content/eicu-crd/",
        "domains": ["physionet.org"],
        "coverage": "United States, multiple hospitals",
        "data_available": "ICU clinical records and treatment information",
        "access": CREDENTIALLED,
        "search_enabled": False,
    },
    {
        "id": "amsterdamumcdb",
        "name": "AmsterdamUMCdb",
        "category": "clinical_dataset",
        "url": "https://amsterdammedicaldatascience.nl/",
        "domains": ["amsterdammedicaldatascience.nl"],
        "coverage": "Netherlands",
        "data_available": "ICU demographics, vital signs, laboratory tests, and medications",
        "access": CONTROLLED,
        "search_enabled": False,
    },
    {
        "id": "hirid",
        "name": "HiRID",
        "category": "clinical_dataset",
        "url": "https://physionet.org/content/hirid/",
        "domains": ["physionet.org"],
        "coverage": "Switzerland",
        "data_available": "High-resolution ICU clinical time-series data",
        "access": CREDENTIALLED,
        "search_enabled": False,
    },
    {
        "id": "all_of_us",
        "name": "All of Us Research Hub",
        "category": "clinical_dataset",
        "url": "https://researchallofus.org/",
        "domains": ["researchallofus.org"],
        "coverage": "United States",
        "data_available": "EHRs, surveys, physical measurements, wearables, and genomics",
        "access": CONTROLLED,
        "search_enabled": False,
    },
    {
        "id": "uk_biobank",
        "name": "UK Biobank",
        "category": "clinical_dataset",
        "url": "https://www.ukbiobank.ac.uk/",
        "domains": ["ukbiobank.ac.uk"],
        "coverage": "United Kingdom",
        "data_available": "Longitudinal health and research data",
        "access": CONTROLLED,
        "search_enabled": False,
    },
    # Imaging and omics repositories
    {
        "id": "tcia",
        "name": "The Cancer Imaging Archive (TCIA)",
        "category": "imaging_dataset",
        "url": "https://www.cancerimagingarchive.net/",
        "domains": ["cancerimagingarchive.net"],
        "coverage": "International cancer-imaging collections",
        "data_available": "De-identified cancer imaging collections",
        "access": MIXED,
        "search_enabled": True,
    },
    {
        "id": "idc",
        "name": "NCI Imaging Data Commons (IDC)",
        "category": "imaging_dataset",
        "url": "https://portal.imaging.datacommons.cancer.gov/",
        "domains": ["imaging.datacommons.cancer.gov", "datacommons.cancer.gov"],
        "coverage": "Cancer imaging",
        "data_available": "DICOM radiology, pathology, segmentations, and annotations",
        "access": OPEN,
        "search_enabled": True,
    },
    {
        "id": "openneuro",
        "name": "OpenNeuro",
        "category": "imaging_dataset",
        "url": "https://openneuro.org/",
        "domains": ["openneuro.org"],
        "coverage": "International neuroscience",
        "data_available": "MRI, PET, EEG, MEG, and related datasets",
        "access": OPEN,
        "search_enabled": True,
    },
    {
        "id": "geo",
        "name": "Gene Expression Omnibus (GEO)",
        "category": "genomics_dataset",
        "url": "https://www.ncbi.nlm.nih.gov/geo/",
        "domains": ["ncbi.nlm.nih.gov"],
        "coverage": "International functional genomics",
        "data_available": "Gene-expression and functional-genomics datasets",
        "access": OPEN,
        "search_enabled": True,
    },
    {
        "id": "dbgap",
        "name": "dbGaP",
        "category": "genomics_dataset",
        "url": "https://dbgap.ncbi.nlm.nih.gov/",
        "domains": ["dbgap.ncbi.nlm.nih.gov"],
        "coverage": "Genotype–phenotype studies",
        "data_available": "Genotype, phenotype, and associated clinical data",
        "access": MIXED,
        "search_enabled": True,
    },
    {
        "id": "gdc",
        "name": "Genomic Data Commons (GDC)",
        "category": "genomics_dataset",
        "url": "https://portal.gdc.cancer.gov/",
        "domains": ["portal.gdc.cancer.gov", "gdc.cancer.gov"],
        "coverage": "Cancer genomics",
        "data_available": "Cancer genomics, clinical, and biospecimen data including TCGA",
        "access": MIXED,
        "search_enabled": True,
    },
    # Public-health data
    {
        "id": "who_data",
        "name": "WHO Data",
        "category": "public_health",
        "url": "https://www.who.int/data/",
        "domains": ["who.int"],
        "coverage": "Worldwide",
        "data_available": "Health indicators, mortality, disease burden, and health inequalities",
        "access": OPEN,
        "search_enabled": True,
    },
    {
        "id": "cdc_open_data",
        "name": "CDC Open Data",
        "category": "public_health",
        "url": "https://data.cdc.gov/",
        "domains": ["data.cdc.gov", "cdc.gov"],
        "coverage": "United States",
        "data_available": "Public-health surveillance, diseases, vaccination, and risk factors",
        "access": OPEN,
        "search_enabled": True,
    },
    {
        "id": "nhanes",
        "name": "NHANES",
        "category": "public_health",
        "url": "https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/",
        "domains": ["wwwn.cdc.gov", "cdc.gov"],
        "coverage": "United States",
        "data_available": "Examination, laboratory, dietary, and questionnaire data",
        "access": MIXED,
        "search_enabled": True,
    },
    {
        "id": "dhs",
        "name": "The DHS Program",
        "category": "public_health",
        "url": "https://dhsprogram.com/data/",
        "domains": ["dhsprogram.com", "api.dhsprogram.com"],
        "coverage": "Multiple countries",
        "data_available": "Health, nutrition, and demographic surveys",
        "access": MIXED,
        "search_enabled": True,
    },
    {
        "id": "icmr_nhrdr",
        "name": "ICMR National Health Research Data Repository",
        "category": "public_health",
        "url": "https://nhrdr.icmr.org.in/",
        "domains": ["nhrdr.icmr.org.in"],
        "coverage": "India",
        "data_available": "Health-research metadata and research insights",
        "access": "Metadata and request-based insights",
        "search_enabled": True,
    },
    # Evidence indexes and full-text collections
    {
        "id": "pubmed",
        "name": "PubMed",
        "category": "evidence_index",
        "url": "https://pubmed.ncbi.nlm.nih.gov/",
        "domains": ["pubmed.ncbi.nlm.nih.gov"],
        "coverage": "Biomedical literature",
        "data_available": "Studies, clinical trials, reviews, and meta-analyses",
        "access": "Free search and abstracts; full text varies",
        "search_enabled": True,
    },
    {
        "id": "pmc",
        "name": "PubMed Central (PMC)",
        "category": "evidence_index",
        "url": "https://pmc.ncbi.nlm.nih.gov/",
        "domains": ["pmc.ncbi.nlm.nih.gov"],
        "coverage": "Biomedical and life-science articles",
        "data_available": "Open full-text articles",
        "access": OPEN,
        "search_enabled": True,
    },
    {
        "id": "europe_pmc",
        "name": "Europe PMC",
        "category": "evidence_index",
        "url": "https://europepmc.org/",
        "domains": ["europepmc.org"],
        "coverage": "Biomedical literature",
        "data_available": "Articles, reviews, preprints, and links to supporting datasets",
        "access": "Free search; substantial open full text",
        "search_enabled": True,
    },
    {
        "id": "cochrane_library",
        "name": "Cochrane Library",
        "category": "evidence_index",
        "url": "https://www.cochranelibrary.com/",
        "domains": ["cochranelibrary.com"],
        "coverage": "Healthcare interventions",
        "data_available": "Systematic reviews",
        "access": "Abstracts and plain-language summaries are free; full text varies",
        "search_enabled": True,
    },
    {
        "id": "doaj",
        "name": "Directory of Open Access Journals (DOAJ)",
        "category": "evidence_index",
        "url": "https://doaj.org/",
        "domains": ["doaj.org"],
        "coverage": "Open-access journals",
        "data_available": "Open-access journal and article records",
        "access": OPEN,
        "search_enabled": True,
    },
    {
        "id": "scielo",
        "name": "SciELO",
        "category": "evidence_index",
        "url": "https://www.scielo.org/",
        "domains": ["scielo.org"],
        "coverage": "National and thematic journal collections",
        "data_available": "Open-access research including health sciences",
        "access": OPEN,
        "search_enabled": True,
    },
    # Guidelines and high-quality clinical sources
    {
        "id": "uspstf",
        "name": "U.S. Preventive Services Task Force",
        "category": "guideline",
        "url": "https://www.uspreventiveservicestaskforce.org/",
        "domains": ["uspreventiveservicestaskforce.org"],
        "coverage": "United States preventive care",
        "data_available": "Evidence reviews and screening recommendations",
        "access": OPEN,
        "search_enabled": True,
    },
    {
        "id": "nice",
        "name": "NICE",
        "category": "guideline",
        "url": "https://www.nice.org.uk/",
        "domains": ["nice.org.uk"],
        "coverage": "United Kingdom",
        "data_available": "Evidence-based clinical guidelines and technology guidance",
        "access": OPEN,
        "search_enabled": True,
    },
    {
        "id": "who_guidelines",
        "name": "World Health Organization",
        "category": "guideline",
        "url": "https://www.who.int/",
        "domains": ["who.int"],
        "coverage": "Global",
        "data_available": "Guidelines, recommendations, and public-health evidence",
        "access": OPEN,
        "search_enabled": True,
    },
    {
        "id": "ats",
        "name": "American Thoracic Society",
        "category": "guideline",
        "url": "https://www.thoracic.org/",
        "domains": ["thoracic.org", "atsjournals.org"],
        "coverage": "Respiratory medicine",
        "data_available": "Clinical practice guidelines and respiratory research",
        "access": MIXED,
        "search_enabled": True,
    },
    {
        "id": "ers",
        "name": "European Respiratory Society",
        "category": "guideline",
        "url": "https://www.ersnet.org/",
        "domains": ["ersnet.org", "ersjournals.com"],
        "coverage": "Respiratory medicine",
        "data_available": "Clinical guidelines and respiratory research",
        "access": MIXED,
        "search_enabled": True,
    },
    {
        "id": "rsna",
        "name": "Radiological Society of North America",
        "category": "guideline",
        "url": "https://www.rsna.org/",
        "domains": ["rsna.org", "radiology.rsna.org", "radiologyinfo.org"],
        "coverage": "Radiology and imaging",
        "data_available": "Imaging guidance and peer-reviewed research",
        "access": MIXED,
        "search_enabled": True,
    },
    # Open-access clinical journals
    {
        "id": "plos_medicine",
        "name": "PLOS Medicine",
        "category": "journal",
        "url": "https://journals.plos.org/plosmedicine/",
        "domains": ["journals.plos.org"],
        "coverage": "Clinical research, population health, and health policy",
        "data_available": "Peer-reviewed open-access articles",
        "access": OPEN,
        "search_enabled": True,
    },
    {
        "id": "bmc_medicine",
        "name": "BMC Medicine",
        "category": "journal",
        "url": "https://link.springer.com/journal/12916",
        "domains": ["link.springer.com"],
        "coverage": "General and translational medicine",
        "data_available": "Peer-reviewed open-access articles",
        "access": OPEN,
        "search_enabled": True,
    },
    {
        "id": "bmj_open",
        "name": "BMJ Open",
        "category": "journal",
        "url": "https://bmjopen.bmj.com/",
        "domains": ["bmjopen.bmj.com", "bmj.com"],
        "coverage": "Medical disciplines",
        "data_available": "Peer-reviewed open-access articles",
        "access": OPEN,
        "search_enabled": True,
    },
    {
        "id": "jama_network_open",
        "name": "JAMA Network Open",
        "category": "journal",
        "url": "https://jamanetwork.com/journals/jamanetworkopen",
        "domains": ["jamanetwork.com"],
        "coverage": "Clinical care, health policy, and global health",
        "data_available": "Peer-reviewed open-access articles",
        "access": OPEN,
        "search_enabled": True,
    },
    {
        "id": "eclinicalmedicine",
        "name": "eClinicalMedicine",
        "category": "journal",
        "url": "https://www.sciencedirect.com/journal/eclinicalmedicine",
        "domains": ["sciencedirect.com", "thelancet.com"],
        "coverage": "Clinical and public-health research",
        "data_available": "Trials, systematic reviews, and clinical studies",
        "access": OPEN,
        "search_enabled": True,
    },
    {
        "id": "nejm",
        "name": "New England Journal of Medicine",
        "category": "journal",
        "url": "https://www.nejm.org/",
        "domains": ["nejm.org"],
        "coverage": "Clinical medicine",
        "data_available": "Landmark trials, clinical research, and reviews",
        "access": MIXED,
        "search_enabled": True,
    },
    {
        "id": "thelancet",
        "name": "The Lancet",
        "category": "journal",
        "url": "https://www.thelancet.com/",
        "domains": ["thelancet.com"],
        "coverage": "Clinical medicine and global health",
        "data_available": "Clinical studies, trials, reviews, and public-health research",
        "access": MIXED,
        "search_enabled": True,
    },
    {
        "id": "nature_medicine",
        "name": "Nature Medicine",
        "category": "journal",
        "url": "https://www.nature.com/nm/",
        "domains": ["nature.com"],
        "coverage": "Translational and clinical medicine",
        "data_available": "Clinical and translational research articles",
        "access": MIXED,
        "search_enabled": True,
    },
]


def get_sources(
    *,
    category: str | None = None,
    search_enabled_only: bool = False,
) -> list[dict[str, object]]:
    """Return catalogue entries, optionally filtered without mutating them."""
    sources: Iterable[dict[str, object]] = MEDICAL_SOURCE_CATALOG
    if category:
        sources = (source for source in sources if source["category"] == category)
    if search_enabled_only:
        sources = (source for source in sources if source["search_enabled"])
    return [dict(source) for source in sources]


def source_domains(*source_ids: str) -> list[str]:
    """Return de-duplicated allowed domains for the requested catalogue IDs."""
    wanted = set(source_ids)
    domains: list[str] = []
    for source in MEDICAL_SOURCE_CATALOG:
        if source["id"] not in wanted:
            continue
        for domain in source["domains"]:  # type: ignore[index]
            if domain not in domains:
                domains.append(str(domain))
    return domains


def _bucket(*source_ids: str) -> list[str]:
    return source_domains(*source_ids)


# General clinical questions start with the three evidence groups below.  A
# question can add one specialised group, which keeps Fast mode bounded and
# avoids drowning a clinical answer in irrelevant dataset pages.
BASE_TRUSTED_BUCKETS: dict[str, list[str]] = {
    "Guidelines and health agencies": _bucket("uspstf", "nice", "who_guidelines", "ats", "ers", "rsna"),
    "Evidence indexes and full text": _bucket("pubmed", "pmc", "europe_pmc", "cochrane_library"),
    "Independent clinical journals": _bucket(
        "plos_medicine", "bmc_medicine", "bmj_open", "jama_network_open",
        "eclinicalmedicine", "nejm", "thelancet", "nature_medicine",
    ),
}

SPECIALISED_BUCKETS: dict[str, list[str]] = {
    "Imaging and imaging datasets": _bucket("tcia", "idc", "openneuro", "rsna"),
    "Genomics and molecular datasets": _bucket("geo", "dbgap", "gdc"),
    "Public-health data": _bucket("who_data", "cdc_open_data", "nhanes", "dhs", "icmr_nhrdr"),
    "Clinical-data discovery": _bucket("physionet", "amsterdamumcdb", "all_of_us", "uk_biobank"),
}

ROUTING_KEYWORDS: tuple[tuple[set[str], str], ...] = (
    ({"dicom", "ct", "mri", "imaging", "radiology", "hrct", "segmentation"}, "Imaging and imaging datasets"),
    ({"genomic", "genomics", "gene", "rna", "transcript", "gwas", "biomarker"}, "Genomics and molecular datasets"),
    ({"epidemiology", "prevalence", "incidence", "surveillance", "mortality rate", "population health", "india"}, "Public-health data"),
    ({"dataset", "ehr", "electronic health record", "icu", "physiological signal", "clinical data"}, "Clinical-data discovery"),
)


def build_trusted_retrieval_buckets(question: str) -> dict[str, list[str]]:
    """Select evidence-first domains and one relevant specialised source group.

    The result is suitable for Tavily's ``include_domains`` parameter.  It
    contains only public webpages for discovery/search; it never attempts to
    retrieve patient-level records or bypass a login, credential, or DUA.
    """
    buckets = {name: list(domains) for name, domains in BASE_TRUSTED_BUCKETS.items()}
    normalised_question = question.casefold()

    for keywords, bucket_name in ROUTING_KEYWORDS:
        # Word boundaries prevent short terms such as ``ct`` from matching
        # inside unrelated words such as ``colorectal``.
        if any(
            re.search(rf"\b{re.escape(keyword)}\b", normalised_question)
            for keyword in keywords
        ):
            buckets[bucket_name] = list(SPECIALISED_BUCKETS[bucket_name])
            break

    return buckets


def catalogue_summary() -> dict[str, int]:
    """Provide a small audit summary for a future Streamlit catalogue panel."""
    categories: dict[str, int] = {}
    for source in MEDICAL_SOURCE_CATALOG:
        category = str(source["category"])
        categories[category] = categories.get(category, 0) + 1
    return categories
