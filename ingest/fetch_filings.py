"""Downloads SEC 10-K filings, resolving filing URLs from SEC EDGAR's public
JSON APIs at runtime rather than hardcoded accession numbers or paths.

SEC EDGAR requires a descriptive User-Agent under its fair-access policy.
"""
import os

import requests

import llm_client.config as config

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_PAGINATED_SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{accession_no_dashes}/{primary_document}"


def _headers() -> dict:
    return {"User-Agent": config.SEC_EDGAR_USER_AGENT}


def _get_json(url: str) -> dict:
    resp = requests.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def _download_bytes(url: str) -> bytes:
    resp = requests.get(url, headers=_headers(), timeout=60)
    resp.raise_for_status()
    return resp.content


def resolve_cik(ticker: str) -> str:
    tickers = _get_json(_TICKERS_URL)
    for entry in tickers.values():
        if entry["ticker"].upper() == ticker.upper():
            return f"{entry['cik_str']:010d}"
    raise ValueError(f"no CIK found for ticker {ticker!r}")


def _find_10k(filings: dict, fiscal_year: int) -> tuple[str, str] | None:
    for form, report_date, accession, primary_doc in zip(
        filings["form"], filings["reportDate"], filings["accessionNumber"], filings["primaryDocument"]
    ):
        if form == "10-K" and report_date.startswith(str(fiscal_year)):
            return accession, primary_doc
    return None


async def fetch_filing(ticker: str, fiscal_year: int, document_id: str, raw_dir: str = "data/raw") -> str:
    os.makedirs(raw_dir, exist_ok=True)
    cached_path = os.path.join(raw_dir, f"{document_id}.html")
    if os.path.exists(cached_path):
        return cached_path

    cik = resolve_cik(ticker)
    submissions = _get_json(_SUBMISSIONS_URL.format(cik=cik))

    match = _find_10k(submissions["filings"]["recent"], fiscal_year)

    # "recent" is a fixed-size window, not a fixed time window, so a frequent
    # filer's 10-K can drop out of it within a couple of years. Fall back to
    # the paginated archives, newest page first.
    if match is None:
        for page in submissions["filings"].get("files", []):
            paginated = _get_json(_PAGINATED_SUBMISSIONS_URL.format(name=page["name"]))
            match = _find_10k(paginated, fiscal_year)
            if match is not None:
                break

    if match is None:
        raise ValueError(f"no 10-K found for {ticker} fiscal year {fiscal_year}")

    accession, primary_doc = match
    accession_no_dashes = accession.replace("-", "")
    cik_no_zeros = str(int(cik))
    url = _ARCHIVE_URL.format(
        cik_no_zeros=cik_no_zeros,
        accession_no_dashes=accession_no_dashes,
        primary_document=primary_doc,
    )
    content = _download_bytes(url)
    with open(cached_path, "wb") as handle:
        handle.write(content)
    return cached_path
