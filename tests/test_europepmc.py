"""Europe PMC client: record parsing, preprint flag, retraction, year policy.

The HTTP layer is mocked at ``researchclaw.literature.europepmc_client.fetch_json``
so every test is offline and asserts on the payload shape the live API returns
(captured from ``rest/search?resultType=core``).
"""

from __future__ import annotations

import pytest

from researchclaw.literature.europepmc_client import (
    _build_query,
    _clean,
    _retracted,
    search_europepmc,
)

_MED_RECORD = {
    "id": "39532094",
    "source": "MED",
    "pmid": "39532094",
    "doi": "10.1038/s41586-024-08100-w",
    "title": "Transcriptome-scale <i>RNA</i>-targeting CRISPR screens",
    "authorString": "Smith J, Doe A, et al.",
    "authorList": {
        "author": [
            {"fullName": "Jane Smith", "affiliation": "Broad Institute"},
            {"firstName": "A.", "lastName": "Doe"},
        ]
    },
    "journalInfo": {"journal": {"title": "Nature"}},
    "pubYear": "2024",
    "abstractText": "A <i>CRISPR</i> screen &amp; analysis.",
    "citedByCount": 12,
    "pubTypeList": {"pubType": ["Journal Article"]},
    "fullTextUrlList": {
        "fullTextUrl": [
            {
                "documentStyle": "pdf",
                "availability": "Open access",
                "url": "https://europepmc.org/pdf/PMC1.pdf",
            }
        ]
    },
}

_PPR_RECORD = {
    "id": "PPR1354099",
    "source": "PPR",
    "title": "Prevention of <i>Klebsiella pneumoniae</i> resistance",
    "authorString": "Solo A.",
    "pubYear": "2026",
    "firstPublicationDate": "2026-10-29",
    "bookOrReportDetails": {"publisher": "bioRxiv", "yearOfPublication": 2026},
    "pubTypeList": {"pubType": ["Preprint"]},
    "citedByCount": 0,
}

_RETRACTED_RECORD = {
    "id": "41136797",
    "source": "MED",
    "pmid": "41136797",
    "doi": "10.1007/s10142-026-01952-7",
    "title": "Artificial intelligence-driven epigenetic CRISPR therapeutic",
    "pubYear": "2024",
    "pubTypeList": {
        "pubType": ["Meta-Analysis", "Retracted Publication", "Journal Article"]
    },
    "commentCorrectionList": {
        "commentCorrection": [{"type": "Retraction in", "id": "42332124"}]
    },
}


def _payload(records: list[dict]) -> dict:
    return {"hitCount": len(records), "resultList": {"result": records}}


class TestParsing:
    def test_medline_record_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "researchclaw.literature.europepmc_client.fetch_json",
            lambda *a, **k: _payload([_MED_RECORD]),
        )
        paper = search_europepmc("crispr", limit=5)[0]
        assert paper.source == "europepmc"
        assert paper.paper_id == "pmid:39532094"
        assert paper.pmid == "39532094"
        assert paper.doi == "10.1038/s41586-024-08100-w"
        assert paper.venue == "Nature"
        assert paper.year == 2024
        assert paper.citation_count == 12
        assert paper.is_preprint is False
        assert paper.retracted is False
        assert paper.pub_types == ("Journal Article",)
        assert paper.pdf_url == "https://europepmc.org/pdf/PMC1.pdf"
        # DOI first: the reference links at the publisher, not at a
        # syndication page that needs the source code to resolve.
        assert paper.url == "https://doi.org/10.1038/s41586-024-08100-w"

    def test_inline_markup_is_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "researchclaw.literature.europepmc_client.fetch_json",
            lambda *a, **k: _payload([_MED_RECORD]),
        )
        paper = search_europepmc("crispr", limit=5)[0]
        assert "<i>" not in paper.title
        assert "CRISPR" in paper.title
        assert "&amp;" not in paper.abstract
        assert "&" in paper.abstract

    def test_authors_fall_back_to_author_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "researchclaw.literature.europepmc_client.fetch_json",
            lambda *a, **k: _payload([_PPR_RECORD]),
        )
        paper = search_europepmc("klebsiella", limit=5)[0]
        assert len(paper.authors) == 1
        assert paper.authors[0].name == "Solo A"

    def test_author_list_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "researchclaw.literature.europepmc_client.fetch_json",
            lambda *a, **k: _payload([_MED_RECORD]),
        )
        paper = search_europepmc("crispr", limit=5)[0]
        assert [a.name for a in paper.authors] == ["Jane Smith", "A. Doe"]
        assert paper.authors[0].affiliation == "Broad Institute"


class TestPreprints:
    def test_preprint_flag_and_venue(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "researchclaw.literature.europepmc_client.fetch_json",
            lambda *a, **k: _payload([_PPR_RECORD]),
        )
        paper = search_europepmc("klebsiella", limit=5)[0]
        assert paper.is_preprint is True
        assert paper.venue == "bioRxiv"

    def test_preprint_without_author_list_has_no_authors_when_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = dict(_PPR_RECORD)
        record["authorString"] = ""
        monkeypatch.setattr(
            "researchclaw.literature.europepmc_client.fetch_json",
            lambda *a, **k: _payload([record]),
        )
        assert search_europepmc("x")[0].authors == ()


class TestRetraction:
    def test_retracted_publication_type(self) -> None:
        assert _retracted(_RETRACTED_RECORD) is True

    def test_retraction_in_comment_correction(self) -> None:
        record = {"pubTypeList": {"pubType": ["Journal Article"]},
                  "commentCorrectionList": {"commentCorrection": [{"type": "Retraction in"}]}}
        assert _retracted(record) is True

    def test_retraction_notice_is_not_the_retracted_paper(self) -> None:
        """The notice is citable (you cite it to warn readers), so not flagged."""
        record = {
            "pubTypeList": {"pubType": ["Retraction of Publication"]},
            "commentCorrectionList": {"commentCorrection": [{"type": "Retraction of"}]},
        }
        assert _retracted(record) is False

    def test_clean_record_is_not_retracted(self) -> None:
        assert _retracted(_MED_RECORD) is False

    def test_retracted_flag_reaches_paper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "researchclaw.literature.europepmc_client.fetch_json",
            lambda *a, **k: _payload([_RETRACTED_RECORD]),
        )
        assert search_europepmc("crispr")[0].retracted is True


class TestYearPolicy:
    def test_year_filter_leaves_a_year_of_slack(self) -> None:
        """FIRST_PDATE is the first appearance, the displayed year is pubYear.

        A paper first posted in Dec 2019 and published in a 2020 issue must not
        be filtered out server-side, so the floor is one year below the caller's
        and the local check (``test_old_records_are_dropped_client_side``)
        enforces the exact year.
        """
        query = _build_query("crispr", 2020)
        assert "FIRST_PDATE:[2019-01-01 TO 3000-12-31]" in query

    def test_no_year_filter_when_year_min_zero(self) -> None:
        assert "FIRST_PDATE" not in _build_query("crispr", 0)

    def test_old_records_are_dropped_client_side(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = dict(_MED_RECORD, pubYear="2015")
        monkeypatch.setattr(
            "researchclaw.literature.europepmc_client.fetch_json",
            lambda *a, **k: _payload([record]),
        )
        assert search_europepmc("crispr", year_min=2020) == []

    def test_posting_date_fills_a_missing_year(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = dict(_MED_RECORD, pubYear="", firstPublicationDate="2021-03-04")
        monkeypatch.setattr(
            "researchclaw.literature.europepmc_client.fetch_json",
            lambda *a, **k: _payload([record]),
        )
        assert search_europepmc("crispr")[0].year == 2021

    def test_record_without_year_is_kept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = dict(_MED_RECORD, pubYear="")
        monkeypatch.setattr(
            "researchclaw.literature.europepmc_client.fetch_json",
            lambda *a, **k: _payload([record]),
        )
        assert len(search_europepmc("crispr", year_min=2020)) == 1


class TestContactAddress:
    def test_email_goes_into_the_user_agent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """EBI asks for a contact address; the adapter forwards the OpenAlex one."""
        seen: dict[str, dict[str, str]] = {}

        def fake(url: str, **kwargs: object) -> dict:
            seen["headers"] = kwargs["headers"]  # type: ignore[assignment]
            return _payload([])

        monkeypatch.setattr(
            "researchclaw.literature.europepmc_client.fetch_json", fake
        )
        search_europepmc("crispr", email="me@example.org")
        assert "mailto:me@example.org" in seen["headers"]["User-Agent"]
        search_europepmc("crispr")
        assert "mailto" not in seen["headers"]["User-Agent"]


class TestDegradation:
    def test_network_failure_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: object, **kwargs: object) -> dict:
            raise OSError("connection reset")

        monkeypatch.setattr(
            "researchclaw.literature.europepmc_client.fetch_json", boom
        )
        assert search_europepmc("crispr") == []

    def test_record_without_title_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "researchclaw.literature.europepmc_client.fetch_json",
            lambda *a, **k: _payload([{"id": "1", "title": ""}, _MED_RECORD]),
        )
        papers = search_europepmc("crispr")
        assert len(papers) == 1
        assert papers[0].pmid == "39532094"

    def test_limit_is_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        urls: list[str] = []

        def fake(url: str, **kwargs: object) -> dict:
            urls.append(url)
            return _payload([])

        monkeypatch.setattr(
            "researchclaw.literature.europepmc_client.fetch_json", fake
        )
        search_europepmc("x", limit=99_999)
        assert "pageSize=1000" in urls[0]

    def test_clean_handles_empty_values(self) -> None:
        assert _clean(None) == ""
        assert _clean("") == ""

class TestTextAndFlags:
    def test_clean_keeps_comparison_operators(self) -> None:
        from researchclaw.literature.europepmc_client import _clean

        assert _clean("Effect for p<0.01 and n>100") == "Effect for p<0.01 and n>100"

    def test_clean_strips_inline_markup(self) -> None:
        from researchclaw.literature.europepmc_client import _clean

        assert _clean("<i>E. coli</i> &amp; <sup>13</sup>C") == "E. coli & 13 C"

    def test_partial_retraction_is_flagged(self) -> None:
        from researchclaw.literature.europepmc_client import _retracted

        payload = {
            "commentCorrectionList": {
                "commentCorrection": [{"type": "Partial Retraction in"}]
            }
        }
        assert _retracted(payload) is True

    def test_expression_of_concern_is_a_flag_not_a_retraction(self) -> None:
        from researchclaw.literature.europepmc_client import _concern, _retracted

        payload = {"pubTypeList": {"pubType": ["Expression of Concern"]}}
        assert _concern(payload) is True
        assert _retracted(payload) is False

    def test_unparseable_payload_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from researchclaw.literature import europepmc_client

        monkeypatch.setattr(
            europepmc_client, "fetch_json", lambda *_a, **_kw: {"resultList": {"result": ["bad"]}}
        )
        assert europepmc_client.search_europepmc("q") == []



def test_browser_only_pdf_links_are_not_advertised() -> None:
    """Europe PMC answers 403 to programmatic PDF fetches."""
    from researchclaw.literature.europepmc_client import _pdf_url

    payload = {
        "fullTextUrlList": {
            "fullTextUrl": [
                {
                    "documentStyle": "pdf",
                    "url": "https://europepmc.org/articles/PMC1?pdf=render",
                },
                {
                    "documentStyle": "pdf",
                    "url": "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC1",
                },
                {"documentStyle": "pdf", "url": "https://example.org/paper.pdf"},
            ]
        }
    }
    assert _pdf_url(payload) == "https://example.org/paper.pdf"
    assert _pdf_url({"fullTextUrlList": {"fullTextUrl": [
        {"documentStyle": "pdf", "url": "https://europepmc.org/articles/PMC1?pdf=render"}
    ]}}) == ""
