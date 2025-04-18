import pytest
from pathlib import Path
from nidhinetra.core.ingest import extract_text_from_pdf, ingest_pdf_to_chroma, search_documents


@pytest.mark.skipif(
    not Path(r"C:\Users\kashi\Yawaris\nidhinetra-clean\nidhinetra\data\returns\LARVYLLC2024TaxReturn.pdf").exists(),
    reason="Sample return file not found"
)
def test_ingest_and_search():
    """End-to-end test: ingest a real PDF and query Chroma vector DB."""
    print("\n⚙️ Ingesting real tax document into Chroma...")
    count = ingest_pdf_to_chroma(
        pdf_path=r"C:\Users\kashi\Yawaris\nidhinetra-clean\nidhinetra\data\returns\LARVYLLC2024TaxReturn.pdf",
        namespace="llc_returns",
        user_id="user_dhirendra",
        year="2024"
    )
    assert int(count) > 0

    print("\n🔍 Querying ingested data...")
    results = search_documents(
        query="What is the total income?",
        namespace="llc_returns",
        user_id="user_dhirendra",
        year="2024"
    )
    assert results, "Expected non-empty results"
    for doc, score in results:
        print(f"Score: {score:.2f}\n{doc.page_content}\n")


def test_extract_text_from_pdf():
    """Unit test: Extract text from sample PDF."""
    path = Path(r"C:\Users\kashi\Yawaris\nidhinetra-clean\nidhinetra\data\returns\LARVYLLC2024TaxReturn.pdf")
    if not path.exists():
        pytest.skip("PDF file not found")

    text = extract_text_from_pdf(path)
    assert "Total income" in text or "Income" in text
    print("\n📄 Extracted Text Snippet:\n", text[:500])
