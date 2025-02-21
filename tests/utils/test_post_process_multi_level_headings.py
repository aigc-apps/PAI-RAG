import pytest
import os


@pytest.mark.skipif(
    os.getenv("SKIP_GPU_TESTS", "false") == "true",
    reason="Need to execute in a CUDA environment.",
)
def test_post_process_multi_level_headings():
    title_list = [
        ("title_1", 6),
        ("title_2", 10),
        ("title_3", 8),
        ("title_4", 7),
        ("title_5", 14),
    ]
    from pai_rag.integrations.readers.pai_pdf_reader import PaiPDFReader

    pdf_process = PaiPDFReader()
    new_title_list = pdf_process.post_process_multi_level_headings(title_list, 0, 0)
    assert new_title_list == [
        "### title_1",
        "## title_2",
        "### title_3",
        "### title_4",
        "# title_5",
    ]
