import json
from pai_rag.integrations.readers.pai_pdf_reader import PaiPDFReader

json_str = r"""{
    "pdf_info": [
        {
            "preproc_blocks": [
                {
                    "type": "title",
                    "bbox": [
                        59,
                        67,
                        196,
                        89
                    ],
                    "lines": [
                        {
                            "bbox": [
                                57,
                                70,
                                72,
                                85
                            ],
                            "spans": [
                                {
                                    "bbox": [
                                        57,
                                        70,
                                        72,
                                        85
                                    ],
                                    "score": 0.75,
                                    "content": "\\leftarrow",
                                    "type": "inline_equation"
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        ]
}"""

md_str = """#  $\leftarrow$  """


def test_post_process_multi_level_headings():
    pdf_process = PaiPDFReader()
    json_content = json.loads(json_str)
    md_content_escape = pdf_process.post_process_multi_level_headings(
        json_content, md_str
    )
    assert md_content_escape == "#  $\leftarrow$  "
