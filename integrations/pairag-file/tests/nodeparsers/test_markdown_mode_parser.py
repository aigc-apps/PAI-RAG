import json
from pairag.file.store.file_store_helper import create_file_store_from_env
from pairag.file.nodeparsers.file_parser import FileParser
from pairag.file.models.file_item import FileItem


def test_markdown_parser():


    file_store = create_file_store_from_env()

    input_file = "tests/testdata/md_data/pai_document.md"
    file_parser = FileParser(file_store=file_store)
    file_item = FileItem.from_path(
                file_path=input_file,
                kb_id="kb_id",
            )
    documents, nodes = file_parser.parse(file_item,is_attachment=False)
    text_list = [node.text for node in nodes]
    with open(
        "tests/testdata/json_data/pai_document.json", "r", encoding="utf-8"
    ) as file:
        chunk_text = json.load(file)

    assert text_list == chunk_text
    assert len(nodes) == 4