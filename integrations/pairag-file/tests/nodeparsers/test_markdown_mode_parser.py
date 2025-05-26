import json


def test_markdown_parser():
    from pairag.file.nodeparsers.pai.pai_markdown_parser import (
        MarkdownNodeParser,
    )
    from pairag.file.readers.pai.pai_data_reader import (
        PaiDataReader,
        DataReaderConfig,
    )

    reader_config = DataReaderConfig()
    directory_reader = PaiDataReader(reader_config=reader_config)

    input_dir = "tests/testdata/md_data"
    documents = directory_reader.load_data(file_path_or_directory=input_dir)
    md_node_parser = MarkdownNodeParser(enable_multimodal=False)
    splitted_nodes = []
    for doc_node in documents:
        splitted_nodes.extend(md_node_parser.get_nodes_from_documents([doc_node]))

    text_list = [node.text for node in splitted_nodes]

    with open(
        "tests/testdata/json_data/pai_document.json", "r", encoding="utf-8"
    ) as file:
        chunk_text = json.load(file)

    assert text_list == chunk_text
    assert len(splitted_nodes) == 10
