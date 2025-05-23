import json


def test_markdown_parser():
    from pai_rag.file.nodeparsers.pai.pai_markdown_parser import (
        MarkdownNodeParser,
    )
    from pai_rag.file.readers.pai.pai_data_reader import (
        PaiDataReader,
        DataReaderConfig,
    )
    from pai_rag.utils.download_models import ModelScopeDownloader

    reader_config = DataReaderConfig()
    directory_reader = PaiDataReader(reader_config=reader_config)

    input_dir = "tests/testdata/data/md_data"
    ModelScopeDownloader().load_rag_models()
    documents = directory_reader.load_data(file_path_or_directory=input_dir)
    md_node_parser = MarkdownNodeParser(enable_multimodal=False)
    splitted_nodes = []
    for doc_node in documents:
        splitted_nodes.extend(md_node_parser.get_nodes_from_documents([doc_node]))

    text_list = [node.text for node in splitted_nodes]

    with open(
        "tests/testdata/data/json_data/pai_document.json", "r", encoding="utf-8"
    ) as file:
        chunk_text = json.load(file)

    assert text_list == chunk_text
    assert len(splitted_nodes) == 10
