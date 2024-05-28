"""
class TestEasyOcrPdfReader(unittest.TestCase):
    def setUp(self):
        # load config
        base_dir = Path(__file__).parent.parent.parent
        config_file = os.path.join(base_dir, "src/pai_rag/config/settings.local.yaml")
        config = RagConfiguration.from_file(config_file).get_value()
        module_registry.init_modules(config)
        reader_config = config["data_reader"]
        self.directory_reader = SimpleDirectoryReader(
            input_dir="data/pdf_data",
            file_extractor={
                ".pdf": PaiPDFReader(
                    enable_image_ocr=reader_config.get("enable_image_ocr", False),
                    model_dir=reader_config.get("easyocr_model_dir", None),
                )
            },
        )

    def test_load_documents(self):
        # load documents
        self.documents = self.directory_reader.load_data()
        self.assertEqual(len(self.documents), 40)


if __name__ == "__main__":
    unittest.main()
"""
