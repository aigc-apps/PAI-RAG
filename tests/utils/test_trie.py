from pai_rag.utils.trie import TrieTree


def test_trietree():
    word_list = ["Water", "What", "Watt", "Wow", "Go", "Goose", "Way"]
    tree = TrieTree(word_list)

    for w in word_list:
        assert tree.match(w)

    for no_w in ["Wa", "W", "what", "water", "Goos", "G"]:
        assert not tree.match(no_w)


def test_trietree_special_characters():
    word_list = [
        "Water",
        "What",
        "Watt",
        "Wow",
        "Go",
        "Goose",
        "Way",
        "，",
        "！",
        "!",
        "&&",
    ]
    tree = TrieTree(word_list)

    for w in word_list:
        assert tree.match(w)
