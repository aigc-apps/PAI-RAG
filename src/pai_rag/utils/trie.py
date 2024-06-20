from typing import Dict


class TrieNode:
    def __init__(self, char, is_word=False):
        self.char: str = char
        self.is_word: bool = is_word
        self.children: Dict[str, TrieNode] = {}


class TrieTree:
    def __init__(self, word_list):
        self.root = TrieNode("###")

        self.build_tree(word_list)

    def build_tree(self, word_list):
        for word in word_list:
            current_node = self.root
            for w in word:
                if w not in current_node.children:
                    current_node.children[w] = TrieNode(w)
                current_node = current_node.children[w]
            current_node.is_word = True

    def match(self, word):
        current_node = self.root
        for w in word:
            if w not in current_node.children:
                return False
            current_node = current_node.children[w]

        return current_node.is_word


if __name__ == "__main__":
    tree = TrieTree(["abc", "Her", "She", "He", "Hereby"])
    assert tree.match("Her")
    assert tree.match("Hereby")
    assert tree.match("She")
    assert tree.match("abc")
    assert tree.match("He")
    assert not tree.match("Here")
