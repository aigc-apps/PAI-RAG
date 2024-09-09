import json
import re

json_str = r"""{
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
                        }"""

md_str = """#  $\leftarrow$  """


def test_json_to_md():
    json_content = json.loads(json_str)
    title_text = ""
    for span in json_content["spans"]:
        if span["type"] == "inline_equation":
            span["content"] = " $" + span["content"] + "$ "
        title_text += span["content"]
    title_text_escape = title_text.replace("\\", "\\\\")
    new_title_escape = "##" + " " + title_text_escape
    md_content_escape = re.sub(re.escape(md_str), new_title_escape, md_str)
    assert title_text == " $\leftarrow$ "
    assert md_content_escape == "##  $\leftarrow$ "
