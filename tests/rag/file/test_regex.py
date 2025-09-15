from pairag.file.readers.markdown_reader import MARKDOWN_IMAGE_PATTERN, HTML_IMAGE_PATTERN


TEST_MARKDOWN_CONTENT = """
# Title

This is a test markdown file.

图片1:
![Image1](./image1.png)

图片2:
![Image2](www.baidu.com/image2.png)

图片3
![Image3](http://www.baidu.com/image3.png)


图片3
![Image4](https://1.com/image4.png)
"""

TEST_HTML_CONTENT = """
<html>
<head>
<title>Title</title>
</head>
<body>
<h1>Title</h1>
<p>This is a test html file.</p>
<img src="./image1.png" alt="Image1">
<img src="www.baidu.com/image2.png" alt="Image2">
<img src="http://www.baidu.com/image3.png" alt="Image3">
<img src="https://1.com/image4.png" alt="Image4">
<img src="https://1.com/image5.png">
</body>
</html>
"""

def test_markdown_regex():
    image_matches = MARKDOWN_IMAGE_PATTERN.finditer(TEST_MARKDOWN_CONTENT)
    for match in image_matches:
        assert (
            (match[0] == '![Image3](http://www.baidu.com/image3.png)' and match[1] == 'http://www.baidu.com/image3.png')
            or
            (match[0] == '![Image4](https://1.com/image4.png)' and match[1] == 'https://1.com/image4.png')
        ), "Markdown url is not regonized correctly: {match[0]} - {match[1]}."



def test_html_regex():
    image_matches = HTML_IMAGE_PATTERN.finditer(TEST_MARKDOWN_CONTENT)
    for match in image_matches:
        assert (
            (match[0] == '<img src="http://www.baidu.com/image3.png" alt="Image3">' and match[1] == 'http://www.baidu.com/image3.png')
            or
            (match[0] == '<img src="https://1.com/image4.png" alt="Image4">' and match[1] == 'https://1.com/image4.png')
            or
            (match[0] == '<img src="https://1.com/image5.png">' and match[1] == 'https://1.com/image5.png')
        ), "Html url is not regonized correctly: {match[0]} - {match[1]}."

