// htmlRenderer.tsx
import parse, { Element, HTMLReactParserOptions } from "html-react-parser";
import React from "react"; // 确保导入 React

// --- 1. 通用的 isHtmlContent 函数 ---
// 检查文本是否包含 <html> 标签（无论是作为开头还是嵌入其中）
const isHtmlContent = (text: string) => {
  const trimmedText = text?.trim();
  // 只要包含 <html> 就认为是包含 HTML 内容
  return trimmedText?.includes("<html>");
};

// --- 2. 提取 <body> 内容的函数 (用于纯 HTML 文档) ---
const extractBodyContent = (htmlString: string) => {
  const bodyStart = htmlString.indexOf("<body");
  const bodyEnd = htmlString.lastIndexOf("</body>");

  if (bodyStart !== -1 && bodyEnd !== -1) {
    const bodyOpenEnd = htmlString.indexOf(">", bodyStart);
    if (bodyOpenEnd !== -1) {
      return htmlString.substring(bodyOpenEnd + 1, bodyEnd).trim();
    }
  }
  return htmlString; // Fallback
};

// --- 3. 提取所有 <html>...</html> 块的函数 (用于嵌入 HTML 的文本) ---
const extractHtmlBlocks = (text: string): string[] => {
  const htmlBlocks: string[] = [];
  const prefix = "<html>";
  const suffix = "</html>";
  let startIndex = 0;

  while (startIndex < text.length) {
    const blockStart = text.indexOf(prefix, startIndex);
    if (blockStart === -1) break; // No more <html> found

    const blockEnd = text.indexOf(suffix, blockStart);
    if (blockEnd === -1) break; // Unmatched <html>, stop processing

    // Extract the full <html>...</html> block
    const htmlBlock = text.substring(blockStart, blockEnd + suffix.length);
    htmlBlocks.push(htmlBlock);

    // Move the search start index past the end of the current block
    startIndex = blockEnd + suffix.length;
  }

  return htmlBlocks;
};

// --- 4. 定义 HTMLReactParserOptions (为表格和单元格添加样式) ---
const tableOptions: HTMLReactParserOptions = {
  replace(domNode) {
    // 处理 <table> 标签
    if (domNode.type === "tag" && domNode.name === "table") {
      const element = domNode as Element;
      const existingTableClass =
        element.attribs.className || element.attribs.class || "";
      // 添加边框、边框合并、居中、上下边距
      const tableBorderClasses =
        "border border-collapse border-gray-500 mx-auto my-2";
      const updatedTableClassName = existingTableClass
        ? `${existingTableClass} ${tableBorderClasses}`
        : tableBorderClasses;
      element.attribs.className = updatedTableClassName;
      delete element.attribs.class;
      return undefined;
    }

    // 处理 <td> 和 <th> 标签
    if (
      domNode.type === "tag" &&
      (domNode.name === "td" || domNode.name === "th")
    ) {
      const element = domNode as Element;
      const existingCellClass =
        element.attribs.className || element.attribs.class || "";
      // 添加内边距和边框
      const cellBorderClasses = "border border-gray-500 p-2";
      const updatedCellClassName = existingCellClass
        ? `${existingCellClass} ${cellBorderClasses}`
        : cellBorderClasses;
      element.attribs.className = updatedCellClassName;
      delete element.attribs.class;
      return undefined;
    }

    // 对于其他元素，使用默认行为
    return undefined;
  },
};

// --- 5. 主要的 HTML 渲染函数 ---
export const htmlRender = (text: string) => {
  const trimmedText = text.trim();

  // 情况 1: chunk.text 本身就是一个完整的 HTML 文档
  if (trimmedText.startsWith("<html>")) {
    try {
      // 提取 <body> 内容并解析
      const bodyContent = extractBodyContent(trimmedText);
      return parse(bodyContent, tableOptions);
    } catch (error) {
      console.error("Error parsing standalone HTML document:", error);
      // 如果解析失败，回退到显示原始文本
      return <pre>{text}</pre>;
    }
  }
  // 情况 2: chunk.text 是包含 HTML 块的文本
  else if (isHtmlContent(text)) {
    try {
      // 提取所有 <html>...</html> 块
      const htmlBlocks = extractHtmlBlocks(text);

      if (htmlBlocks.length > 0) {
        // 对每个 HTML 块提取 <body> 内容并解析，然后将结果组合起来
        return (
          <>
            {/* 渲染 HTML 块之前的部分文本 */}
            {text.substring(0, text.indexOf("<html>"))}
            {/* 渲染每个解析后的 HTML 块 */}
            {htmlBlocks.map((htmlBlock, index) => {
              const bodyContent = extractBodyContent(htmlBlock);
              // 使用 React Fragment 包裹，以防 parse 返回多个根元素
              // key 用于 React 列表渲染
              return (
                <React.Fragment key={index}>
                  {parse(bodyContent, tableOptions)}
                </React.Fragment>
              );
            })}
            {/* 渲染最后一个 HTML 块之后的部分文本 */}
            {text.substring(text.lastIndexOf("</html>") + "</html>".length)}
          </>
        );
      } else {
        return text;
      }
    } catch (error) {
      console.error("Error parsing embedded HTML blocks:", error);
      // 如果解析嵌入的 HTML 失败，回退到显示原始文本
      return <pre>{text}</pre>;
    }
  }
  // 情况 3: 不包含 HTML，按普通文本渲染
  else {
    return text;
  }
};
