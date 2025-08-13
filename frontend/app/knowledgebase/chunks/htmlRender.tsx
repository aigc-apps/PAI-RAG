import parse, { Element, HTMLReactParserOptions } from "html-react-parser";
import React from "react";

const isTableContent = (text: string) => {
  const trimmedText = text?.trim();
  return trimmedText?.includes("<table>");
};
const extractTableBlocks = (text: string): string[] => {
  const tableBlocks: string[] = [];
  const prefix = "<table";
  const suffix = "</table>";
  let startIndex = 0;

  while (startIndex < text.length) {
    const blockStart = text.indexOf(prefix, startIndex);
    if (blockStart === -1) break; // No more <table> found

    const blockEnd = text.indexOf(suffix, blockStart);
    if (blockEnd === -1) break; // Unmatched <table>, stop processing

    // Extract the full <table>...</table> block (包含 <table> 和 </table> 标签)
    const tableBlock = text.substring(blockStart, blockEnd + suffix.length);
    tableBlocks.push(tableBlock);

    // Move the search start index past the end of the current block
    startIndex = blockEnd + suffix.length;
  }

  return tableBlocks;
};

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

export const htmlRender = (text: string) => {
  const trimmedText = text.trim();

  // Case 1: chunk.text 本身就是一个独立的 <table> 元素
  // 检查是否以 <table 开头并以 </table> 结尾
  if (
    trimmedText.startsWith("<table") &&
    trimmedText.endsWith("</table>") &&
    trimmedText.indexOf("</table>") === trimmedText.lastIndexOf("</table>")
  ) {
    try {
      return parse(trimmedText, tableOptions);
    } catch (error) {
      console.error("Error parsing standalone <table> element:", error);
      return <pre>{text}</pre>;
    }
  }
  // Case 2: chunk.text 是包含一个或多个 <table> 块的文本
  else if (isTableContent(text)) {
    try {
      // 提取所有 <table>...</table> 块
      const tableBlocks = extractTableBlocks(text);

      if (tableBlocks.length > 0) {
        return (
          <>
            {text.substring(0, text.indexOf("<table"))}
            {tableBlocks.map((tableBlock, index) => {
              return (
                <React.Fragment key={index}>
                  {parse(tableBlock, tableOptions)}
                </React.Fragment>
              );
            })}
            {text.substring(text.lastIndexOf("</table>") + "</table>".length)}
          </>
        );
      } else {
        return text;
      }
    } catch (error) {
      console.error("Error parsing embedded <table> blocks:", error);
      return <pre>{text}</pre>;
    }
  }
  // Case 3: 不包含 <table>，按普通文本渲染
  else {
    return text;
  }
};
