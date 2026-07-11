import { beforeEach, describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Markdown } from "../Markdown";
import { useI18nStore } from "../../i18n";

beforeEach(() => {
  useI18nStore.getState().setLang("en");
});

describe("Markdown", () => {
  it("renders headings and bold", () => {
    render(<Markdown content={"# Title\n\nsome **bold** text"} />);
    expect(screen.getByRole("heading", { name: "Title" })).toBeInTheDocument();
    expect(screen.getByText("bold")).toBeInTheDocument();
  });

  it("renders text fences as output blocks", () => {
    render(<Markdown content={"```text\nhello world!\n```"} />);
    expect(screen.getByText("Output")).toBeInTheDocument();
    expect(screen.getByText("hello world!")).toBeInTheDocument();
  });
});
