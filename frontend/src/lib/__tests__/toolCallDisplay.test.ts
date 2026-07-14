import { describe, expect, it } from "vitest";
import { getToolCallDisplay } from "../toolCallDisplay";

describe("getToolCallDisplay", () => {
  it.each([
    ["shell", { command: "ls /opt/code" }, "tool.name.shell", "ls /opt/code"],
    [
      "code_interpreter",
      { language: "python", code: "print(1)\nprint(2)" },
      "tool.name.codeInterpreter",
      "python · print(1)",
    ],
    ["web_search", { query: "PAI-RAG" }, "tool.name.webSearch", "PAI-RAG"],
    [
      "web_fetch",
      { url: "https://example.com" },
      "tool.name.webFetch",
      "https://example.com",
    ],
    [
      "knowledge_search",
      { query: "向量检索" },
      "tool.name.knowledgeSearch",
      "向量检索",
    ],
    [
      "knowledge_find",
      { query: "ERR_42" },
      "tool.name.knowledgeFind",
      "ERR_42",
    ],
    [
      "knowledge_read",
      { document_id: "doc_1", chunk_id: "chk_1" },
      "tool.name.knowledgeRead",
      "doc_1",
    ],
    ["knowledge_list", {}, "tool.name.knowledgeList", undefined],
    ["current_datetime", {}, "tool.name.currentDatetime", undefined],
    ["load_skill", { skill_id: "skill.pdf" }, "tool.name.loadSkill", "skill.pdf"],
    [
      "enable_skill_for_agent",
      { skill_id: "skill.pdf", enabled: false },
      "tool.name.enableSkillForAgent",
      "skill.pdf",
    ],
    [
      "read_skill_resource",
      { skill_id: "skill.pdf", path: "references/a.md" },
      "tool.name.readSkillResource",
      "skill.pdf · references/a.md",
    ],
    [
      "install_skill",
      {
        source: {
          type: "git",
          url: "https://github.com/acme/skills",
          path: "pdf",
        },
      },
      "tool.name.installSkill",
      "git · https://github.com/acme/skills · pdf",
    ],
    [
      "publish_artifact",
      { name: "report.pdf", path: "/mnt/user/report.pdf" },
      "tool.name.publishArtifact",
      "report.pdf",
    ],
    [
      "spawn_subagent",
      { agent_id: "explore", task: "inspect the repository" },
      "tool.name.spawnSubagent",
      "explore · inspect the repository",
    ],
    [
      "read_handle",
      { handle: "store://tool/call_1" },
      "tool.name.readHandle",
      "store://tool/call_1",
    ],
  ])("formats %s", (name, args, labelKey, summary) => {
    expect(getToolCallDisplay(name, JSON.stringify(args))).toEqual({
      labelKey,
      summary,
    });
  });

  it("falls back from document_id to chunk_id", () => {
    expect(
      getToolCallDisplay("knowledge_read", '{"chunk_id":"chk_2"}').summary,
    ).toBe("chk_2");
  });

  it("normalizes multiline and repeated whitespace", () => {
    expect(
      getToolCallDisplay(
        "shell",
        '{"command":"git  status\\n --short"}',
      ).summary,
    ).toBe("git status --short");
  });

  it.each(["not-json", "[]", "null", "{}"])(
    "keeps a known label without a usable summary for %s",
    (raw) => {
      expect(getToolCallDisplay("shell", raw)).toEqual({
        labelKey: "tool.name.shell",
        summary: undefined,
      });
    },
  );

  it("preserves unknown tools through an empty policy result", () => {
    expect(
      getToolCallDisplay("custom_tool", '{"query":"secret"}'),
    ).toEqual({});
  });
});
