import type { MessageKey } from "../i18n";

type ArgumentsObject = Record<string, unknown>;
type SummaryFormatter = (args: ArgumentsObject) => string | undefined;

interface ToolDisplayConfig {
  labelKey: MessageKey;
  summary?: SummaryFormatter;
}

export interface ToolCallDisplay {
  labelKey?: MessageKey;
  summary?: string;
}

function normalize(value: string): string | undefined {
  const normalized = value.replace(/\s+/g, " ").trim();
  return normalized || undefined;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" ? normalize(value) : undefined;
}

function objectValue(value: unknown): ArgumentsObject | undefined {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? (value as ArgumentsObject)
    : undefined;
}

function joinValues(
  ...values: Array<string | undefined>
): string | undefined {
  const present = values.filter((value): value is string => Boolean(value));
  return present.length > 0 ? present.join(" · ") : undefined;
}

const CONFIG: Record<string, ToolDisplayConfig> = {
  shell: {
    labelKey: "tool.name.shell",
    summary: (args) => stringValue(args.command),
  },
  code_interpreter: {
    labelKey: "tool.name.codeInterpreter",
    summary: (args) =>
      joinValues(
        stringValue(args.language),
        typeof args.code === "string"
          ? normalize(args.code.split(/\r?\n/, 1)[0])
          : undefined,
      ),
  },
  web_search: {
    labelKey: "tool.name.webSearch",
    summary: (args) => stringValue(args.query),
  },
  web_fetch: {
    labelKey: "tool.name.webFetch",
    summary: (args) => stringValue(args.url),
  },
  knowledge_search: {
    labelKey: "tool.name.knowledgeSearch",
    summary: (args) => stringValue(args.query),
  },
  knowledge_find: {
    labelKey: "tool.name.knowledgeFind",
    summary: (args) => stringValue(args.query),
  },
  knowledge_read: {
    labelKey: "tool.name.knowledgeRead",
    summary: (args) =>
      stringValue(args.document_id) ?? stringValue(args.chunk_id),
  },
  knowledge_list: { labelKey: "tool.name.knowledgeList" },
  current_datetime: { labelKey: "tool.name.currentDatetime" },
  load_skill: {
    labelKey: "tool.name.loadSkill",
    summary: (args) => stringValue(args.skill_id),
  },
  enable_skill_for_agent: {
    labelKey: "tool.name.enableSkillForAgent",
    summary: (args) => stringValue(args.skill_id),
  },
  read_skill_resource: {
    labelKey: "tool.name.readSkillResource",
    summary: (args) =>
      joinValues(stringValue(args.skill_id), stringValue(args.path)),
  },
  install_skill: {
    labelKey: "tool.name.installSkill",
    summary: (args) => {
      const source = objectValue(args.source);
      return source
        ? joinValues(
            stringValue(source.type),
            stringValue(source.url),
            stringValue(source.path),
            stringValue(source.upload_id),
          )
        : undefined;
    },
  },
  publish_artifact: {
    labelKey: "tool.name.publishArtifact",
    summary: (args) => stringValue(args.name) ?? stringValue(args.path),
  },
  spawn_subagent: {
    labelKey: "tool.name.spawnSubagent",
    summary: (args) =>
      joinValues(stringValue(args.agent_id), stringValue(args.task)),
  },
  read_handle: {
    labelKey: "tool.name.readHandle",
    summary: (args) => stringValue(args.handle),
  },
};

function parseArguments(rawArguments: string): ArgumentsObject | undefined {
  try {
    return objectValue(JSON.parse(rawArguments));
  } catch {
    return undefined;
  }
}

export function getToolCallDisplay(
  toolName: string,
  rawArguments: string,
): ToolCallDisplay {
  const config = CONFIG[toolName];
  if (!config) return {};

  const args = parseArguments(rawArguments);
  let summary: string | undefined;
  if (args && config.summary) {
    try {
      summary = config.summary(args);
    } catch {
      summary = undefined;
    }
  }

  return { labelKey: config.labelKey, summary };
}
