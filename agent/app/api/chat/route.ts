import { openai, createOpenAI } from "@ai-sdk/openai";
import {
  experimental_createMCPClient,
  streamText,
  tool,
  generateText,
} from "ai";

export const runtime = "edge";
export const maxDuration = 30;

// system: "1. 你是一个 agent，请持续调用工具直至完美完成用户的任务，停止调用工具后，系统会自动交还控制权给用户。请只有在确定问题已解决后才终止调用工具。\n 2. 请善加利用你的工具收集相关信息，绝对不要猜测或编造答案。\n 3. 「思考和规划」是一个系统工具，在每次调用其他任务工具之前，你必须**首先调用思考和规划工具**：针对用户的任务详细思考和规划，如果用户需要生成一些报告类的内容，你需要先列出一个大纲，并对之前工具调用的结果进行深入反思（如有），输出的顺序是thought, plan, action, thoughtNumber。\n - 「思考和规划」工具不会获取新信息或更改数据库，只会将你的想法保存到记忆中。\n - 思考完成之后不需要等待工具返回，你可以继续调用其他任务工具，你一次可以调用多个任务工具。\n - 任务工具调用完成之后，你可以停止输出，系统会把工具调用结果给你，你必须再次调用思考和规划工具，然后继续调用任务工具，如此循环，直到完成用户的任务。"

interface MCPServerConfig {
  id: number;
  name: string;
  url: string;
  type: "sse";
}
const today = new Date().toISOString().split("T")[0];

const SYSTEM_PROMPT = `
当前系统时间：${today}

1. 你是一个 agent，请持续调用工具直至完美完成用户的任务，停止调用工具后，系统会自动交还控制权给用户。请只有在确定问题已解决后才终止调用工具。
2. 请善加利用你的工具收集相关信息，绝对不要猜测或编造答案。
3. 在每次调用任务工具之前，
  - 你必须**首先思考和规划**：针对用户的任务进行详细思考，并给出你对拆解后任务的规划，同时需要对之前工具调用的结果进行深入反思并继续规划（如有）。
  - 思考完成之后不需要等待工具返回，你可以继续调用其他任务工具，你一次可以调用多个任务工具。
  - 任务工具调用完成之后，你可以停止输出，系统会把工具调用结果给你，你必须再次思考和规划，然后继续调用任务工具，如此循环，直到完美地完成用户的任务。
`;

function getModelInstance(modelName: string, modelSource: string, apiKey: string) {

  if (modelSource === "openai") {
    const openaiModel = createOpenAI({
      apiKey: apiKey,
      baseURL: "https://api.openai.com/v1",
      compatibility: 'strict',
    }); // 自定义 OpenAI 封装

    return openaiModel(modelName);
  } else if (modelSource === "qwen") {
    const model = createOpenAI({
      apiKey: apiKey,
      baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }); // 自定义 Qwen 封装
    return model(modelName);
  }
  throw new Error(`Unsupported model provider: ${modelSource}`);
}

export async function POST(req: Request) {
  const { messages, system, tools } = await req.json();
  const model_name = req.headers.get("X-Model-Name");
  
  const api_key = req.headers.get("X-Api-Key");
  const model_source = req.headers.get("X-Model-Source");
  console.log("use model_name", model_name);
  console.log("use model_source", model_source);
  let mcpServers = [];
  // 动态获取 MCP 配置（示例 URL）
  try {
    // 发起对 /api/configs 的请求
    const response = await fetch("http://localhost:8097/api/configs");
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const configData = await response.json();
    console.log("configData", configData);
    mcpServers = (configData["mcp_config"] || []).filter(
      (item: { active: boolean }) => item.active === true,
    ); // 提取 MCP 服务列表
  } catch (fetchError) {
    console.error("Failed to fetch MCP server configurations:", fetchError);
  }
  try {
    // 并行初始化 MCP 客户端
    const clientPromises = mcpServers.map(async (server: MCPServerConfig) => {
      try {
        const client = await experimental_createMCPClient({
          transport: {
            type: server.type,
            url: server.url,
          },
        });
        return client.tools();
      } catch (err) {
        console.error(`❌ MCP 客户端 ${server.name} 初始化失败:`, err);
        return {};
      }
    });

    // 批量获取工具集并合并
    const toolSets = await Promise.all(clientPromises);
    const mergedTools = Object.assign({}, ...toolSets);
    console.log("SYSTEM_PROMPT:", SYSTEM_PROMPT);
    const modelInstance = getModelInstance(model_name as string, model_source as string, api_key as string);
    console.log("modelInstance:", modelInstance);
    const response = await streamText({
      model: modelInstance,
      messages: messages,
      system: SYSTEM_PROMPT,
      tools: mergedTools,
      maxSteps: 20,
      toolChoice: "auto",
    });

    return response.toDataStreamResponse();
  } catch (error) {
    return new Response("Internal Server Error", { status: 500 });
  }
}
