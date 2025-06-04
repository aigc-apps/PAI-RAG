"use client";
import { GlobeIcon } from "@radix-ui/react-icons";
import type { FC } from "react";
import { makeAssistantToolUI } from "@assistant-ui/react";
import React, { useState, useEffect } from "react";
import { CarIcon } from "lucide-react"; // 可以使用你喜欢的图标库
import { Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";

export type MapsGeoArgs = {
  address: string;
  city: string;
};

export type MapsGeoAddressInfo = {
  country: string;
  province: string;
  city: string;
  district?: string;
  location: string;
  level: string;
  adcode?: string;
};

type RawResult = {
  content: Array<{
    type: "text";
    text: string;
  }>;
  isError: boolean;
};
function parseGeoResult(result: RawResult): MapsGeoAddressInfo[] | null {
  if (!result || result.isError) {
    console.error("接口返回错误或为空");
    return null;
  }

  const textContent = result.content.find((item) => item.type === "text");
  if (!textContent) {
    console.warn("未找到文本类型的内容");
    return null;
  }

  try {
    const parsedJson = JSON.parse(textContent.text);
    const addresses = parsedJson.return || [];

    return addresses.map((addr: any) => ({
      country: addr.country,
      province: addr.province,
      city: addr.city,
      district: addr.district,
      location: addr.location,
      level: addr.level,
      adcode: addr.adcode,
    }));
  } catch (e) {
    console.error("JSON 解析失败", e);
    return null;
  }
}

export const MapsGeoToolUI = makeAssistantToolUI<MapsGeoArgs, RawResult>({
  toolName: "maps_geo",
  render: ({ args, status, result }) => {
    if (!result) {
      return null;
    }
    const addresses = parseGeoResult(result);

    console.log("MapsGeoToolUI 参数:", args);
    console.log("MapsGeoToolUI 状态:", status);
    console.log("MapsGeoToolUI 结果:", addresses);

    if (status.type == "running") {
      return (
        <div className="flex items-center gap-2 text-sm font-medium text-gray-500">
          <GlobeIcon className="h-4 w-4 animate-pulse" />
          <span>正在查询地理信息...</span>
        </div>
      );
    }
    if (!addresses || addresses.length === 0) {
      return (
        <div className="flex items-center gap-2 text-sm font-medium text-red-500">
          <GlobeIcon className="h-4 w-4" />
          <span>未找到相关地理信息</span>
        </div>
      );
    }

    return (
      <div className="flex flex-col gap-1 text-sm font-medium">
        <div className="flex items-center gap-2 text-blue-700">
          <GlobeIcon className="h-4 w-4" />
          <span>地图信息查询结果：</span>
        </div>
        {addresses.map((addr, index) => (
          <div key={index} className="ml-6 border-l pl-2 text-xs text-gray-600">
            <div>国家：{addr.country}</div>
            <div>省份：{addr.province}</div>
            <div>城市：{addr.city}</div>
            {addr.district && <div>区/县：{addr.district}</div>}
            <div>坐标：{addr.location}</div>
            <div>级别：{addr.level}</div>
            {addr.adcode && <div>行政区划代码：{addr.adcode}</div>}
          </div>
        ))}
      </div>
    );
  },
});

export type MapsDirectionDrivingArgs = {
  origin: string;
  destination: string;
};
// 路线规划的每一步
export type RouteStep = {
  instruction: string; // 导航指令
  road?: string; // 道路名称
  distance: number; // 距离（米）
  orientation?: string; // 方向
  duration: number; // 耗时（秒）
};

// 单条路径（可能有多个备选路径）
export type RoutePath = {
  distance: number;
  duration: number;
  steps: RouteStep[];
};

// 完整路线数据
export type RouteResult = {
  origin: string; // 起点坐标 "120.210792,30.246026"
  destination: string; // 终点坐标 "121.473667,31.230525"
  paths: RoutePath[]; // 所有路径（这里只取第一条展示）
};

function parseRouteResult(result: RawResult | undefined): RouteResult | null {
  if (!result || result.isError) {
    console.error("接口返回错误或为空");
    return null;
  }

  const textContent = result.content.find((item) => item.type === "text");
  if (!textContent) {
    console.warn("未找到文本类型的内容");
    return null;
  }

  try {
    const parsedJson = JSON.parse(textContent.text);
    const routeData = parsedJson.route;

    return {
      origin: routeData.origin,
      destination: routeData.destination,
      paths: routeData.paths.map((path: any) => ({
        distance: parseInt(path.distance),
        duration: parseInt(path.duration),
        steps: path.steps.map((step: any) => ({
          instruction: step.instruction,
          road: step.road,
          distance: parseInt(step.distance),
          orientation: step.orientation,
          duration: parseInt(step.duration),
        })),
      })),
    };
  } catch (e) {
    console.error("JSON 解析失败", e);
    return null;
  }
}

export const MapsDirectionDrivingToolUI = makeAssistantToolUI<
  MapsDirectionDrivingArgs,
  RawResult
>({
  toolName: "maps_direction_driving",
  render: ({ args, status, result }) => {
    if (!result) {
      return null;
    }
    const [route, setRoute] = useState<RouteResult | null>(null);
    useEffect(() => {
      const parsed = parseRouteResult(result);
      setRoute(parsed);
    }, [result]);

    if (status.type == "running") {
      return (
        <div className="flex items-center gap-2 text-sm font-medium text-gray-500">
          <GlobeIcon className="h-4 w-4 animate-pulse" />
          <span>正在规划路线...</span>
        </div>
      );
    }
    if (!route) {
      return (
        <div className="flex items-center gap-2 text-sm font-medium text-red-500">
          <CarIcon className="h-4 w-4" />
          <span>未能获取路线信息</span>
        </div>
      );
    }

    const firstPath = route.paths[0];

    return (
      <div className="p-3 border rounded-md bg-white shadow-sm max-w-xl mx-auto">
        <div className="flex items-center gap-2 mb-2 text-blue-700">
          <CarIcon className="h-5 w-5" />
          <h3 className="font-bold">路线规划结果</h3>
        </div>

        <div className="mb-2">
          <strong>从：</strong> {args.origin}（{route.origin}）
        </div>
        <div className="mb-2">
          <strong>到：</strong> {args.destination}（{route.destination}）
        </div>
        <div className="mb-2 text-green-600">
          <strong>总距离：</strong> {(firstPath.distance / 1000).toFixed(2)}{" "}
          千米
        </div>
        <div className="mb-2 text-purple-600">
          <strong>预计耗时：</strong> {(firstPath.duration / 60).toFixed(0)}{" "}
          分钟
        </div>

        <h4 className="font-semibold mt-4 mb-1">详细路线指引：</h4>
        <ol className="list-decimal pl-5 space-y-1 text-sm">
          {firstPath.steps.map((step, index) => (
            <li key={index}>
              <strong>({step.distance} 米)</strong> {step.instruction}
              {step.road && (
                <span className="text-gray-500 ml-1">
                  （道路：{step.road}）
                </span>
              )}
            </li>
          ))}
        </ol>
      </div>
    );
  },
});

/* Search Web Tool UI */

export type SearchWebArgs = {
  query: string;
};

type SearchWebResult = {
  result: {
    text: string;
    metadata: {
      source: string;
      file_url: string;
      file_name: string;
      host_name: string;
      host_logo: string;
      publish_time: string;
    };
    score: string;
  }[];
};

export const SearchWebToolUI = makeAssistantToolUI<
  SearchWebArgs,
  SearchWebResult
>({
  toolName: "search_web",
  render: ({ args, status, result }) => {
    if (!result) {
      return null;
    }
    console.log("SearchWebToolUI 参数:", args);
    console.log("SearchWebToolUI 状态:", status);
    console.log("SearchWebToolUI 结果:", result);
    if (status.type == "running") {
      return (
        <div className="flex items-center gap-2 text-sm font-medium text-gray-500">
          <GlobeIcon className="h-4 w-4 animate-pulse" />
          <span>正在搜索网页...{args.query}</span>
        </div>
      );
    }
    return (
      <div className="thinking-box rounded-md p-1 bg-muted/50 border-l-4 border-primary cursor-pointer hover:bg-muted/70 transition-colors">
        <Sheet>
          <SheetTrigger asChild>
            <Button
              variant="link"
              className="flex items-center gap-2 px-4 text-blue-800"
            >
              {" "}
              <Search className="size-4" /> 完成网页搜索: {args.query}{" "}
              (点击查看结果){" "}
            </Button>
          </SheetTrigger>
          <SheetContent side="right">
            <SheetHeader>
              <SheetTitle>网页搜索结果 · {result?.result.length}</SheetTitle>
              <SheetDescription>{args.query}</SheetDescription>
            </SheetHeader>
            <div className="flex flex-col gap-2 border-t pt-2 pb-2 overflow-y-auto">
              <div className="pl-6 pr-2">
                {result?.result.map((item, index) => (
                  <div
                    key={index}
                    className="text-sm p-3 hover:bg-muted/50 rounded-md transition-colors"
                  >
                    <div className="flex flex-col gap-1 p-1 hover:bg-muted/50 rounded-md transition-colors">
                      {/* Logo与标题行 */}
                      <div className="flex items-center gap-1">
                        <div className="flex-shrink-0 w-8 h-8 rounded-md bg-muted flex items-center justify-center">
                          <img
                            src={item.metadata["host_logo"]}
                            alt={item.metadata["host_name"]}
                            className="w-5 h-5 object-cover rounded-sm"
                          />
                        </div>

                        {/* 标题链接 */}
                        <a
                          href={item.metadata["file_url"]}
                          className="font-medium text-foreground hover:text-primary hover:underline truncate transition-colors"
                        >
                          {item.metadata["file_name"]}
                        </a>
                      </div>

                      {/* 内容区域 */}
                      <p className="text-muted-foreground text-xs mt-1 leading-relaxed line-clamp-3">
                        {item.text}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </SheetContent>
        </Sheet>
      </div>
    );
  },
});

/* Think Tool UI */

export type ThinkArgs = {
  thought: string;
};

type ThinkResult = {
  type: string;
  text: string;
  thoughts_count: number;
};

export const ThinkToolUI = makeAssistantToolUI<ThinkArgs, ThinkResult>({
  toolName: "think",
  render: ({ args, status, result }) => {
    if (!result) {
      return null;
    }
    console.log("think 结果:", result);
    return (
      <div
        className="thinking-box rounded-md p-4 bg-muted/50 border-l-4 border-primary cursor-pointer hover:bg-muted/70 transition-colors"
        role="button"
      >
        {/* 条件渲染头部内容：仅当 thoughts_count 为 1 时显示 */}
        {result?.thoughts_count === 1 && (
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xl" aria-hidden="true">
              🧠
            </span>
            <span className="font-semibold">思考中... </span>
          </div>
        )}
        <div className="text-sm mt-2">{args.thought}</div>
      </div>
    );
  },
});

const ToolUIWrapper: FC = () => {
  return (
    <>
      {/* <MapsGeoToolUI /> */}
      {/* <MapsDirectionDrivingToolUI /> */}
      <SearchWebToolUI />
      <ThinkToolUI />
    </>
  );
};

export default ToolUIWrapper;
