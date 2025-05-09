"use client";
import { GlobeIcon } from "@radix-ui/react-icons";
import { z } from "zod";
import type { FC, ReactNode } from "react";
import { makeAssistantToolUI } from "@assistant-ui/react";
import React, { useState, useEffect } from "react";
import { CarIcon } from "lucide-react"; // 可以使用你喜欢的图标库
import { ToolCallContentPartComponent } from "@assistant-ui/react";
import { CheckIcon, ChevronDownIcon, ChevronUpIcon } from "lucide-react";
import { Button } from "../ui/button";

// type WebSearchArgs = {
//     query: string;
// };

// type WebSearchResult = {
//     searchResults: {
//         title: string;
//         description: string;
//         url: string;
//     }[];
// };

// const WebSearchToolUI = makeAssistantToolUI<WebSearchArgs, WebSearchResult>({
//     toolName: "web_search",
//     render: ({ args, status, result }) => {
//         return (
//             <div className="rounded-md border bg-muted p-4 space-y-3">
//                 <div className="flex items-center gap-2 text-sm font-medium">
//                     <GlobeIcon className="h-4 w-4" />
//                     <span>
//                         Searching the web for:{" "}
//                         <span className="font-semibold text-blue-600">{args.query}</span>
//                     </span>
//                 </div>
//                 <div className="space-y-2 pl-6">
//                     {result?.searchResults.map((item, index) => (
//                         <div key={index} className="text-sm">
//                             <a
//                                 href={item.url}
//                                 target="_blank"
//                                 rel="noopener noreferrer"
//                                 className="text-blue-700 hover:underline font-medium"
//                             >
//                                 {item.title}
//                             </a>
//                             <p className="text-muted-foreground text-xs mt-0.5">
//                                 {item.description}
//                             </p>
//                         </div>
//                     ))}
//                 </div>
//             </div>
//         );
//     },
// });

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
    description: string;
    score: string;
  }[];
};

export const SearchWebToolUI = makeAssistantToolUI<SearchWebArgs, SearchWebResult>({
  toolName: "search_web",
  render: ({ args, status, result }) => {
    if (!result) {
      return null;
    }
    const [isCollapsed, setIsCollapsed] = useState(true);
    console.log("MapsGeoToolUI 参数:", args);
    console.log("MapsGeoToolUI 状态:", status);
    console.log("MapsGeoToolUI 结果:", result);
    if (status.type == "running") {
      return (
        <div className="flex items-center gap-2 text-sm font-medium text-gray-500">
          <GlobeIcon className="h-4 w-4 animate-pulse" />
          <span>正在搜索网页...{args.query}</span>
        </div>
      );
    }

    return (
              <div className="rounded-md border bg-muted p-4 space-y-3">
                <div className="flex items-center gap-2 px-4">
                  <CheckIcon className="size-4" />
                  <p className="">
                    正在搜索网页<b>{args.query}</b>
                  </p>
                  <div className="flex-grow" />
                  <Button onClick={() => setIsCollapsed(!isCollapsed)}>
                    {isCollapsed ? <ChevronUpIcon /> : <ChevronDownIcon />}
                  </Button>
                </div>
                {!isCollapsed && (
                  <div className="flex flex-col gap-2 border-t pt-2">
                    <div className="space-y-2 pl-6">
                      {result?.result.map((item, index) => (
                          <div key={index} className="text-sm">
                              <p className="text-muted-foreground text-xs mt-0.5">
                                  {item.text}
                              </p>
                          </div>
                      ))}
                    </div> 
                  </div>
                )}
              </div>
          );
    
  },
});

const ToolUIWrapper: FC = () => {
  return (
    <>
      {/* <MapsGeoToolUI /> */}
      {/* <MapsDirectionDrivingToolUI /> */}
      < SearchWebToolUI />
    </>
  );
};

export default ToolUIWrapper;
