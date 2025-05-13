import { Calendar, Home, Inbox, Search, Settings, Bot, Wrench } from "lucide-react";
import React, { useState, useEffect } from "react";
import {
    Sidebar,
    SidebarContent,
    SidebarGroup,
    SidebarGroupContent,
    SidebarGroupLabel,
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem,
} from "@/components/ui/sidebar";
import { Button } from "./ui/button";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
    DialogFooter
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import LLMConfig from "@/components/setting/llm-config"
import MCPConfig from "@/components/setting/mcp-config"

// Menu items.
const items = [
    {
        title: "Home",
        url: "#",
        icon: Home,
    },
    {
        title: "Inbox",
        url: "#",
        icon: Inbox,
    },
    {
        title: "Calendar",
        url: "#",
        icon: Calendar,
    },
    {
        title: "Search",
        url: "#",
        icon: Search,
    },
    {
        title: "Settings",
        url: "#",
        icon: Settings,
    },
]

export function AppSidebar() {
    const [llmConfig, setLlmConfig] = useState([{
        id: 1,
        source: "",
        model_name: "",
        api_key: "",
        max_context: 1024,
    }]);
    const [mcpConfig, setMCPConfig] = useState([
        { id: 1, name: "", url: "", type: "", active: false },
    ]);

    const [isLoading, setIsLoading] = useState(true);
    // 从 API 加载配置
    useEffect(() => {
        const fetchConfig = async () => {
            setIsLoading(true);
            try {
                const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8097;
                console.log("config-page BACKEND_PORT", port);
                const res = await fetch(`http://localhost:${port}/api/configs`);
                if (!res.ok) throw new Error("获取配置失败");
                const data = await res.json();
                setLlmConfig(data["llm_config"]);
                setMCPConfig(data["mcp_config"])
            } catch (err) {
                console.error(err);
                // 可选：设置默认值或提示用户
            } finally {
                setIsLoading(false);
            }
        };

        fetchConfig();
    }, []);
    const handleSave = async () => {
        // 保存逻辑（如 localStorage 或 API 请求）
        try {
            const port = process.env.NEXT_PUBLIC_BACKEND_PORT || 8097;
            console.log("config-page handleSave BACKEND_PORT", port);
            const res = await fetch(`http://localhost:${port}/api/configs`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ llm_config: llmConfig, mcp_config: mcpConfig }),
            });
            if (!res.ok) throw new Error("保存失败");
            alert("配置已保存至本地文件");
        } catch (err) {
            console.error(err);
            alert("保存失败，请重试");
        }
    };
    return (
        <Sidebar side='left'>
            <SidebarContent>
                <SidebarGroup>
                    <SidebarGroupLabel>系统设置</SidebarGroupLabel>
                    <SidebarGroupContent className="gap-4">
                        <SidebarMenu className="gap-4">
                            <Dialog>
                                <DialogTrigger>
                                    <SidebarMenuItem>
                                        <SidebarMenuButton asChild>
                                            <a href="#">
                                                <Bot />
                                                <span>LLM配置</span>
                                            </a>
                                        </SidebarMenuButton>
                                    </SidebarMenuItem>
                                </DialogTrigger>
                                <DialogContent className="sm:max-w-[425px] sm:max-h-[525px] overflow-y-auto">
                                    <DialogHeader>
                                        <DialogTitle>LLM配置信息</DialogTitle>
                                        <DialogDescription>
                                            新增或删除新的LLM连接，完成后点击保存。
                                        </DialogDescription>
                                    </DialogHeader>
                                    <LLMConfig
                                        config={llmConfig}
                                        onChange={(updatedConfig) => setLlmConfig(updatedConfig)}
                                    />
                                    <DialogFooter>
                                        <Button onClick={handleSave} type="submit">保存配置</Button>
                                    </DialogFooter>
                                </DialogContent>
                            </Dialog>
                            <Dialog>
                                <DialogTrigger>
                                    <SidebarMenuItem>
                                        <SidebarMenuButton asChild>
                                            <a href="#">
                                                <Wrench />
                                                <span>MCP配置</span>
                                            </a>
                                        </SidebarMenuButton>
                                    </SidebarMenuItem>
                                </DialogTrigger>
                                <DialogContent className="sm:max-w-[425px] sm:max-h-[525px] overflow-y-auto">
                                    <DialogHeader>
                                        <DialogTitle>MCP Server配置信息</DialogTitle>
                                        <DialogDescription>
                                            新增或删除新的MCP Server，完成后点击保存。
                                        </DialogDescription>
                                    </DialogHeader>
                                    <MCPConfig
                                        config={mcpConfig}
                                        onChange={(updatedConfig) => setMCPConfig(updatedConfig)}
                                    />
                                    <DialogFooter>
                                        <Button onClick={handleSave} type="submit">保存配置</Button>
                                    </DialogFooter>
                                </DialogContent>
                            </Dialog>
                            <Dialog>
                                <DialogTrigger>
                                    <SidebarMenuItem>
                                        <SidebarMenuButton asChild>
                                            <a href="#">
                                                <Search />
                                                <span>搜索配置</span>
                                            </a>
                                        </SidebarMenuButton>
                                    </SidebarMenuItem>
                                </DialogTrigger>
                                <DialogContent className="sm:max-w-[425px] sm:max-h-[525px] overflow-y-auto">
                                    <DialogHeader>
                                        <DialogTitle>网络搜索配置</DialogTitle>
                                        <DialogDescription>
                                            配置阿里云通用搜索功能，完成后点击保存。
                                        </DialogDescription>
                                    </DialogHeader>
                                        <div className="grid gap-4">
                                            <div className="grid grid-cols-2 gap-4">
                                                <Label>AccessKey ID</Label>
                                                <Input placeholder="请输入阿里云账号的AccessKey ID" />
                                            </div>
                                            <div className="grid grid-cols-2 gap-4">
                                                <Label>AccessKey Secret</Label>
                                                <Input placeholder="请输入AccessKey Secret" />
                                            </div>
                                        </div>
                                    <DialogFooter>
                                        <Button onClick={handleSave} type="submit">保存配置</Button>
                                    </DialogFooter>
                                </DialogContent>
                            </Dialog>
                        </SidebarMenu>
                        {/* <ConfigPage/> */}
                    </SidebarGroupContent>
                </SidebarGroup>
            </SidebarContent>
        </Sidebar>
    )
}
