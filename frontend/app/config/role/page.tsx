"use client";

import React, { useState, useEffect } from "react";
import {
  TrashIcon,
  SettingsIcon,
  Edit,
  EyeIcon,
  EyeOffIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import * as Toast from "@radix-ui/react-toast";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Checkbox } from "@/components/ui/checkbox";
import { v4 as uuidv4 } from "uuid";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import RolePage from "./role";
import UserRolePage from "./user_role";

export interface Permission {
  id: string;
  name: string;
  role_id: string;
}

export default function RoleConfigPage() {
  return (
    <div id="access-control">
      <div className="flex flex-col h-screen p-6 space-y-6">
        {/* 顶部标题栏 */}
        <div className="flex justify-between items-center h-1/10">
          <h1 className="text-2xl font-bold">权限控制</h1>
        </div>

        {/* 卡片容器 */}
        <div className="h-4/5">
          <div className="">
            <Tabs defaultValue="roles">
              <TabsList className="py-4 bg-muted rounded-lg flex-none">
                <TabsTrigger value="roles" className="p-4">
                  角色配置
                </TabsTrigger>
                <TabsTrigger value="userroles" className="p-4">
                  用户-角色关系
                </TabsTrigger>
              </TabsList>
              <TabsContent value="roles" className="py-4">
                <RolePage />
              </TabsContent>
              <TabsContent value="userroles" className="py-4">
                <UserRolePage />
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    </div>
  );
}
