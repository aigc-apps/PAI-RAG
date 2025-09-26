'use client';

import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import RolePage from './role';
import UserRolePage from './user_role';

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
          <h1 className="text-xl font-medium">权限控制</h1>
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
