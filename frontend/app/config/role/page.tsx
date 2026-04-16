'use client';

import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import RolePage from './role';
import UserRolePage from './user_role';
import { useI18n } from '@/app/providers/i18n';

export interface Permission {
  id: string;
  name: string;
  role_id: string;
}

export default function RoleConfigPage() {
  const { t } = useI18n();

  return (
    <div id="access-control">
      <div className="flex flex-col h-screen p-6 space-y-6">
        <div className="flex justify-between items-center h-1/10">
          <h1 className="page-title">{t('config.role.title')}</h1>
        </div>

        <div className="h-4/5">
          <div className="">
            <Tabs defaultValue="roles">
              <TabsList className="py-4 tabs-modern flex-none">
                <TabsTrigger value="roles" className="p-4">
                  {t('config.role.roles')}
                </TabsTrigger>
                <TabsTrigger value="userroles" className="p-4">
                  {t('config.role.users')}
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
