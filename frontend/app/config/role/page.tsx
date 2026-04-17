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
      <Tabs defaultValue="roles">
        <TabsList className="tabs-modern flex-none">
          <TabsTrigger value="roles" className="px-4">
            {t('config.role.roles')}
          </TabsTrigger>
          <TabsTrigger value="userroles" className="px-4">
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
  );
}
