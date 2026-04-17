'use client';

import React, { useState, useEffect } from 'react';
import {
  TrashIcon,
  Edit,
  AlertCircleIcon,
  CheckCircle,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardFooter,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { EmbeddingModelDialog } from '@/app/config/model/embedding/modelDialog';
import { PaginationComponent } from '@/components/customized/pagination/pagination-component';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableFooter,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { set } from 'date-fns';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

export interface Role {
  id: string;
  name: string;
  description: string;
}

const newRole = {
  name: '',
  description: '',
};

export default function RolePage() {
  const { t } = useI18n();
  const [editRole, setEditRole] = useState(newRole);
  const [roles, setRoles] = useState<Role[]>([]);
  const [modelloading, setModelLoading] = useState(true);
  const [modelerror, setModelError] = useState('');
  const [errorMsg, setErrorMsg] = useState('');

  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const modelSizePerPage = 8;

  const [isEditOpen, setIsEditOpen] = useState(false);
  const { tenantFetch } = useTenantFetch();

  useEffect(() => {
    const fetchModelConfigs = async () => {
      try {
        const res = await tenantFetch(
          `/api/config/roles?page=${page}&size=${modelSizePerPage}`,
        );
        if (!res.ok) throw new Error(t('config.role.fetchRoleListFailed'));
        const json_data = await res.json();
        const data = json_data.data.items;
        setRoles(data);
        setTotalPages(json_data.data.pages);
      } catch (err: any) {
        setModelError(err || t('config.role.loadFailed'));
      } finally {
        setModelLoading(false);
      }
    };
    fetchModelConfigs();
  }, [page, roles.length, isEditOpen]);

  const handleDelete = async (role_id: string) => {
    try {
      const res = await tenantFetch(`/api/config/roles/${role_id}`, {
        method: 'DELETE',
      });
      if (!res.ok) throw new Error(t('config.role.fetchRoleListFailed'));
      const json_data = await res.json();
      const data = json_data.data;
      console.log(t('config.role.deleteRoleSuccess'), data);
      setRoles((prev) => prev.filter((role) => role.id !== role_id));
      setIsEditOpen(false);
    } catch (err: any) {
      setModelError(err || t('config.role.loadFailed'));
    } finally {
      setModelLoading(false);
    }
  };

  const handleAddRole = async () => {
    try {
      const res = await tenantFetch(`/api/config/roles`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editRole),
      });
      if (!res.ok) throw new Error(t('config.role.fetchRoleListFailed'));
      const json_data = await res.json();
      const data = json_data.data;
      console.log(t('config.role.createRoleSuccess'), data);
      setRoles([...roles, data]);
      setIsEditOpen(false);
      setModelError('');
    } catch (err: any) {
      setModelError(err || t('config.role.loadFailed'));
    } finally {
      setModelLoading(false);
    }
  };

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > totalPages) return;
    setPage(newPage);
  };

  return (
    <div id="role">
      <div className="flex items-center gap-12">
        <div className="font-medium text-md">{t('config.role.roleConfigTable')}</div>
        <Popover open={isEditOpen} onOpenChange={setIsEditOpen}>
          <PopoverTrigger asChild>
            <Button variant="default">{t('config.role.addRole')}</Button>
          </PopoverTrigger>
          <PopoverContent className="w-120">
            <div className="grid gap-4">
              <div className="space-y-2">
                <h4 className="leading-none font-medium">{t('config.role.createRole')}</h4>
              </div>
              <div className="grid gap-2">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="rolename">{t('config.role.roleName')}</Label>
                  <Input
                    id="rolename"
                    placeholder={t('config.role.roleNamePlaceholder')}
                    value={editRole.name}
                    onChange={(e) => {
                      setEditRole({ ...editRole, name: e.target.value });
                    }}
                    className="col-span-3 h-8"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="roledesc">{t('config.role.roleDescription')}</Label>
                  <Textarea
                    id="roledesc"
                    placeholder={t('config.role.roleDescPlaceholder')}
                    value={editRole.description}
                    onChange={(e) => {
                      setEditRole({ ...editRole, description: e.target.value });
                    }}
                    className="col-span-3 h-20"
                  />
                </div>
                {modelerror && (
                  <p className="text-red-500 truncate">{modelerror}</p>
                )}
                <Button variant="secondary" onClick={handleAddRole}>
                  {t('common.save')}
                </Button>
              </div>
            </div>
          </PopoverContent>
        </Popover>
      </div>
      <div className="w-full">
        <Table className="w-full">
          <TableHeader className="w-full">
            <TableRow>
              <TableHead className="w-1/5">{t('config.role.roleName')}</TableHead>
              <TableHead className="w-3/5">{t('config.role.roleDescription')}</TableHead>
              <TableHead className="text-right"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody className="w-full">
            {roles.length > 0 &&
              roles.map((role) => (
                <TableRow key={role.id}>
                  <TableCell className="font-medium">{role.name}</TableCell>
                  <TableCell>{role.description}</TableCell>
                  <TableCell>
                    <Button
                      variant="link"
                      onClick={() => {
                        handleDelete(role.id);
                      }}
                    >
                      <TrashIcon />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
          </TableBody>
        </Table>
        {roles.length == 0 && (
          <div>
            <h3 className="text-md font-medium text-gray-500 py-6 w-full text-center">
              {t('config.role.noRolesYet')}
            </h3>
          </div>
        )}
        <div>
          <PaginationComponent
            currentPage={page}
            totalPages={totalPages}
            onPageChange={handlePageChange}
          />
        </div>
      </div>
      <div></div>
      <div className="block w-full">
        {errorMsg !== '' && (
          <Alert variant="destructive">
            <AlertCircleIcon />
            <AlertDescription>
              <p>{errorMsg}</p>
            </AlertDescription>
          </Alert>
        )}
      </div>
    </div>
  );
}
