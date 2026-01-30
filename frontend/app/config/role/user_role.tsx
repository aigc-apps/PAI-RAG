'use client';

import React, { useState, useEffect } from 'react';
import { TrashIcon, AlertCircleIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { PaginationComponent } from '@/components/customized/pagination/pagination-component';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Table,
  TableBody,
  TableCell,
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
import { Role } from './role';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useTenantFetch } from '@/hooks/use-tenant-fetch';
import { useI18n } from '@/app/providers/i18n';

export interface UserRole {
  id: string;
  user_id: string;
  role_id: string;
}

const newUserRole = {
  name: '',
  user_id: '',
  role_id: '',
};

export default function UserRolePage() {
  const { t } = useI18n();
  const [roles, setRoles] = useState<Role[]>([]);
  const [editRole, setEditRole] = useState(newUserRole);
  const [userRoles, setUserRoles] = useState<UserRole[]>([]);
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
          `/api/config/roles/user_roles?page=${page}&size=${modelSizePerPage}`,
        );
        if (!res.ok) throw new Error(t('config.role.fetchRoleListFailed'));
        const json_data = await res.json();
        const data = json_data.data.items;
        console.log('userRoles', data);
        setUserRoles(data);
        setTotalPages(json_data.data.pages);
      } catch (err: any) {
        setModelError(err || t('config.role.loadFailed'));
      } finally {
        setModelLoading(false);
      }
    };
    fetchModelConfigs();

    const fetchRoles = async () => {
      try {
        const res = await tenantFetch(`/api/config/roles?page=${page}&size=1000`);
        if (!res.ok) throw new Error(t('config.role.fetchRoleListFailed'));
        const json_data = await res.json();
        const data = json_data.data.items;
        console.log('roles:', data);
        setRoles(data);
      } catch (err: any) {
        setModelError(err || t('config.role.loadFailed'));
      } finally {
        setModelLoading(false);
      }
    };
    fetchRoles();
  }, [page, userRoles.length, isEditOpen]);

  const handleDelete = async (role_id: string) => {
    try {
      const res = await tenantFetch(`/api/config/roles/user_roles/${role_id}`, {
        method: 'DELETE',
      });
      if (!res.ok) throw new Error(t('config.role.fetchRoleListFailed'));
      const json_data = await res.json();
      const data = json_data.data;
      console.log(t('config.role.deleteUserRoleSuccess'), data);
      setUserRoles((prev) => prev.filter((role) => role.id !== role_id));
      setIsEditOpen(false);
    } catch (err: any) {
      setModelError(err || t('config.role.loadFailed'));
    } finally {
      setModelLoading(false);
    }
  };

  const handleAddRole = async () => {
    try {
      const res = await tenantFetch(`/api/config/roles/user_roles`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editRole),
      });
      if (!res.ok) throw new Error(t('config.role.fetchRoleListFailed'));
      const json_data = await res.json();
      const data = json_data.data;
      console.log(t('config.role.createUserRoleSuccess'), data);
      setUserRoles([...userRoles, data]);
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
    <div id="userrole">
      <div className="flex items-center gap-12">
        <div className="font-medium text-md">{t('config.role.userRoleRelationTable')}</div>
        <Popover open={isEditOpen} onOpenChange={setIsEditOpen}>
          <PopoverTrigger asChild>
            <Button variant="default">{t('config.role.addUserRole')}</Button>
          </PopoverTrigger>
          <PopoverContent className="w-80">
            <div className="grid gap-4">
              <div className="space-y-2">
                <h4 className="leading-none font-medium">{t('config.role.addUserRole')}</h4>
              </div>
              <div className="grid gap-2">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="userid">{t('config.role.userId')}</Label>
                  <Input
                    id="userid"
                    placeholder={t('config.role.userIdPlaceholder')}
                    value={editRole.user_id}
                    onChange={(e) => {
                      setEditRole({ ...editRole, user_id: e.target.value });
                    }}
                    className="col-span-3 h-8"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="roledesc">{t('config.role.roleName')}</Label>
                  <Select
                    onValueChange={(value) =>
                      setEditRole({ ...editRole, role_id: value })
                    }
                  >
                    <SelectTrigger className="w-[180px]">
                      <SelectValue placeholder={t('config.role.selectRoleName')} />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        <SelectLabel>{t('config.role.roleLabel')}</SelectLabel>
                        {roles.map((role) => (
                          <SelectItem key={role.id} value={role.id}>
                            {role.name}
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    </SelectContent>
                  </Select>
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
              <TableHead className="w-3/5">{t('config.role.userId')}</TableHead>
              <TableHead className="text-right"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody className="w-full">
            {userRoles.length > 0 &&
              userRoles.map((userRole) => (
                <TableRow key={userRole.id}>
                  <TableCell className="font-medium">
                    {
                      roles.filter((role) => role.id === userRole.role_id)[0]
                        ?.name
                    }
                  </TableCell>
                  <TableCell>{userRole.user_id}</TableCell>
                  <TableCell>
                    <Button
                      variant="link"
                      onClick={() => {
                        handleDelete(userRole.id);
                      }}
                    >
                      <TrashIcon />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
          </TableBody>
        </Table>
        {userRoles.length == 0 && (
          <div>
            <h3 className="text-md font-medium text-gray-500 py-6 w-full text-center">
              {t('config.role.noUserRolesYet')}
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
