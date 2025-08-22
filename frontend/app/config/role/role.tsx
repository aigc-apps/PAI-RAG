'use client';

import React, { useState, useEffect } from 'react';
import {
  TrashIcon,
  Edit,
  AlertCircleIcon,
  Loader2,
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
  const [editRole, setEditRole] = useState(newRole); // 存储 Embedding 配置
  const [roles, setRoles] = useState<Role[]>([]); // 存储 Embedding 配置
  const [modelloading, setModelLoading] = useState(true); // 加载状态
  const [modelerror, setModelError] = useState(''); // 错误信息
  const [errorMsg, setErrorMsg] = useState(''); // 删除时的错误信息

  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const modelSizePerPage = 8;

  const [isEditOpen, setIsEditOpen] = useState(false);

  useEffect(() => {
    const fetchModelConfigs = async () => {
      try {
        const res = await fetch(
          `/api/config/roles?page=${page}&size=${modelSizePerPage}`,
        );
        if (!res.ok) throw new Error('获取角色列表失败');
        const json_data = await res.json();
        const data = json_data.data.items;
        setRoles(data); // 合并
        setTotalPages(json_data.data.pages);
      } catch (err: any) {
        setModelError(err || '加载失败');
      } finally {
        setModelLoading(false);
      }
    };
    fetchModelConfigs();
  }, [page, roles.length, isEditOpen]);

  const handleDelete = async (role_id: string) => {
    try {
      const res = await fetch(`/api/config/roles/${role_id}`, {
        method: 'DELETE',
      });
      if (!res.ok) throw new Error('获取角色列表失败');
      const json_data = await res.json();
      const data = json_data.data;
      console.log('delete role success', data);
      setRoles((prev) => prev.filter((role) => role.id !== role_id)); // 合并
      setIsEditOpen(false);
    } catch (err: any) {
      setModelError(err || '加载失败');
    } finally {
      setModelLoading(false);
    }
  };

  const handleAddRole = async () => {
    try {
      const res = await fetch(`/api/config/roles`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editRole), // 包装为数组
      });
      if (!res.ok) throw new Error('获取角色列表失败');
      const json_data = await res.json();
      const data = json_data.data;
      console.log('create role success', data);
      setRoles([...roles, data]); // 合并
      setIsEditOpen(false);
      setModelError('');
    } catch (err: any) {
      setModelError(err || '加载失败');
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
        <div className="font-medium text-md">角色配置表</div>
        <Popover open={isEditOpen} onOpenChange={setIsEditOpen}>
          <PopoverTrigger asChild>
            <Button variant="default">添加角色</Button>
          </PopoverTrigger>
          <PopoverContent className="w-120">
            <div className="grid gap-4">
              <div className="space-y-2">
                <h4 className="leading-none font-medium">创建角色</h4>
              </div>
              <div className="grid gap-2">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="rolename">角色名称</Label>
                  <Input
                    id="rolename"
                    placeholder="输入名称"
                    value={editRole.name}
                    onChange={(e) => {
                      setEditRole({ ...editRole, name: e.target.value });
                    }}
                    className="col-span-3 h-8"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="roledesc">角色描述</Label>
                  <Textarea
                    id="roledesc"
                    placeholder="输入描述"
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
                  保存
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
              <TableHead className="w-1/5">角色名称</TableHead>
              <TableHead className="w-3/5">角色描述</TableHead>
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
              暂无角色，请点击上方按钮添加
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
