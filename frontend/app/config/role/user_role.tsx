"use client";

import React, { useState, useEffect } from "react";
import { TrashIcon, AlertCircleIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { PaginationComponent } from "@/components/customized/pagination/pagination-component";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Role } from "./role";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export interface UserRole {
  id: string;
  user_id: string;
  role_id: string;
}

const newUserRole = {
  name: "",
  user_id: "",
  role_id: "",
};

export default function UserRolePage() {
  const [roles, setRoles] = useState<Role[]>([]);
  const [editRole, setEditRole] = useState(newUserRole);
  const [userRoles, setUserRoles] = useState<UserRole[]>([]);
  const [modelloading, setModelLoading] = useState(true); // 加载状态
  const [modelerror, setModelError] = useState(""); // 错误信息
  const [errorMsg, setErrorMsg] = useState(""); // 删除时的错误信息

  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const modelSizePerPage = 8;

  const [isEditOpen, setIsEditOpen] = useState(false);

  useEffect(() => {
    const fetchModelConfigs = async () => {
      try {
        const res = await fetch(
          `/v1/config/roles/user_roles?page=${page}&size=${modelSizePerPage}`,
        );
        if (!res.ok) throw new Error("获取角色列表失败");
        const json_data = await res.json();
        const data = json_data.data.items;
        console.log("userRoles", data);
        setUserRoles(data); // 合并
        setTotalPages(json_data.data.pages);
      } catch (err: any) {
        setModelError(err || "加载失败");
      } finally {
        setModelLoading(false);
      }
    };
    fetchModelConfigs();

    const fetchRoles = async () => {
      try {
        const res = await fetch(`/v1/config/roles?page=${page}&size=1000`);
        if (!res.ok) throw new Error("获取角色列表失败");
        const json_data = await res.json();
        const data = json_data.data.items;
        console.log("roles:", data);
        setRoles(data); // 合并
      } catch (err: any) {
        setModelError(err || "加载失败");
      } finally {
        setModelLoading(false);
      }
    };
    fetchRoles();
  }, [page, userRoles.length, isEditOpen]);

  const handleDelete = async (role_id: string) => {
    try {
      const res = await fetch(`/v1/config/roles/user_roles/${role_id}`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error("获取角色列表失败");
      const json_data = await res.json();
      const data = json_data.data;
      console.log("delete role success", data);
      setUserRoles((prev) => prev.filter((role) => role.id !== role_id)); // 合并
      setIsEditOpen(false);
    } catch (err: any) {
      setModelError(err || "加载失败");
    } finally {
      setModelLoading(false);
    }
  };

  const handleAddRole = async () => {
    try {
      const res = await fetch(`/v1/config/roles/user_roles`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(editRole), // 包装为数组
      });
      if (!res.ok) throw new Error("获取角色列表失败");
      const json_data = await res.json();
      const data = json_data.data;
      console.log("create role success", data);
      setUserRoles([...userRoles, data]); // 合并
      setIsEditOpen(false);
      setModelError("");
    } catch (err: any) {
      setModelError(err || "加载失败");
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
        <div className="font-medium text-md">用户角色关系表</div>
        <Popover open={isEditOpen} onOpenChange={setIsEditOpen}>
          <PopoverTrigger asChild>
            <Button variant="default">添加用户角色</Button>
          </PopoverTrigger>
          <PopoverContent className="w-80">
            <div className="grid gap-4">
              <div className="space-y-2">
                <h4 className="leading-none font-medium">添加用户角色</h4>
              </div>
              <div className="grid gap-2">
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="userid">用户ID</Label>
                  <Input
                    id="userid"
                    placeholder="输入用户ID"
                    value={editRole.user_id}
                    onChange={(e) => {
                      setEditRole({ ...editRole, user_id: e.target.value });
                    }}
                    className="col-span-3 h-8"
                  />
                </div>
                <div className="grid grid-cols-4 items-center gap-4">
                  <Label htmlFor="roledesc">角色名称</Label>
                  <Select
                    onValueChange={(value) =>
                      setEditRole({ ...editRole, role_id: value })
                    }
                  >
                    <SelectTrigger className="w-[180px]">
                      <SelectValue placeholder="选择角色名称" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        <SelectLabel>角色</SelectLabel>
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
              <TableHead className="w-3/5">用户ID</TableHead>
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
              暂无用户角色，请点击上方按钮添加
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
        {errorMsg !== "" && (
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
