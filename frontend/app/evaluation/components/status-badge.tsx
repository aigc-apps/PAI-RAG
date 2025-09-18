// app/evaluation/[evalId]/components/StatusBadge.tsx

'use client';

import { Badge } from '@/components/ui/badge';
import {
  RefreshCw,
  CheckCircle,
  XCircle,
  Clock
} from "lucide-react";

interface StatusBadgeProps {
  status: string;
}

export function StatusBadge({ status }: StatusBadgeProps) {
  switch (status) {
    case "running":
      return (
        <Badge variant="secondary" className="bg-blue-100 text-blue-800 hover:bg-blue-200">
          <RefreshCw className="mr-1 h-3 w-3 animate-spin" /> 运行中
        </Badge>
      );
    case "success":
      return (
        <Badge variant="secondary" className="bg-green-100 text-green-800 hover:bg-green-200">
          <CheckCircle className="mr-1 h-3 w-3" /> 成功
        </Badge>
      );
    case "failed":
      return (
        <Badge variant="secondary" className="bg-red-100 text-red-800 hover:bg-red-200">
          <XCircle className="mr-1 h-3 w-3" /> 失败
        </Badge>
      );
    case "pending":
      return (
        <Badge variant="secondary" className="bg-yellow-100 text-yellow-800 hover:bg-yellow-200">
          <Clock className="mr-1 h-3 w-3 animate-spin" /> 等待中
        </Badge>
      );
    default:
      return <Badge>{status}</Badge>;
  }
}