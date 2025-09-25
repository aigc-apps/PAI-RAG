// components/UsernameSetter.tsx
'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { CircleUser } from 'lucide-react';
import { Label } from '@/components/ui/label';

interface UserIdProps {
  user_id: string | undefined;
  onChange: (user_id: string) => void;
}

export default function UserIdInput({
  user_id,
  onChange,
}: UserIdProps) {
  const [open, setOpen] = useState(false);
  const [username, setUsername] = useState(user_id);

  const handleSave = () => {
    if (username?.trim()) {
      // ✅ Save to localStorage, context, or API
      console.log('Username saved:', username.trim());
      onChange(username.trim())
    }
    setOpen(false);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="sm" className="absolute right-1 top-1">
          <CircleUser className='w-5 h-5'/>
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>设置user_id</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="username">user_id，用于测试知识库权限</Label>
            <Input
              id="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="e.g., john_doe"
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSave();
              }}
            />
          </div>
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleSave} disabled={!username?.trim()}>
              Save
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}