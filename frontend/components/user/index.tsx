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
import { useI18n } from '@/app/providers/i18n';

interface UserIdProps {
  user_id: string | undefined;
  onChange: (user_id: string) => void;
}

export default function UserIdInput({
  user_id,
  onChange,
}: UserIdProps) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const [username, setUsername] = useState(user_id);

  const handleSave = () => {
    if (username?.trim()) {
      // Save to localStorage, context, or API
      console.log('Username saved:', username.trim());
      onChange(username.trim())
    }
    setOpen(false);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 px-2 gap-1.5 text-xs text-muted-foreground hover:text-foreground"
          title={user_id}
        >
          <CircleUser className="w-3.5 h-3.5" />
          {user_id && <span className="truncate max-w-[100px]">{user_id}</span>}
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t('user.setUserId')}</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="username">{t('user.userIdLabel')}</Label>
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
              {t('common.cancel')}
            </Button>
            <Button onClick={handleSave} disabled={!username?.trim()}>
              {t('common.save')}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}