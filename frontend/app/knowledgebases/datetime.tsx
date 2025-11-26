'use client';
import { FC, useState, useEffect } from 'react';
import * as React from 'react';
import { format } from 'date-fns';
import { CalendarIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Calendar } from '@/components/ui/calendar';
import { Input } from '@/components/ui/input';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { Label } from '@/components/ui/label';

interface DatetimeProps {
  value: number;
  width: string;
  onValueChange: (value: number) => void;
}

// 知识库配置卡片
export const DatetimeInput: FC<DatetimeProps> = ({
  value,
  width,
  onValueChange,
}) => {
  const getInitialDate = () => {
    if (value !== undefined && value !== null) {
      return new Date(value);
    }
    return new Date();
  };

  const getInitialTime = (date: Date) => {
    return date.toTimeString().substring(0, 8);
  };

  const initialDate = getInitialDate();
  const initialTime = getInitialTime(initialDate);

  const [open, setOpen] = useState(false);
  const [date, setDate] = useState<Date>(initialDate);
  const [time, setTime] = useState(initialTime);

  // 当 value prop 变化时，更新内部状态（仅在弹窗关闭时）
  useEffect(() => {
    if (!open && value !== undefined && value !== null) {
      const newDate = new Date(value);
      setDate(newDate);
      setTime(getInitialTime(newDate));
    }
  }, [value, open]);

  // 当弹窗打开时，根据当前 value 更新 date 和 time
  useEffect(() => {
    if (open) {
      const currentDate = value !== undefined && value !== null ? new Date(value) : new Date();
      setDate(currentDate);
      setTime(getInitialTime(currentDate));
    }
  }, [open, value]);

  const handleSaveDate = (newDate: Date) => {
    console.log('选择日期: ', newDate);

    const datetime = newDate;
    datetime.setHours(parseInt(time.split(':')[0]));
    datetime.setMinutes(parseInt(time.split(':')[1]));
    datetime.setSeconds(parseInt(time.split(':')[2]));
    setDate(newDate);
  };

  const handleSaveTime = (newTime: string) => {
    const datetime = date;
    datetime.setHours(parseInt(newTime.split(':')[0]));
    datetime.setMinutes(parseInt(newTime.split(':')[1]));
    datetime.setSeconds(parseInt(newTime.split(':')[2]));
    setDate(datetime);
    setTime(newTime);
  };

  const setNow = () => {
    const now = new Date();
    setDate(now);
    setTime(now.toTimeString().substring(0, 8));
  };

  const saveValue = () => {
    console.log('保存时间: ', date.getTime());
    onValueChange(date.getTime());
    setOpen(false);
  };

  return (
    <Popover
      open={open}
      onOpenChange={(newOpen) => {
        setOpen(newOpen);
        if (!newOpen) {
          saveValue();
        }
      }}
      modal={true}
    >
      <PopoverTrigger asChild>
        {width === 'sm' ? (
          <Button
            variant="outline"
            id="date-picker"
            className="w-[180px] justify-between text-xs"
          >
            {date ? (
              format(date, 'yyyy-MM-dd HH:mm:ss')
            ) : (
              <span>Pick a date</span>
            )}
            <CalendarIcon className="size-3.5" />
          </Button>
        ) : (
          <Button
            variant="outline"
            id="date-picker"
            className="w-[280px] h-6 justify-between text-xs"
          >
            {date ? (
              format(date, 'yyyy-MM-dd HH:mm:ss')
            ) : (
              <span>Pick a date</span>
            )}
            <CalendarIcon className="size-3.5" />
          </Button>
        )}
      </PopoverTrigger>
      <PopoverContent className="w-auto overflow-hidden p-0 relative" align="start">
        <div className="flex items-center pt-2 px-2 gap-2">
          <Label className="text-xs">Time</Label>
          <Input
            type="time"
            step="1"
            value={time}
            onChange={(e) => handleSaveTime(e.target.value)}
            className="pointer-events-auto w-30 h-6" // 很重要，不然会失去焦点，无法交互
          />
          <Button variant="outline" className="text-xs h-6" onClick={setNow}>
            now
          </Button>  
        </div>
        <Calendar
          className="pointer-events-auto"
          mode="single"
          selected={date}
          captionLayout="dropdown"
          onSelect={(newDate) => {
            if (newDate) {
              handleSaveDate(newDate);
            }
          }}
        />
      </PopoverContent>
    </Popover>
  );
};
