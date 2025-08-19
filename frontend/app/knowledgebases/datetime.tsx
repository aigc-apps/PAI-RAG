'use client';
import { FC, useState } from 'react';
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
  let initialDate = new Date();
  if (value !== undefined) {
    initialDate = new Date(value);
  }
  initialDate.setHours(0);
  initialDate.setMinutes(0);
  initialDate.setSeconds(0);


  const initalTime = '00:00:00';
  const [open, setOpen] = useState(false);
  const [date, setDate] = useState<Date>(initialDate);
  const [time, setTime] = useState(initalTime);

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
      <PopoverContent className="w-auto overflow-hidden p-0" align="start">
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
        <div className="flex items-center">
          <Input
            type="time"
            step="1"
            defaultValue={time}
            onChange={(e) => handleSaveTime(e.target.value)}
            className="pointer-events-auto" // 很重要，不然会失去焦点，无法交互
          />
          <Button variant="link" onClick={setNow}>
            now
          </Button>
          <Button variant="default" className="h-8" onClick={saveValue}>
            OK
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
};
