import dayjs from 'dayjs';
import timezone from 'dayjs/plugin/timezone';
import utc from 'dayjs/plugin/utc';

dayjs.extend(utc);
dayjs.extend(timezone);

export function formatFileSize(bytes: number, decimalPlaces = 1): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  const formatted = parseFloat((bytes / Math.pow(k, i)).toFixed(decimalPlaces));

  return `${formatted} ${sizes[i]}`;
}

export function formatBeijingTime(utcTime: string): string {
  const beijingTime = dayjs
    .utc(utcTime)
    .tz('Asia/Shanghai')
    .format('YYYY-MM-DD HH:mm:ss');
  return beijingTime;
}

export function calculateTimeDifference(startTime: string, endTime: string): string {
  const formatUTC = (str: string) => 
          str.replace(' ', 'T').replace(/\.\d+$/, '') + 'Z';

  const start = new Date(formatUTC(startTime));
  const end = new Date(formatUTC(endTime));

  if (isNaN(Number(start)) || isNaN(Number(end))) return "-";

  const diffSeconds = Math.floor((Number(end) - Number(start)) / 1000);
  return `${diffSeconds}s`;
}