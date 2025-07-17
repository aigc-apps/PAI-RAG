import dayjs from "dayjs";
import timezone from "dayjs/plugin/timezone";
import utc from "dayjs/plugin/utc";

dayjs.extend(utc);
dayjs.extend(timezone);

export function formatFileSize(bytes: number, decimalPlaces = 1): string {
  if (bytes === 0) return "0 Bytes";

  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  const formatted = parseFloat((bytes / Math.pow(k, i)).toFixed(decimalPlaces));

  return `${formatted} ${sizes[i]}`;
}

export function formatBeijingTime(utcTime: string): string {
  const beijingTime = dayjs
    .utc(utcTime)
    .tz("Asia/Shanghai")
    .format("YYYY-MM-DD HH:mm:ss");
  return beijingTime;
}
