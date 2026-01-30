import dayjs from 'dayjs';
import utc from 'dayjs/plugin/utc';
import { i18n } from './i18n';

dayjs.extend(utc);

/**
 * Format friendly time with i18n support
 * @param utcTime UTC time string
 * @returns Localized friendly time string
 */
export function formatFriendlyTime(utcTime: string): string {
  const now = dayjs.utc();
  const time = dayjs.utc(utcTime);
  const diffMinutes = now.diff(time, 'minute');
  const diffHours = now.diff(time, 'hour');
  const diffDays = now.diff(time, 'day');
  const diffMonths = now.diff(time, 'month');

  if (diffMinutes < 1) {
    return i18n.t('time.justNow');
  } else if (diffDays < 1) {
    // Within today: show hours or minutes
    if (diffHours < 1) {
      return i18n.t('time.minutesAgo', { count: diffMinutes });
    } else {
      return i18n.t('time.hoursAgo', { count: diffHours });
    }
  } else if (diffDays < 30) {
    return i18n.t('time.daysAgo', { count: diffDays });
  } else if (diffMonths < 1) {
    return i18n.t('time.oneMonthAgo');
  } else if (diffMonths < 6) {
    return i18n.t('time.monthsAgo', { count: diffMonths });
  } else {
    return i18n.t('time.halfYearAgo');
  }
}

/**
 * Format Beijing time (for compatibility with existing code)
 * @param utcTime UTC time string
 * @returns Formatted Beijing time string
 */
export function formatBeijingTime(utcTime: string): string {
  return dayjs.utc(utcTime).format('YYYY-MM-DD HH:mm:ss');
}
