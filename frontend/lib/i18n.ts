/**
 * i18n utility class for bilingual support
 * Supports Chinese (zh) and English (en) languages
 */

import { translations } from './translations';

export type Language = 'zh' | 'en';

const LANGUAGE_COOKIE_NAME = 'language';
const LANGUAGE_COOKIE_MAX_AGE = 31536000; // 1 year

function getLanguageFromCookie(): Language | null {
  if (typeof document === 'undefined') return null;
  const match = document.cookie.match(new RegExp('(^| )' + LANGUAGE_COOKIE_NAME + '=([^;]+)'));
  if (!match) return null;
  const value = decodeURIComponent(match[2]);
  return value === 'zh' || value === 'en' ? value : null;
}

function setLanguageCookie(lang: Language): void {
  if (typeof document === 'undefined') return;
  document.cookie = `${LANGUAGE_COOKIE_NAME}=${encodeURIComponent(lang)}; path=/; max-age=${LANGUAGE_COOKIE_MAX_AGE}`;
}

export interface Translation {
  [key: string]: string | Translation;
}

export interface I18nConfig {
  zh: Translation;
  en: Translation;
}

class I18n {
  private static instance: I18n;
  private currentLanguage: Language = 'en';
  private translations: I18nConfig = { zh: {}, en: {} };
  private listeners: Set<() => void> = new Set();

  private constructor() {
    // Auto-register translations
    this.registerTranslations(translations);
    
    if (typeof window !== 'undefined') {
      const savedLang = getLanguageFromCookie();
      if (savedLang) {
        this.currentLanguage = savedLang;
      }
    }
  }

  public static getInstance(): I18n {
    if (!I18n.instance) {
      I18n.instance = new I18n();
    }
    return I18n.instance;
  }

  /**
   * Register translations
   */
  public registerTranslations(translations: I18nConfig): void {
    this.translations = {
      zh: { ...this.translations.zh, ...translations.zh },
      en: { ...this.translations.en, ...translations.en },
    };
  }

  /**
   * Get translation by key (supports nested keys with dot notation)
   */
  public t(key: string, params?: Record<string, string | number>): string {
    const keys = key.split('.');
    let value: any = this.translations[this.currentLanguage];

    for (const k of keys) {
      if (value && typeof value === 'object') {
        value = value[k];
      } else {
        return key; // Return key if translation not found
      }
    }

    if (typeof value !== 'string') {
      return key;
    }

    // Replace parameters if provided
    if (params) {
      return value.replace(/\{(\w+)\}/g, (match, paramKey) => {
        return params[paramKey]?.toString() || match;
      });
    }

    return value;
  }

  /**
   * Get current language
   */
  public getLanguage(): Language {
    return this.currentLanguage;
  }

  /**
   * Set current language
   */
  public setLanguage(lang: Language): void {
    if (lang === this.currentLanguage) return;

    this.currentLanguage = lang;

    if (typeof window !== 'undefined') {
      setLanguageCookie(lang);
    }

    this.listeners.forEach(listener => listener());
  }

  /**
   * Subscribe to language changes
   */
  public subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  /**
   * Toggle between languages
   */
  public toggleLanguage(): void {
    this.setLanguage(this.currentLanguage === 'zh' ? 'en' : 'zh');
  }
}

export const i18n = I18n.getInstance();
