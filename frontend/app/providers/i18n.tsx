'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { i18n, Language } from '@/lib/i18n';
import { translations } from '@/lib/translations';

interface I18nContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  toggleLanguage: () => void;
  t: (key: string, params?: Record<string, string | number>) => string;
}

const I18nContext = createContext<I18nContextType | undefined>(undefined);

interface I18nProviderProps {
  children: ReactNode;
}

export function I18nProvider({ children }: I18nProviderProps) {
  const [language, setLanguageState] = useState<Language>(i18n.getLanguage());

  useEffect(() => {
    // Ensure translations are registered (defensive, should already be done in i18n constructor)
    i18n.registerTranslations(translations);
    
    const unsubscribe = i18n.subscribe(() => {
      setLanguageState(i18n.getLanguage());
    });

    return unsubscribe;
  }, []);

  const setLanguage = (lang: Language) => {
    i18n.setLanguage(lang);
  };

  const toggleLanguage = () => {
    i18n.toggleLanguage();
  };

  const t = (key: string, params?: Record<string, string | number>) => {
    return i18n.t(key, params);
  };

  return (
    <I18nContext.Provider value={{ language, setLanguage, toggleLanguage, t }}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n(): I18nContextType {
  const context = useContext(I18nContext);
  if (context === undefined) {
    throw new Error('useI18n must be used within an I18nProvider');
  }
  return context;
}
