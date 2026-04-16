// app/layout.tsx

import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import * as Toast from '@radix-ui/react-toast';
import React from 'react';

import './globals.css';
import { SidebarInset, SidebarProvider, SidebarTrigger } from '@/components/ui/sidebar';
import { AppSidebar } from '@/components/app-sidebar';
import { HEADER_SLOT_ID } from '@/components/header-portal';
import { MyChatRuntimeProvider, TokenUsageProvider } from './runtime/usePaiChatThreadRuntime';
import { ChatProvider } from './providers/chat';
import { TenantProvider } from './providers/tenant';
import { I18nProvider } from './providers/i18n';
import { Toaster } from '@/components/ui/sonner';

const geistSans = Geist({ variable: '--font-geist-sans', subsets: ['latin'] });
const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'PAI-RAG',
  description: 'Created by PAI.',
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <head>
        <title>PAI-RAG</title>
      </head>
      <body>
        <I18nProvider>
          <TenantProvider>
            <ChatProvider>
              <TokenUsageProvider>
                <MyChatRuntimeProvider>
                  <SidebarProvider>
                    <AppSidebar />
                    <SidebarInset className="main-gradient-bg">
                      <div className="h-screen w-full overflow-hidden flex flex-col">
                        <header className="flex items-center gap-2 shrink-0 w-full px-3 py-2 bg-background/40 backdrop-blur-sm border-b border-border">
                          <SidebarTrigger className="w-9 h-9 shrink-0" />
                          <div id={HEADER_SLOT_ID} className="flex-1 min-w-0 flex items-center gap-3" />
                        </header>
                        <div className="w-full flex-1 min-h-0">
                          {children}
                          <Toaster duration={3000} position="top-right" />
                        </div>
                      </div>
                    </SidebarInset>
                  </SidebarProvider>
                </MyChatRuntimeProvider>
              </TokenUsageProvider>
            </ChatProvider>
          </TenantProvider>
        </I18nProvider>
      </body>
    </html>
  );
}
