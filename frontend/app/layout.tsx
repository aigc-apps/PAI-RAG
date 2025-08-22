// app/layout.tsx

import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import * as Toast from '@radix-ui/react-toast';
import React from 'react';

import './globals.css';
import { SidebarInset, SidebarProvider, SidebarTrigger } from '@/components/ui/sidebar';
import { AppSidebar } from '@/components/app-sidebar';
import { MyChatRuntimeProvider } from './runtime/usePaiChatThreadRuntime';
import { ChatProvider } from './providers/chat';

const geistSans = Geist({ variable: '--font-geist-sans', subsets: ['latin'] });
const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  icons: {
    icon: '/favicon.ico',
  },
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
        <link
          rel="icon"
          type="image/png"
          sizes="80x80"
          href="https://pai-rag.oss-cn-hangzhou.aliyuncs.com/logo/pairag_1.png"
        />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <ChatProvider>
          <MyChatRuntimeProvider>
            <SidebarProvider>
              <AppSidebar />
              <SidebarInset>
                  <div className="h-screen w-full overflow-hidden">
                    <SidebarTrigger className="w-10"/>
                    <div className="w-full pt-0">
                      <Toast.Provider>{children}</Toast.Provider>
                    </div>
                  </div>
              </SidebarInset>
            </SidebarProvider>
          </MyChatRuntimeProvider>
        </ChatProvider>
      </body>
    </html>
  );
}
