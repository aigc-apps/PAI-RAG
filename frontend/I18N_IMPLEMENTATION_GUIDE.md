# PAI-RAG Frontend i18n Implementation Guide

## 概述 / Overview

本文档说明 PAI-RAG 前端的国际化(i18n)实现方案，支持中英文双语切换。

This document describes the internationalization (i18n) implementation for PAI-RAG frontend, supporting Chinese and English bilingual switching.

---

## 🎯 已实现功能 / Implemented Features

✅ **核心 i18n 类** - 单例模式管理翻译和语言状态  
✅ **Core i18n Class** - Singleton pattern for managing translations and language state

✅ **翻译文件** - 包含所有中英文翻译  
✅ **Translation Files** - Contains all Chinese and English translations

✅ **React Context Provider** - 全局 i18n 状态管理  
✅ **React Context Provider** - Global i18n state management

✅ **useI18n Hook** - React Hook 用于在组件中访问翻译  
✅ **useI18n Hook** - React hook for accessing translations in components

✅ **语言切换器** - UI 组件用于切换语言  
✅ **Language Switcher** - UI component for switching languages

✅ **LocalStorage 持久化** - 保存用户的语言偏好  
✅ **LocalStorage Persistence** - Save user's language preference

✅ **参数插值** - 支持动态参数替换  
✅ **Parameter Interpolation** - Support dynamic parameter replacement

✅ **时间格式化工具** - 支持多语言的友好时间显示  
✅ **Time Formatting Utility** - Localized friendly time display

---

## 📁 文件结构 / File Structure

```
frontend/
├── app/
│   ├── layout.tsx                      # 集成 I18nProvider / I18nProvider integration
│   └── providers/
│       └── i18n.tsx                    # I18n Context Provider
├── components/
│   ├── app-sidebar.tsx                 # 已集成 i18n / Integrated with i18n
│   └── language-switcher.tsx           # 语言切换器 / Language switcher
└── lib/
    ├── i18n.ts                         # 核心 i18n 类 / Core i18n class
    ├── translations.ts                 # 所有翻译 / All translations
    ├── init-i18n.ts                    # 初始化 Hook / Initialization hook
    ├── time-format.ts                  # 时间格式化工具 / Time formatting utility
    └── README_I18N.md                  # 详细文档 / Detailed documentation
```

---

## 🚀 快速开始 / Quick Start

### 1. 在组件中使用 / Usage in Components

```tsx
'use client';

import { useI18n } from '@/app/providers/i18n';

export function MyComponent() {
  const { t, language, setLanguage } = useI18n();

  return (
    <div>
      <h1>{t('common.search')}</h1>
      <p>{t('sidebar.knowledgebase')}</p>
      <button onClick={() => setLanguage('en')}>English</button>
      <button onClick={() => setLanguage('zh')}>中文</button>
    </div>
  );
}
```

### 2. 添加新翻译 / Add New Translations

编辑 `lib/translations.ts`:

```typescript
export const translations: I18nConfig = {
  zh: {
    myModule: {
      title: '我的模块',
      action: '点击这里',
    },
  },
  en: {
    myModule: {
      title: 'My Module',
      action: 'Click here',
    },
  },
};
```

### 3. 使用参数插值 / Use Parameter Interpolation

```tsx
// Translation definition
{
  zh: { greeting: '你好, {name}!' },
  en: { greeting: 'Hello, {name}!' }
}

// Usage
{t('greeting', { name: 'Alice' })}
// Output: 你好, Alice! / Hello, Alice!
```

---

## 📋 翻译键结构 / Translation Key Structure

```
translations
├── common          - 通用 UI 元素 / Common UI elements
├── time            - 时间相关 / Time-related strings
├── sidebar         - 侧边栏 / Sidebar navigation
├── workspace       - 工作空间 / Workspace management
├── knowledgebase   - 知识库 / Knowledge base module
├── apps            - 应用 / Apps module
├── evaluation      - 评估 / Evaluation module
├── chat            - 对话 / Chat interface
├── config          - 配置页面 / Configuration pages
├── messages        - 消息提示 / Success/error messages
└── validation      - 验证 / Form validation
```

---

## 🔧 API 参考 / API Reference

### `useI18n()` Hook

```typescript
const {
  language,      // 当前语言 / Current language: 'zh' | 'en'
  setLanguage,   // 设置语言 / Set language
  toggleLanguage, // 切换语言 / Toggle between languages
  t             // 获取翻译 / Get translation
} = useI18n();
```

### `i18n` 单例类 / Singleton Class

```typescript
import { i18n } from '@/lib/i18n';

// 在 React 组件外使用 / Use outside React components
const text = i18n.t('common.search');
i18n.setLanguage('en');
i18n.toggleLanguage();
```

---

## 📝 迁移指南 / Migration Guide

### 迁移现有组件步骤 / Steps to Migrate Existing Components

**Before:**
```tsx
<Button>创建</Button>
<h1>知识库</h1>
```

**After:**
```tsx
const { t } = useI18n();

<Button>{t('common.create')}</Button>
<h1>{t('knowledgebase.title')}</h1>
```

### 示例文件 / Example Files

参考 [page_i18n_example.tsx](./app/knowledgebases/page_i18n_example.tsx) 查看完整的迁移示例。

See [page_i18n_example.tsx](./app/knowledgebases/page_i18n_example.tsx) for a complete migration example.

---

## 🌍 已支持的模块 / Supported Modules

| 模块 Module | 状态 Status | 文件 File |
|------------|-------------|-----------|
| Sidebar | ✅ 已完成 / Completed | `components/app-sidebar.tsx` |
| Workspace | ✅ 已完成 / Completed | `components/app-sidebar.tsx` |
| Knowledge Base | 🔄 示例可用 / Example available | `app/knowledgebases/page_i18n_example.tsx` |
| Apps | ⏳ 待迁移 / To be migrated | - |
| Evaluation | ⏳ 待迁移 / To be migrated | - |
| Config Pages | ⏳ 待迁移 / To be migrated | - |

---

## 💡 最佳实践 / Best Practices

### 1. 键命名规范 / Key Naming Convention
- 使用 camelCase / Use camelCase
- 描述性命名 / Use descriptive names
- 按模块分组 / Group by module

### 2. 保持一致性 / Keep Consistency
- 中英文结构相同 / Same structure for both languages
- 参数名称一致 / Consistent parameter names
- 语气风格统一 / Consistent tone and style

### 3. 避免硬编码 / Avoid Hard-coded Strings
```tsx
// ❌ 不好 / Bad
<Button>创建</Button>

// ✅ 好 / Good
<Button>{t('common.create')}</Button>
```

### 4. 测试双语 / Test Both Languages
- 验证两种语言 / Verify both languages
- 检查 UI 布局 / Check UI layout
- 测试长短文本 / Test short and long text

---

## 🎨 UI 集成 / UI Integration

### 语言切换器已集成在侧边栏 / Language Switcher Integrated in Sidebar

语言切换器已添加到侧边栏的设置区域，用户可以轻松切换语言。

The language switcher has been added to the settings area of the sidebar, allowing users to easily switch languages.

**位置 / Location:** 侧边栏底部 > 设置标题旁边  
**Position:** Bottom of sidebar > Next to Settings title

---

## 🔍 故障排除 / Troubleshooting

### 翻译未显示 / Translation Not Showing

1. ✅ 检查键是否存在 / Check if key exists in translations
2. ✅ 验证键路径正确 / Verify key path is correct
3. ✅ 确保初始化已调用 / Ensure initialization is called
4. ✅ 检查浏览器控制台 / Check browser console

### 语言未持久化 / Language Not Persisting

1. ✅ 检查 localStorage / Check localStorage in DevTools
2. ✅ 验证 localStorage 已启用 / Verify localStorage is enabled
3. ✅ 检查配额问题 / Check quota issues

---

## 📚 参考资源 / Reference Resources

- [详细文档 Detailed Documentation](./lib/README_I18N.md)
- [示例组件 Example Component](./app/knowledgebases/page_i18n_example.tsx)
- [核心实现 Core Implementation](./lib/i18n.ts)
- [翻译文件 Translation File](./lib/translations.ts)

---

## 🚧 未来改进 / Future Enhancements

- [ ] 支持更多语言 / Support more languages (ja, ko, etc.)
- [ ] 翻译文件懒加载 / Lazy loading of translation files
- [ ] 按模块拆分翻译 / Split translations by module
- [ ] 翻译管理 UI / Translation management UI
- [ ] 复数形式支持 / Pluralization support
- [ ] 日期时间本地化 / Date/time localization
- [ ] 数字格式本地化 / Number format localization

---

## 📞 联系与贡献 / Contact & Contributing

添加新功能时请：

When adding new features:

1. 为所有用户可见文本添加翻译键  
   Add translation keys for all user-facing text

2. 提供中英文翻译  
   Provide both Chinese and English translations

3. 在两种语言下测试  
   Test in both languages

4. 如添加新模式，更新文档  
   Update documentation if adding new patterns

---

**Status:** ✅ Ready for Production  
**Version:** 1.0.0  
**Last Updated:** 2026-01-29
