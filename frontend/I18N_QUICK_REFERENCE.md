# i18n Quick Reference Card

## 🚀 Quick Usage

### Import and Use
```tsx
import { useI18n } from '@/app/providers/i18n';

const { t, language, setLanguage } = useI18n();
```

### Basic Translation
```tsx
{t('common.search')}        // 搜索 / Search
{t('common.create')}        // 创建 / Create
{t('common.delete')}        // 删除 / Delete
```

### With Parameters
```tsx
{t('time.minutesAgo', { count: 5 })}  // 5分钟前 / 5 minutes ago
```

### Change Language
```tsx
<button onClick={() => setLanguage('en')}>English</button>
<button onClick={() => setLanguage('zh')}>中文</button>
```

## 📋 Common Translation Keys

### Common Actions
```
common.search          搜索 / Search
common.create          创建 / Create
common.delete          删除 / Delete
common.edit            编辑 / Edit
common.save            保存 / Save
common.cancel          取消 / Cancel
common.confirm         确认 / Confirm
common.submit          提交 / Submit
common.back            返回 / Back
```

### Sidebar
```
sidebar.knowledgebase  知识库 / Knowledge Base
sidebar.apps           应用 / Apps
sidebar.evaluation     评估 / Evaluation
sidebar.conversation   对话 / Conversation
sidebar.settings       设置 / Settings
sidebar.model          模型 / Model
sidebar.vectordb       向量数据库 / Vector DB
```

### Knowledge Base
```
knowledgebase.title              知识库 / Knowledge Base
knowledgebase.create             新建知识库 / Create Knowledge Base
knowledgebase.delete             删除知识库 / Delete Knowledge Base
knowledgebase.searchPlaceholder  搜索知识库... / Search knowledge base...
knowledgebase.emptyTitle         暂无知识库 / No Knowledge Base
knowledgebase.emptyMessage       还没有创建任何知识库... / No knowledge base has been created yet...
```

### Workspace
```
workspace.select            选择工作空间 / Select Workspace
workspace.create            新建工作空间 / Create Workspace
workspace.idLabel           工作空间 ID / Workspace ID
workspace.nameLabel         工作空间名称 / Workspace Name
```

### Time
```
time.justNow          刚刚 / Just now
time.minutesAgo       {count}分钟前 / {count} minutes ago
time.hoursAgo         {count}小时前 / {count} hours ago
time.daysAgo          {count}天前 / {count} days ago
time.monthsAgo        {count}个月前 / {count} months ago
```

### Messages
```
messages.saveSuccess    保存成功 / Saved successfully
messages.saveError      保存失败 / Failed to save
messages.deleteSuccess  删除成功 / Deleted successfully
messages.deleteError    删除失败 / Failed to delete
messages.createSuccess  创建成功 / Created successfully
messages.createError    创建失败 / Failed to create
```

## 🔧 Outside React Components

```typescript
import { i18n } from '@/lib/i18n';

// Get translation
const text = i18n.t('common.search');

// Get current language
const lang = i18n.getLanguage();  // 'zh' | 'en'

// Set language
i18n.setLanguage('en');

// Toggle language
i18n.toggleLanguage();
```

## 🌏 Time Formatting

```typescript
import { formatFriendlyTime } from '@/lib/time-format';

const friendlyTime = formatFriendlyTime(utcTimeString);
// Output: 刚刚 / Just now
//         5分钟前 / 5 minutes ago
//         3小时前 / 3 hours ago
```

## ➕ Adding New Translations

Edit `frontend/lib/translations.ts`:

```typescript
export const translations: I18nConfig = {
  zh: {
    myModule: {
      title: '我的模块',
      description: '这是描述',
      action: '点击{name}',
    },
  },
  en: {
    myModule: {
      title: 'My Module',
      description: 'This is description',
      action: 'Click {name}',
    },
  },
};
```

## 🎨 Language Switcher Component

```tsx
import { LanguageSwitcher } from '@/components/language-switcher';

<LanguageSwitcher />
```

## 📝 Migration Pattern

### Before
```tsx
<h1>知识库</h1>
<Button>创建</Button>
<p>还没有创建任何知识库</p>
```

### After
```tsx
const { t } = useI18n();

<h1>{t('knowledgebase.title')}</h1>
<Button>{t('common.create')}</Button>
<p>{t('knowledgebase.emptyMessage')}</p>
```

## 🔍 Files to Check

| Purpose | File |
|---------|------|
| Core i18n class | `lib/i18n.ts` |
| All translations | `lib/translations.ts` |
| React provider | `app/providers/i18n.tsx` |
| Language switcher | `components/language-switcher.tsx` |
| Time formatting | `lib/time-format.ts` |
| Example usage | `app/knowledgebases/page_i18n_example.tsx` |
| Full documentation | `lib/README_I18N.md` |

---

**Tip:** Keep this card handy while migrating components to i18n!
