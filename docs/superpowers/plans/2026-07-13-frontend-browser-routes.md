# Frontend Browser Routes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add refresh-safe browser routes for chat, settings, knowledge bases, and users while fixing knowledge-base list serialization.

**Architecture:** Wrap the application in `BrowserRouter`, keep authentication/setup in `App`, and render authenticated surfaces through declarative routes. Route parameters drive persisted resource loading; component-local state remains only for transient UI.

**Tech Stack:** React 19, TypeScript, React Router DOM, Zustand, Vitest, Testing Library, FastAPI, pytest.

## Global Constraints

- Supported paths are `/`, `/chat/:conversationId`, `/settings/:section?`, `/knowledge`, `/knowledge/:kbId/:tab?`, and `/users`.
- Browser URL/history is authoritative for page navigation.
- Admin routes redirect non-admin users to `/`.
- Unknown paths, IDs, sections, and tabs are canonicalized with replace navigation.
- Existing streaming runtimes must not be overwritten by stale server hydration.
- Dialogs, drawers, previews, and unsaved form state remain out of the URL.
- `KnowledgeBaseRow` must never be passed to `data_source_payload`.

---

### Task 1: Fix Knowledge-Base List Serialization

**Files:**
- Modify: `backend/app/routes/knowledge.py:168-176`
- Modify: `backend/tests/test_routes_knowledge.py`

**Interfaces:**
- Consumes: `KnowledgeService.list_kbs(user=user) -> list[KnowledgeBaseRow]`
- Produces: `GET /v1/knowledge-bases` JSON serialized with `_dump`.

- [ ] **Step 1: Add a failing non-empty list route test**

Extend the existing knowledge route scenario to create one KB, request
`GET /v1/knowledge-bases`, and assert:

```python
response = client.get("/v1/knowledge-bases")
assert response.status_code == 200
assert response.json()["data"][0]["id"] == kb["id"]
assert "active_job_id" not in response.json()["data"][0]
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
cd backend
uv run pytest tests/test_routes_knowledge.py -q
```

Expected: FAIL with `KnowledgeBaseRow object has no attribute active_job_id`.

- [ ] **Step 3: Restore the correct serializer**

```python
rows = await svc.list_kbs(user=user)
return {"data": [_dump(row) for row in rows]}
```

- [ ] **Step 4: Verify the focused backend tests**

```bash
cd backend
uv run pytest tests/test_routes_knowledge.py tests/test_routes_knowledge_datasource.py -q
uv run ruff check app/routes/knowledge.py tests/test_routes_knowledge.py
```

Expected: all selected tests pass.

---

### Task 2: Install and Bootstrap React Router

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/package-lock.json`
- Modify: `frontend/src/main.tsx`
- Modify: `frontend/src/components/App.tsx`
- Modify: `frontend/src/components/__tests__/App.test.tsx`
- Modify: `frontend/src/components/__tests__/AppGuards.test.tsx`

**Interfaces:**
- Consumes: browser pathname and authenticated/admin state.
- Produces: declarative route elements and canonical redirects.

- [ ] **Step 1: Add failing top-level route tests**

Wrap test renders in `MemoryRouter` and add assertions that:

```tsx
render(<App />, { wrapper: ({ children }) => <MemoryRouter initialEntries={["/users"]}>{children}</MemoryRouter> });
expect(await screen.findByText(/user management/i)).toBeInTheDocument();
```

Also mount an admin at `/settings`, `/knowledge`, and an unknown path; assert the
first three render their surfaces and the unknown path renders chat.

- [ ] **Step 2: Run App tests and verify RED**

```bash
cd frontend
npm test -- --run src/components/__tests__/App.test.tsx src/components/__tests__/AppGuards.test.tsx
```

Expected: route-specific surfaces are absent because `App` always initializes
its local `view` to `chat`.

- [ ] **Step 3: Add React Router dependency**

```bash
cd frontend
npm install react-router-dom
```

- [ ] **Step 4: Bootstrap BrowserRouter**

Wrap `App` in `frontend/src/main.tsx`:

```tsx
<BrowserRouter>
  <App />
</BrowserRouter>
```

- [ ] **Step 5: Replace App view state with authenticated routes**

Use `Routes`, `Route`, and `Navigate`. Keep auth/setup returns before the route
tree. Render admin elements through a small guard:

```tsx
const admin = (element: ReactNode) => isAdmin ? element : <Navigate to="/" replace />;

<Routes>
  <Route path="/" element={<ChatPage />} />
  <Route path="/chat/:conversationId" element={<ChatPage />} />
  <Route path="/settings/:section?" element={admin(<SettingsPage doc={doc!} />)} />
  <Route path="/knowledge" element={admin(<KnowledgePage />)} />
  <Route path="/knowledge/:kbId/:tab?" element={admin(<KnowledgePage />)} />
  <Route path="/users" element={admin(<UsersView onBack={() => navigate("/")} />)} />
  <Route path="*" element={<Navigate to="/" replace />} />
</Routes>
```

Use route-aware callbacks for Settings, Knowledge, Users, Sidebar, and setup
completion. Invite cleanup replaces the URL with `/`.

- [ ] **Step 6: Verify top-level route and guard tests**

Run the App tests from Step 2. Expected: all pass.

---

### Task 3: Route-Driven Chat Selection and Recovery

**Files:**
- Modify: `frontend/src/components/App.tsx`
- Modify: `frontend/src/components/Sidebar.tsx`
- Modify: `frontend/src/components/__tests__/App.test.tsx`
- Modify: `frontend/src/components/__tests__/Sidebar.test.tsx`

**Interfaces:**
- Consumes: `conversationId` route parameter, `getConversation`, and chat/conversation stores.
- Produces: URL-driven conversation activation and `/` new-chat navigation.

- [ ] **Step 1: Add failing direct-chat and navigation tests**

Test `/chat/c1` with `getConversation` resolving a detail and assert the message
appears and `selectedId` becomes `c1`. Test a rejected request and assert the
router location becomes `/`. Click a sidebar conversation and assert the
location becomes `/chat/c1`; click New Chat and assert `/`.

- [ ] **Step 2: Verify chat route tests fail**

```bash
cd frontend
npm test -- --run src/components/__tests__/App.test.tsx src/components/__tests__/Sidebar.test.tsx
```

Expected: the URL does not select/hydrate conversations.

- [ ] **Step 3: Implement `ChatPage` route synchronization**

In `App.tsx`, read `conversationId` with `useParams`. On parameter change:

1. Call `activateByConversationId(id)` and select it when already loaded.
2. Otherwise call `getConversation(id)`, hydrate it, and select it.
3. On failure, clear selection, show the existing load error, and navigate to
   `/` with replace.

Watch the active runtime's server conversation ID. When the current path is `/`
and a new response receives an ID, replace the URL with `/chat/:id`.

- [ ] **Step 4: Make Sidebar navigation route-aware**

Use `useNavigate` for persisted rows, local runtimes, New Chat, and active-row
deletion. Preserve `newDraft`, runtime activation, deletion, and streaming state;
only move server hydration into `ChatPage`.

- [ ] **Step 5: Verify chat route tests**

Run the command from Step 2. Expected: all pass.

---

### Task 4: Settings Section Routes

**Files:**
- Modify: `frontend/src/components/SettingsView.tsx`
- Modify: `frontend/src/components/App.tsx`
- Modify: `frontend/src/components/__tests__/SettingsView.test.tsx`
- Modify: `frontend/src/components/__tests__/App.test.tsx`

**Interfaces:**
- Produces: exported `SettingsSection` type and controlled route synchronization.
- Consumes: optional `initialSection` and `onSectionChange(section)` props.

- [ ] **Step 1: Add failing settings deep-link tests**

Mount `/settings/connections`, assert Models is selected, click Capabilities,
and assert `/settings/tools`. Mount `/settings/not-a-section` and assert the
location is replaced with `/settings/agents`.

- [ ] **Step 2: Verify settings tests fail**

Run App and SettingsView tests. Expected: Settings always selects Agents and the
URL never changes.

- [ ] **Step 3: Add controlled section props**

Export:

```typescript
export type SettingsSection = "agents" | "connections" | "tools" | "skills" | "org-persona" | "yaml";
```

Add `initialSection` and `onSectionChange` props. Synchronize local state when
the route changes, invoke the callback on tab clicks, and retain async YAML
loading before navigating to `yaml`. Keep Knowledge as navigation to
`/knowledge` rather than a settings section.

- [ ] **Step 4: Validate and canonicalize in App**

Validate `section` from `useParams`. Replace invalid/missing values with
`/settings/agents`; navigate valid clicks to `/settings/:section`.

- [ ] **Step 5: Verify settings tests**

Run App and SettingsView tests. Expected: all pass.

---

### Task 5: Knowledge Base ID and Tab Routes

**Files:**
- Modify: `frontend/src/components/KnowledgeView.tsx`
- Modify: `frontend/src/components/App.tsx`
- Modify: `frontend/src/components/__tests__/KnowledgeView.test.tsx`
- Modify: `frontend/src/components/__tests__/App.test.tsx`

**Interfaces:**
- Produces: exported `KnowledgeTab` and route-controlled KB detail.
- Consumes: optional `kbId`, optional validated `tab`, and navigation callbacks.

- [ ] **Step 1: Add failing knowledge deep-link tests**

Mock two KBs. Mount `/knowledge/kb_1/datasources` and assert KB 1 plus Data
Sources content appears. Click Files and assert `/knowledge/kb_1/files`. Mount an
unknown KB and assert `/knowledge`; mount an invalid tab and assert
`/knowledge/kb_1/overview`.

- [ ] **Step 2: Verify knowledge route tests fail**

Run KnowledgeView and App tests. Expected: KnowledgeView always renders its list
because `openId` and detail tab are local state.

- [ ] **Step 3: Make KnowledgeView route-controlled**

Export:

```typescript
export type KnowledgeTab = "overview" | "config" | "datasources" | "files" | "recall";
```

Replace `openId` and detail-local tab state with `kbId`/`tab` props and callbacks.
Opening/creating a KB navigates to its overview. Back/deletion navigates to the
list. Tab clicks navigate to the selected tab. After loading completes, an
unknown ID invokes list replacement.

- [ ] **Step 4: Validate knowledge parameters in App**

Use `useParams`, validate the tab, and replace invalid tabs with `overview`.
Connect all callbacks to `/knowledge` and `/knowledge/:kbId/:tab`.

- [ ] **Step 5: Verify knowledge route tests**

Run KnowledgeView and App tests. Expected: all pass.

---

### Task 6: Full Regression and Production Build

**Files:**
- Verify all files modified above.

**Interfaces:**
- Consumes: complete backend correction and frontend route tree.
- Produces: verified refresh-safe application build.

- [ ] **Step 1: Run backend regression**

```bash
cd backend
uv run pytest -q
```

Expected: zero failures; opt-in PostgreSQL tests may skip when their URL is not configured.

- [ ] **Step 2: Run frontend tests and build**

```bash
cd frontend
npm test -- --run
npm run build
```

Expected: zero test failures and a successful Vite production build.

- [ ] **Step 3: Run scoped static checks**

```bash
cd backend
uv run ruff check app/routes/knowledge.py tests/test_routes_knowledge.py
cd ../frontend
npx tsc -b --pretty false
```

Expected: zero errors.

- [ ] **Step 4: Commit the implementation**

```bash
git add backend/app/routes/knowledge.py backend/tests/test_routes_knowledge.py \
  frontend/package.json frontend/package-lock.json frontend/src/main.tsx \
  frontend/src/components/App.tsx frontend/src/components/Sidebar.tsx \
  frontend/src/components/SettingsView.tsx frontend/src/components/KnowledgeView.tsx \
  frontend/src/components/__tests__/App.test.tsx \
  frontend/src/components/__tests__/AppGuards.test.tsx \
  frontend/src/components/__tests__/Sidebar.test.tsx \
  frontend/src/components/__tests__/SettingsView.test.tsx \
  frontend/src/components/__tests__/KnowledgeView.test.tsx
git commit -m "feat(frontend): add refresh-safe browser routes"
```
