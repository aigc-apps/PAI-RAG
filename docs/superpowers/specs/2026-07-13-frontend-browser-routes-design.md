# Frontend Browser Routes Design

## Goal

Give commonly used application pages stable, refresh-safe URLs while preserving
the existing authentication/setup guards and in-memory streaming behavior. Also
fix the knowledge-base list route that currently serializes `KnowledgeBaseRow`
as a data source.

## Routing Architecture

The frontend will use `react-router-dom` with `BrowserRouter`. `App` remains the
authentication and setup guard. Once authenticated and initialized, it renders
route elements instead of selecting a page through local `view` state.

The route table is:

| Path | Surface | Access |
| --- | --- | --- |
| `/` | New chat | Authenticated |
| `/chat/:conversationId` | Persisted conversation | Authenticated |
| `/settings/:section?` | Settings section | Admin |
| `/knowledge` | Knowledge-base list | Admin |
| `/knowledge/:kbId/:tab?` | Knowledge-base detail tab | Admin |
| `/users` | User management | Admin |

The existing `?invite=<token>` entry remains a query parameter because it is an
authentication token flow, not an application page. Dialogs, drawers, preview
panels, unsaved forms, and other transient UI state remain out of the URL.

Production already serves unknown non-file paths through `index.html` in
`deploy/nginx.conf`, so direct requests to browser routes reach the SPA.

## Navigation and State Synchronization

Navigation controls call router navigation rather than mutating an `App` view
state. Browser back/forward therefore becomes authoritative for page changes.

For chat routes:

- `/` creates or activates a local empty draft without repeatedly resetting it
  on ordinary rerenders.
- Selecting a persisted conversation navigates to `/chat/:conversationId`.
- The chat route first activates an already-loaded runtime to preserve partial
  streaming output. Otherwise it loads the conversation and hydrates the chat
  store.
- Once a new response receives its server conversation ID, the URL is replaced
  with `/chat/:conversationId` without adding a redundant history entry.
- An unknown or inaccessible conversation shows the existing load error and
  replaces the route with `/`.

For settings routes, `SettingsView` receives a validated initial section and a
section-change callback. Valid sections are `agents`, `connections`, `tools`,
`skills`, `org-persona`, and `yaml`. The existing Knowledge navigation item is a
link to `/knowledge`, not a settings section. Missing or invalid sections are
canonicalized to `/settings/agents` using replace navigation.

For knowledge routes, `KnowledgeView` receives optional `kbId` and `tab` route
state plus navigation callbacks. Valid detail tabs are `overview`, `config`,
`datasources`, `files`, and `recall`. Opening, creating, deleting, and backing
out of a knowledge base update the route. A missing/inaccessible KB returns to
`/knowledge`; an invalid tab is replaced with `overview`.

## Guards and Error Boundaries

- Authentication/bootstrap/setup guards retain their current priority and are
  evaluated before authenticated route elements.
- A non-admin visiting `/settings`, `/knowledge`, or `/users` is redirected to
  `/` with history replacement.
- Unknown application paths redirect to `/` with history replacement.
- Route validation never trusts IDs for authorization; existing backend calls
  remain authoritative.
- Loading a route must not erase an already-streaming conversation runtime.
- Invite completion removes only the invite query and leaves a canonical `/`
  route.

## Backend Correction

`GET /v1/knowledge-bases` must continue serializing each `KnowledgeBaseRow`
with `_dump`. Only data-source list/get endpoints call
`KnowledgeService.data_source_payload`. A route regression test will create a
knowledge base and list it, proving no `active_job_id` access occurs on the KB
model.

The existing Alembic migration correctly adds `active_job_id` to
`knowledge_data_sources`; no `knowledge_bases.active_job_id` column is needed.

## Testing

Backend tests cover non-empty knowledge-base listing.

Frontend tests cover:

- direct rendering and refresh-equivalent mount for every top-level route;
- chat hydration from `/chat/:conversationId` and failed-ID fallback;
- sidebar navigation updating the URL;
- settings section URL synchronization and invalid-section canonicalization;
- knowledge-base ID/tab direct loading, tab changes, and invalid-route fallback;
- admin route rejection for regular users;
- `popstate`/back navigation returning to the prior surface.

The frontend production build and existing backend/frontend suites must remain
green.
