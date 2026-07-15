# Default OpenAI-Compatible Provider Design

## Goal

Make the deployment's default model provider an explicit `openai_compatible`
provider whose connection values come from `OPENAI_BASE_URL` and
`OPENAI_API_KEY` by default. Administrators may enter manual URL and API-key
fallbacks when either environment variable is absent.

This change applies only to the default provider. Custom providers retain their
current independently configured connection behavior.

## Configuration Model

The model provider schema gains a provider type and optional connection fields:

```yaml
models:
  default_model: openai/gpt-4o-mini
  providers:
    - name: openai
      type: openai_compatible
      use_default_env: true
      base_url: ""
      api_key: ""
      models:
        - id: gpt-4o-mini
```

`type` identifies the provider protocol independently of its display name.
`use_default_env` marks the one environment-managed default connection. It is
explicit configuration rather than an inference from the provider name or from
whichever model happens to be selected as `default_model`.

For a provider with `type: openai_compatible` and `use_default_env: true`, the
runtime resolves connection values in this order:

1. URL: non-empty `OPENAI_BASE_URL`, then authored `base_url`.
2. API key: non-empty `OPENAI_API_KEY`, then authored `api_key`.

Whitespace-only environment values count as absent. The resolved environment
values are runtime-only: they are never copied into the authored document,
configuration API response, raw YAML response, or browser state.

Custom providers do not read the two default OpenAI environment variables.
Their existing `base_url`, `api_key_env`, and optional direct `api_key` semantics
remain intact.

## Default and Compatibility Behavior

The built-in default document changes from a hard-coded OpenAI URL plus
`api_key_env: OPENAI_API_KEY` to the explicit environment-managed provider
shape above. The provider remains named `openai` and retains the shipped model
catalog.

Older authored default-provider records remain valid. During document
normalization, the shipped `openai` default is upgraded to
`type: openai_compatible` and `use_default_env: true` without discarding an
existing manual `base_url`, `api_key`, `api_key_env`, or model list. Other
providers are not migrated based solely on their name.

Existing custom providers with required `base_url` continue to parse and route
without behavior changes. The backend schema permits the default provider to
omit its authored URL because it can be supplied at runtime.

## Runtime Resolution and Errors

Provider configuration keeps authored and resolved values separate. A focused
resolver produces the effective URL and key when a chat, embedding, rerank, or
connection-test client is created. Every client path uses the same resolver so
test results match actual model calls.

Saving a provider is allowed even when neither source is configured. Using or
testing it then fails with an actionable message that identifies the missing
setting and both accepted sources, for example:

- `Default model provider requires OPENAI_BASE_URL or a manual base URL.`
- `Default model provider requires OPENAI_API_KEY or a manual API key.`

An environment value always wins over a manual fallback. Editing the fallback
while an environment value is present is allowed, but it does not change the
active connection until the environment value is removed and the service is
reloaded.

## Secret Handling

A manually entered API key is accepted by the configuration endpoint and stored
in the project's existing server-side configuration store. The current project
does not provide at-rest secret encryption, so this design does not claim that
the stored value is encrypted.

The configuration API masks an authored model-provider `api_key` as
`********`. When the frontend saves an unchanged masked value, the backend
restores the existing secret before persistence, matching the existing provider
and vector-database secret-preservation behavior. Actual values from
`OPENAI_API_KEY` are never persisted or returned.

Runtime status exposes only booleans indicating whether the environment URL and
key are configured and whether manual fallbacks exist. It never exposes secret
values. The URL environment value is also not returned, because deployment
environment values are runtime configuration rather than authored UI state.

## Frontend Experience

The Models settings page identifies the default environment-managed provider
with an `OpenAI-compatible` type and an `Environment default` badge.

Its edit dialog explains the precedence and displays:

- `OPENAI_BASE_URL` status: configured or not configured.
- A manual base-URL fallback input.
- `OPENAI_API_KEY` status: configured or not configured.
- A password-style manual API-key fallback input. When a stored fallback exists,
  the field shows a configured placeholder rather than the secret.

The dialog does not ask administrators to type environment-variable names for
this provider because the names are fixed. Custom provider dialogs retain their
current base URL and API-key environment-name controls.

Connection rows communicate the effective source without revealing values:
`OPENAI_BASE_URL`, `manual fallback`, or `not configured` for the URL; and
`OPENAI_API_KEY`, `manual fallback`, or `not configured` for the key. All new
labels, help text, statuses, validation errors, and accessible names are added
in Chinese and English.

## API and Data Flow

1. The backend loads the authored model catalog without resolving environment
   values into it.
2. Runtime-status decoration checks environment presence and manual-fallback
   presence, then adds non-persisted status flags for the settings UI.
3. The frontend edits only authored fallback fields and submits the document.
4. The update route restores a masked manual key from the current authored
   document before validation and persistence.
5. Provider routing resolves environment-first effective values when building a
   client.
6. Connection testing goes through the same router and therefore verifies the
   same effective values used by production requests.

## Validation and Tests

Backend tests cover:

- built-in default-provider schema and authored output;
- `OPENAI_BASE_URL` and `OPENAI_API_KEY` precedence over manual fallbacks;
- independent fallback when only one environment variable is absent;
- whitespace-only environment values;
- missing URL and missing key errors at use time, not save time;
- chat, embedding, rerank, and connection-test paths using the resolver;
- manual-key masking and masked-value preservation on update;
- environment values never appearing in API or authored YAML output;
- legacy default-provider normalization and unchanged custom-provider behavior.

Frontend tests cover:

- environment-managed badge, type, and configured-status rendering;
- fixed environment names and manual fallback fields;
- password-field behavior for an existing manual key;
- saving manual fallbacks without replacing a masked key;
- custom-provider forms retaining their current controls;
- Chinese and English strings used by the new UI.

The frontend unit suite and production build, plus the focused backend provider
and configuration-route suites, must pass before delivery.

## Out of Scope

- Encrypting secrets at rest or integrating a new secret manager.
- Applying `OPENAI_BASE_URL` or `OPENAI_API_KEY` to custom providers.
- Automatically changing the environment-managed provider when
  `models.default_model` points to another provider.
- Exposing environment values to the browser.
- Changing model registration, model defaults, or provider testing semantics
  beyond using the shared resolver.
