# Sidebar Settings Shortcut Design

## Goal

Give administrators a permanent, low-noise shortcut to Settings without requiring them to open the account menu.

## Scope

- Add a settings icon button to the right side of the `PAI-Loop` brand in the chat sidebar header.
- Keep the existing Settings item in the account menu as a secondary entry point.
- Reuse the existing `/settings/agents` destination and route guard.
- Do not change Settings routes, authentication APIs, account-menu behavior, or sidebar width.

## Visual Design

The shortcut is a 32×32px Ghost icon button aligned to the right edge of the sidebar brand row. Its default state is transparent with a muted gear icon. Hover adds the existing subtle surface background and stronger text color; keyboard focus uses the shared focus ring. The control uses the existing radius and theme variables so it follows light and dark themes.

The button has a localized `title` and `aria-label` of “Settings”. It remains visually secondary to the logo and the primary “New chat” button.

## Component and Data Flow

`ChatPage` already passes `onOpenSettings` to `Sidebar` only when `isAdmin` is true. `Sidebar` will render the header shortcut only when this callback exists, using the same callback already passed to `UserMenu`.

Clicking the shortcut invokes `onOpenSettings`, which navigates to `/settings/agents`. The server-backed admin route guard remains the authority for direct URL access; the conditional shortcut is a presentation boundary, not a replacement for authorization.

No additional role lookup, state, API call, or route is introduced.

## Edge Cases

- Regular users receive no `onOpenSettings` callback and see no shortcut.
- During authentication transitions, the shortcut is absent until the authenticated admin state is known.
- If the callback is unavailable, the brand row keeps its existing layout without reserving an empty button slot.
- The account-menu Settings entry remains available to administrators.

## Testing

- Render `Sidebar` without `onOpenSettings` and assert that the header settings button is absent.
- Render `Sidebar` with `onOpenSettings`, assert that the localized button is present, click it, and verify the callback fires once.
- Keep existing App route-guard and account-menu tests unchanged to verify the shortcut does not weaken authorization or remove the secondary entry point.

## Acceptance Criteria

- Administrators can open Settings from the Logo row in one click.
- Regular users do not see the shortcut.
- The shortcut uses the approved Ghost styling and is keyboard accessible.
- Existing account-menu and route-guard behavior remains intact.
