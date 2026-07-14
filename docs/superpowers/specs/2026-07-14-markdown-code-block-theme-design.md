# Markdown code block theme design

## Goal

Make fenced shell and source-code blocks visually coherent in both light and
dark application themes. Remove the current per-line dark rectangles caused by
combining a light container with the `oneDark` Prism theme.

## Design

- Select a light Prism syntax theme when the application uses the light theme,
  and retain `oneDark` in dark mode.
- Keep the outer container, header, Prism `pre`, and inner `code` background in
  one theme-consistent visual system. The inner `code` element must be
  transparent so syntax-theme defaults cannot create line-sized backgrounds.
- Keep the language label and copy action in a compact header with a quiet
  divider. Preserve the copied-state icon and add a visible keyboard focus.
- Source code keeps its original line structure and scrolls horizontally when
  needed. Plain output remains wrapping and vertically bounded.
- Use the existing application tokens for borders, radii, shadows, and header
  surfaces so the block still belongs to the surrounding conversation UI.

## Accessibility and responsive behavior

- The copy control retains its localized accessible name.
- Keyboard focus is visible in both themes.
- Horizontal overflow stays inside the code block on narrow screens.
- Syntax colors must remain legible against their matching background theme.

## Validation

- Component tests cover fenced code rendering, the transparent inner code
  background contract, and theme-driven syntax-theme selection.
- Run the frontend test suite and production build.
- Visually inspect representative shell content in light and dark themes when a
  local application session is available.
