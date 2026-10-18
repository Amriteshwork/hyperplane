# Hyperplane frontend

Angular 22 (standalone, zoneless, signals) + Tailwind CSS v4 + Spartan UI.

```bash
npm ci
npm start              # ng serve on :4200, proxies /api → :8000 (proxy.conf.json)
npm run api:generate   # typed client in src/app/core/api from the running backend's OpenAPI
npm test               # Vitest component/unit tests (add -- --watch=false for a single run)
npm run lint           # angular-eslint
npm run format:check   # Prettier
npm run e2e            # Playwright (starts ng serve; E2E_BASE_URL=... to target a running app)
```

- `src/app/core/` — generated API client, config tokens, theme, layout shell
- `src/app/shared/ui/` — presentational components; `helm/` holds Spartan primitives
  (add more with `npx ng g @spartan-ng/cli:ui <name>`)
- `src/app/features/<feature>/` — one lazy-loaded route per feature

Never edit `src/app/core/api/` by hand: change the backend's Pydantic schema and regenerate.
