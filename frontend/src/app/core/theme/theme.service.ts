import { DOCUMENT, Injectable, effect, inject, signal } from '@angular/core';

export type Theme = 'light' | 'dark';

/** Must match the inline script in index.html, which applies the theme before first paint. */
export const THEME_STORAGE_KEY = 'hyperplane-theme';

/** Light/dark theme: toggles `.dark` on <html> and remembers the choice. */
@Injectable({ providedIn: 'root' })
export class ThemeService {
  private readonly document = inject(DOCUMENT);
  private readonly storage = this.document.defaultView?.localStorage;

  readonly theme = signal<Theme>(this.initialTheme());

  constructor() {
    effect(() => {
      const theme = this.theme();
      this.document.documentElement.classList.toggle('dark', theme === 'dark');
      try {
        this.storage?.setItem(THEME_STORAGE_KEY, theme);
      } catch {
        // Storage can be unavailable (private mode); the theme still applies for this visit.
      }
    });
  }

  toggle(): void {
    this.theme.update((theme) => (theme === 'dark' ? 'light' : 'dark'));
  }

  private initialTheme(): Theme {
    const stored = this.readStored();
    if (stored === 'light' || stored === 'dark') return stored;
    return this.document.defaultView?.matchMedia?.('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light';
  }

  private readStored(): string | null {
    try {
      return this.storage?.getItem(THEME_STORAGE_KEY) ?? null;
    } catch {
      return null;
    }
  }
}
