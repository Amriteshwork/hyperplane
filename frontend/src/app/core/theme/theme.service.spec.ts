import { DOCUMENT } from '@angular/core';
import { TestBed } from '@angular/core/testing';

import { THEME_STORAGE_KEY, ThemeService } from './theme.service';

describe('ThemeService', () => {
  beforeEach(() => {
    localStorage.clear();
    TestBed.inject(DOCUMENT).documentElement.classList.remove('dark');
  });

  it('respects a stored choice', () => {
    localStorage.setItem(THEME_STORAGE_KEY, 'dark');
    expect(TestBed.inject(ThemeService).theme()).toBe('dark');
  });

  it('toggles the .dark class and persists the choice', () => {
    localStorage.setItem(THEME_STORAGE_KEY, 'light');
    const service = TestBed.inject(ThemeService);
    const root = TestBed.inject(DOCUMENT).documentElement;

    service.toggle();
    TestBed.tick();

    expect(root.classList.contains('dark')).toBe(true);
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark');

    service.toggle();
    TestBed.tick();
    expect(root.classList.contains('dark')).toBe(false);
  });
});
