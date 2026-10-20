import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { App } from './app';

describe('App shell', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [provideRouter([])],
    }).compileComponents();
  });

  it('renders the brand, primary nav and a skip link', async () => {
    const fixture = TestBed.createComponent(App);
    await fixture.whenStable();
    const el: HTMLElement = fixture.nativeElement;

    expect(el.querySelector('header a[href="/"]')?.textContent?.trim()).toBe('Hyperplane');
    const navLabels = [...el.querySelectorAll('nav[aria-label="Primary"] a')].map((a) =>
      a.textContent?.trim(),
    );
    expect(navLabels).toEqual(['Learn', 'Concepts', 'Lab', 'Runs']);
    expect(el.querySelector('a[href="#main"]')?.textContent?.trim()).toBe('Skip to content');
    expect(el.querySelector('footer')).not.toBeNull();
  });
});
