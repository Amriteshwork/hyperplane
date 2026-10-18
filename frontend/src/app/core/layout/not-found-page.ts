import { ChangeDetectionStrategy, Component } from '@angular/core';
import { RouterLink } from '@angular/router';
import { HlmButton } from '@spartan-ng/helm/button';

@Component({
  selector: 'app-not-found-page',
  imports: [RouterLink, HlmButton],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <section class="mx-auto max-w-prose px-4 py-24 text-center">
      <p class="font-mono text-sm text-muted-foreground">404</p>
      <h1 class="mt-2 text-3xl font-semibold tracking-tight">Page not found</h1>
      <p class="mt-4 text-muted-foreground">This page doesn't exist (yet).</p>
      <a hlmBtn class="mt-8" routerLink="/">Back home</a>
    </section>
  `,
})
export class NotFoundPage {}
