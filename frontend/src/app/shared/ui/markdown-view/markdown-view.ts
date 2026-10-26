import {
  ApplicationRef,
  ChangeDetectionStrategy,
  Component,
  ComponentRef,
  DestroyRef,
  ElementRef,
  EnvironmentInjector,
  createComponent,
  inject,
  input,
  viewChild,
} from '@angular/core';
import { Router } from '@angular/router';
import { MarkdownComponent } from 'ngx-markdown';

import { FIGURES } from '../../figures/figure-registry';
import { VideoClip, VideoClipData } from '../video-clip/video-clip';
import { renderMath } from './math-extension';

/**
 * Curated Markdown with KaTeX math, interactive figures and videos. Content is sanitised; math
 * is rendered and components are mounted into their placeholders afterwards. Links to our own
 * pages (`/concepts/...`) navigate in-app instead of reloading.
 */
@Component({
  selector: 'app-markdown-view',
  imports: [MarkdownComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<markdown
    #md
    class="prose-hp"
    [data]="content()"
    (ready)="onReady()"
    (click)="onClick($event)"
  />`,
})
export class MarkdownView {
  readonly content = input.required<string>();
  /** What `::video[name]` directives refer to, by name. */
  readonly videos = input<Readonly<Record<string, VideoClipData>>>({});

  private readonly host = viewChild.required<string, ElementRef<HTMLElement>>('md', {
    read: ElementRef,
  });
  private readonly figures = inject(FIGURES);
  private readonly injector = inject(EnvironmentInjector);
  private readonly appRef = inject(ApplicationRef);
  private readonly router = inject(Router);
  private mounted: ComponentRef<unknown>[] = [];

  constructor() {
    inject(DestroyRef).onDestroy(() => this.unmount());
  }

  protected onReady(): void {
    const root = this.host().nativeElement;
    this.unmount(); // a re-render replaced the old placeholders
    void renderMath(root);
    void this.mountFigures(root);
    this.mountVideos(root);
  }

  protected onClick(event: MouseEvent): void {
    const link = (event.target as Element | null)?.closest('a');
    const href = link?.getAttribute('href');
    if (!href?.startsWith('/') || href.startsWith('//')) return;
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) {
      return; // let the browser open a new tab or window
    }
    event.preventDefault();
    void this.router.navigateByUrl(href);
  }

  private mountVideos(root: HTMLElement): void {
    for (const slot of root.querySelectorAll<HTMLElement>('.video-slot')) {
      const name = slot.textContent?.trim() ?? '';
      const video = Object.hasOwn(this.videos(), name) ? this.videos()[name] : undefined;
      slot.textContent = video ? '' : `(Video “${name}” is not available.)`;
      if (!video) continue;
      const ref = createComponent(VideoClip, {
        environmentInjector: this.injector,
        hostElement: slot,
      });
      ref.setInput('video', video);
      this.appRef.attachView(ref.hostView);
      ref.changeDetectorRef.detectChanges(); // mounted mid-render, so the next tick may miss it
      this.mounted.push(ref);
    }
  }

  private async mountFigures(root: HTMLElement): Promise<void> {
    for (const slot of root.querySelectorAll<HTMLElement>('.figure-slot')) {
      const name = slot.textContent?.trim() ?? '';
      // hasOwn: names like "constructor" must not resolve to Object.prototype members.
      const load = Object.hasOwn(this.figures, name) ? this.figures[name] : undefined;
      slot.textContent = load ? '' : `(Figure “${name}” is not available.)`;
      if (!load) continue;
      const ref = createComponent(await load(), {
        environmentInjector: this.injector,
        hostElement: slot,
      });
      this.appRef.attachView(ref.hostView);
      this.mounted.push(ref);
    }
  }

  private unmount(): void {
    for (const ref of this.mounted) ref.destroy();
    this.mounted = [];
  }
}
