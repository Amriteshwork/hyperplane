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
import { MarkdownComponent } from 'ngx-markdown';

import { FIGURES } from '../../figures/figure-registry';
import { renderMath } from './math-extension';

/**
 * Curated Markdown with KaTeX math and interactive figures. Content is sanitised; math is
 * rendered and figures are mounted into their placeholders afterwards.
 */
@Component({
  selector: 'app-markdown-view',
  imports: [MarkdownComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<markdown #md class="prose-hp" [data]="content()" (ready)="onReady()" />`,
})
export class MarkdownView {
  readonly content = input.required<string>();

  private readonly host = viewChild.required<string, ElementRef<HTMLElement>>('md', {
    read: ElementRef,
  });
  private readonly figures = inject(FIGURES);
  private readonly injector = inject(EnvironmentInjector);
  private readonly appRef = inject(ApplicationRef);
  private mounted: ComponentRef<unknown>[] = [];

  constructor() {
    inject(DestroyRef).onDestroy(() => this.unmount());
  }

  protected onReady(): void {
    const root = this.host().nativeElement;
    void renderMath(root);
    void this.mountFigures(root);
  }

  private async mountFigures(root: HTMLElement): Promise<void> {
    this.unmount(); // a re-render replaced the old placeholders
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
