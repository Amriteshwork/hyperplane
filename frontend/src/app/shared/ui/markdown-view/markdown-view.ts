import { ChangeDetectionStrategy, Component, ElementRef, input, viewChild } from '@angular/core';
import { MarkdownComponent } from 'ngx-markdown';

import { renderMath } from './math-extension';

/** Curated Markdown with KaTeX math. Content is sanitised; math is rendered afterwards. */
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

  protected onReady(): void {
    void renderMath(this.host().nativeElement);
  }
}
