import { ChangeDetectionStrategy, Component } from '@angular/core';

@Component({
  selector: 'app-footer',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <footer class="border-t">
      <div
        class="mx-auto flex max-w-6xl flex-col gap-1 px-4 py-6 text-sm text-muted-foreground sm:flex-row sm:justify-between"
      >
        <p>
          <span class="font-mono font-medium text-foreground">Hyperplane</span> · learn machine
          learning by doing
        </p>
        <p>© {{ year }}</p>
      </div>
    </footer>
  `,
})
export class Footer {
  protected readonly year = new Date().getFullYear();
}
