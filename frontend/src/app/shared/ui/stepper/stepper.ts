import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideCheck } from '@ng-icons/lucide';

export interface StepperStep {
  readonly id: string;
  readonly label: string;
  /** Not reachable yet (e.g. no dataset chosen). */
  readonly disabled?: boolean;
}

/** Linear progress through steps; steps before `current` count as complete. */
@Component({
  selector: 'app-stepper',
  imports: [NgIcon],
  providers: [provideIcons({ lucideCheck })],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <nav [attr.aria-label]="label()">
      <p class="mb-2 text-xs text-muted-foreground sm:hidden" aria-hidden="true">
        Step {{ currentIndex() + 1 }} of {{ steps().length }}
      </p>
      <ol class="flex gap-1 overflow-x-auto pb-1">
        @for (step of steps(); track step.id; let i = $index) {
          <li class="shrink-0">
            <button
              type="button"
              class="group flex items-center gap-2 rounded-md px-2 py-1.5 text-sm whitespace-nowrap hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent"
              [attr.aria-current]="i === currentIndex() ? 'step' : null"
              [disabled]="step.disabled"
              (click)="stepSelect.emit(step.id)"
            >
              <span
                class="grid size-6 place-items-center rounded-full border text-xs font-medium"
                [class.bg-primary]="i <= currentIndex()"
                [class.text-primary-foreground]="i <= currentIndex()"
                [class.border-primary]="i <= currentIndex()"
                aria-hidden="true"
              >
                @if (i < currentIndex()) {
                  <ng-icon name="lucideCheck" />
                } @else {
                  {{ i + 1 }}
                }
              </span>
              <span [class]="i === currentIndex() ? 'font-medium' : 'max-sm:sr-only'">
                {{ step.label }}
                @if (i < currentIndex()) {
                  <span class="sr-only">(complete)</span>
                }
              </span>
            </button>
          </li>
        }
      </ol>
    </nav>
  `,
})
export class Stepper {
  readonly steps = input.required<readonly StepperStep[]>();
  readonly current = input.required<string>();
  readonly label = input('Progress');
  readonly stepSelect = output<string>();

  protected readonly currentIndex = computed(() =>
    Math.max(
      0,
      this.steps().findIndex((step) => step.id === this.current()),
    ),
  );
}
