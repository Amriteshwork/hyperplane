import { ChangeDetectionStrategy, Component, computed, inject, output } from '@angular/core';
import { HlmBadge } from '@spartan-ng/helm/badge';
import { HlmButton } from '@spartan-ng/helm/button';
import { HlmProgressImports } from '@spartan-ng/helm/progress';

import { ThemeService } from '../../core/theme/theme.service';
import { LabStore, isFinished } from './lab.store';
import { RunResults } from './run-results';

/** A run's live progress, then its results (or what went wrong). */
@Component({
  selector: 'app-run-step',
  imports: [HlmBadge, HlmButton, HlmProgressImports, RunResults],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (store.run(); as run) {
      <div class="flex flex-wrap items-center gap-2">
        <h2 class="text-lg font-semibold">Predicting {{ run.experiment.target }}</h2>
        <span hlmBadge variant="outline" class="capitalize" data-testid="run-status">{{
          run.status
        }}</span>
      </div>
      <p class="mt-1 text-sm text-muted-foreground">
        {{ run.algorithm_id }} · {{ featureCount() }} features ·
        {{ run.experiment.test_size * 100 }}% held out · seed {{ run.experiment.seed }}
      </p>

      @if (!finished()) {
        <div class="mt-6 grid gap-2" role="status" aria-live="polite">
          <p class="text-sm">{{ store.progress()?.message ?? 'Waiting for a worker…' }}</p>
          <hlm-progress [value]="percent()" aria-label="Training progress">
            <hlm-progress-indicator />
          </hlm-progress>
        </div>
      } @else if (run.status === 'failed') {
        <div class="mt-6 rounded-lg border border-destructive/50 p-4" role="alert">
          <p class="font-medium">Training failed</p>
          <p class="mt-1 text-sm text-muted-foreground">{{ run.error }}</p>
        </div>
      } @else if (run.report; as report) {
        <app-run-results class="mt-8 block" [report]="report" [dark]="dark()" />
      }

      @if (finished()) {
        <div class="mt-10 flex flex-wrap gap-2">
          <button hlmBtn variant="outline" type="button" (click)="configure.emit()">
            Change settings and train again
          </button>
          <button hlmBtn variant="ghost" type="button" (click)="restart.emit()">
            Use another dataset
          </button>
        </div>
      }
    } @else if (store.runError(); as error) {
      <p class="text-sm text-destructive" role="alert">{{ error }}</p>
    } @else {
      <div class="h-40 animate-pulse rounded bg-muted motion-reduce:animate-none" aria-busy="true">
        <span class="sr-only">Loading…</span>
      </div>
    }
  `,
})
export class RunStep {
  protected readonly store = inject(LabStore);
  private readonly theme = inject(ThemeService);

  readonly configure = output();
  readonly restart = output();

  protected readonly dark = computed(() => this.theme.theme() === 'dark');
  protected readonly finished = computed(() => {
    const run = this.store.run();
    return !!run && isFinished(run.status);
  });
  protected readonly featureCount = computed(() => {
    const experiment = this.store.run()?.experiment;
    return experiment
      ? experiment.numeric_features.length + experiment.categorical_features.length
      : 0;
  });
  protected readonly percent = computed(() => {
    const progress = this.store.progress();
    return progress?.total ? Math.round(((progress.step ?? 0) / progress.total) * 100) : 0;
  });
}
