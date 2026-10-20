import {
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  inject,
  input,
  signal,
} from '@angular/core';
import { Title } from '@angular/platform-browser';
import { Router } from '@angular/router';

import { Stepper, StepperStep } from '../../shared/ui/stepper/stepper';
import { ConfigureStep } from './configure-step';
import { DataStep } from './data-step';
import { LabStore } from './lab.store';
import { RunStep } from './run-step';
import { SetupStep } from './setup-step';

type StepId = 'data' | 'setup' | 'configure' | 'run';

/**
 * The Lab: data → target & features → settings → train & results. The dataset and run live in
 * the URL (`?dataset=…&run=…`), so a reload or a shared link reopens the same step.
 */
@Component({
  selector: 'app-lab-page',
  imports: [Stepper, DataStep, SetupStep, ConfigureStep, RunStep],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="mx-auto max-w-5xl px-4 py-8 sm:py-12">
      <h1 class="text-3xl font-semibold tracking-tight">Lab</h1>
      <p class="mt-2 max-w-prose text-muted-foreground">
        Pick a dataset, choose what to predict, train a model and see how well it does.
      </p>

      <app-stepper
        class="mt-8 block"
        label="Lab steps"
        [steps]="steps()"
        [current]="current()"
        (stepSelect)="goTo($any($event))"
      />

      <div class="mt-8">
        @switch (current()) {
          @case ('data') {
            <app-data-step
              (sampleSelect)="chooseSample($event)"
              (fileSelect)="upload($event)"
              (chooseAnother)="goTo('data')"
            />
          }
          @case ('setup') {
            <app-setup-step (back)="goTo('data')" (next)="goTo('configure')" />
          }
          @case ('configure') {
            <app-configure-step (back)="goTo('setup')" (train)="train()" />
          }
          @case ('run') {
            <app-run-step (configure)="goTo('configure')" (restart)="goTo('data')" />
          }
        }
      </div>
    </div>
  `,
})
export class LabPage {
  protected readonly store = inject(LabStore);
  private readonly router = inject(Router);

  /** Query parameters. */
  readonly dataset = input<string>();
  readonly run = input<string>();
  readonly algorithm = input<string>();

  private readonly editing = signal<'setup' | 'configure'>('setup');

  protected readonly current = computed<StepId>(() => {
    if (this.run()) return 'run';
    if (!this.store.datasetReady()) return 'data';
    return this.editing();
  });
  protected readonly steps = computed<StepperStep[]>(() => {
    const ready = this.store.datasetReady();
    return [
      { id: 'data', label: 'Data' },
      { id: 'setup', label: 'Target & features', disabled: !ready },
      { id: 'configure', label: 'Settings', disabled: !ready || !this.store.setupValid() },
      { id: 'run', label: 'Train & results', disabled: !this.run() },
    ];
  });

  constructor() {
    inject(Title).setTitle('Lab · Hyperplane');
    effect(() => this.store.datasetId.set(this.dataset() ?? null));
    effect(() => this.store.runId.set(this.run() ?? null));
    effect(() => this.store.preferredAlgorithm.set(this.algorithm() ?? null));
  }

  protected goTo(step: StepId): void {
    if (step === 'run') return;
    if (step === 'data') {
      this.navigate({ dataset: null, run: null });
      return;
    }
    this.editing.set(step);
    this.navigate({ run: null });
  }

  protected async chooseSample(sampleId: string): Promise<void> {
    const datasetId = await this.store.createFromSample(sampleId);
    if (datasetId) this.openDataset(datasetId);
  }

  protected async upload(file: File): Promise<void> {
    const datasetId = await this.store.upload(file);
    if (datasetId) this.openDataset(datasetId);
  }

  protected async train(): Promise<void> {
    const runId = await this.store.train();
    if (runId) this.navigate({ run: runId });
  }

  private openDataset(datasetId: string): void {
    this.editing.set('setup');
    this.navigate({ dataset: datasetId, run: null });
  }

  private navigate(queryParams: Record<string, string | null>): void {
    void this.router.navigate([], { queryParams, queryParamsHandling: 'merge' });
  }
}
