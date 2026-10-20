import {
  DestroyRef,
  Injectable,
  computed,
  effect,
  inject,
  linkedSignal,
  resource,
  signal,
} from '@angular/core';

import { Api } from '../../core/api/api';
import { ApiConfiguration } from '../../core/api/api-configuration';
import { createExperiment } from '../../core/api/fn/experiments/create-experiment';
import { getAlgorithm } from '../../core/api/fn/learn/get-algorithm';
import { getDataset } from '../../core/api/fn/datasets/get-dataset';
import { getTrainingOptions } from '../../core/api/fn/datasets/get-training-options';
import { uploadDataset } from '../../core/api/fn/datasets/upload-dataset';
import { getRun } from '../../core/api/fn/runs/get-run';
import { streamRunEvents } from '../../core/api/fn/runs/stream-run-events';
import { createDatasetFromSample } from '../../core/api/fn/samples/create-dataset-from-sample';
import { listSamples } from '../../core/api/fn/samples/list-samples';
import { JsonValue } from '../../core/api/models/json-value';
import { RunEventOut } from '../../core/api/models/run-event-out';
import { RunStatus } from '../../core/api/models/run-status';
import { EVENT_SOURCE_FACTORY } from '../../core/http/event-stream';
import { errorMessage } from '../../core/http/http-errors';
import { JsonSchema, SchemaValues } from '../../shared/ui/json-schema-form/json-schema';

const PROFILING_POLL_MS = 1000;
const RECONNECT_MS = 2000;
const FINISHED: readonly RunStatus[] = ['succeeded', 'failed'];

export function isFinished(status: RunStatus): boolean {
  return FINISHED.includes(status);
}

/** State of one Lab session: dataset → target & features → settings → run. Provided by the route. */
@Injectable()
export class LabStore {
  private readonly api = inject(Api);
  private readonly rootUrl = inject(ApiConfiguration).rootUrl;
  private readonly openEventSource = inject(EVENT_SOURCE_FACTORY);
  private readonly destroyRef = inject(DestroyRef);

  readonly datasetId = signal<string | null>(null);
  readonly runId = signal<string | null>(null);
  /** Algorithm to preselect when it fits the target (e.g. from "Try it in the Lab"). */
  readonly preferredAlgorithm = signal<string | null>(null);

  // ── data ────────────────────────────────────────────────────────────────
  private readonly samplesPage = resource({
    loader: () => this.api.invoke(listSamples, { limit: 100 }),
  });
  readonly samples = computed(() =>
    this.samplesPage.hasValue() ? this.samplesPage.value().items : [],
  );
  readonly samplesError = computed(() =>
    this.samplesPage.error() ? errorMessage(this.samplesPage.error()) : null,
  );

  private readonly datasetResource = resource({
    params: () => this.datasetId() ?? undefined,
    loader: ({ params }) => this.api.invoke(getDataset, { dataset_id: params }),
  });
  readonly dataset = computed(() =>
    this.datasetResource.hasValue() ? this.datasetResource.value() : null,
  );
  readonly datasetError = computed(() =>
    this.datasetResource.error() ? errorMessage(this.datasetResource.error()) : null,
  );
  readonly datasetReady = computed(() => this.dataset()?.status === 'ready');
  readonly creatingDataset = signal(false);
  readonly createDatasetError = signal<string | null>(null);

  // ── target & features ───────────────────────────────────────────────────
  private readonly optionsResource = resource({
    params: () => (this.datasetReady() ? (this.datasetId() ?? undefined) : undefined),
    loader: ({ params }) => this.api.invoke(getTrainingOptions, { dataset_id: params }),
  });
  readonly options = computed(() =>
    this.optionsResource.hasValue() ? this.optionsResource.value() : null,
  );
  readonly optionsError = computed(() =>
    this.optionsResource.error() ? errorMessage(this.optionsResource.error()) : null,
  );

  /** Defaults to the best candidate that something can train. */
  readonly target = linkedSignal(() => {
    const targets = this.options()?.targets ?? [];
    return (targets.find((t) => t.algorithms.length) ?? targets[0])?.column ?? null;
  });
  readonly targetOption = computed(
    () => this.options()?.targets.find((t) => t.column === this.target()) ?? null,
  );
  /** Every usable column except the target, until the learner changes it. */
  readonly features = linkedSignal<readonly string[]>(() => {
    const target = this.target();
    return (this.options()?.features ?? [])
      .filter((f) => f.role !== null && f.column !== target)
      .map((f) => f.column);
  });
  readonly setupValid = computed(
    () => !!this.targetOption()?.algorithms.length && this.features().length > 0,
  );

  // ── settings ────────────────────────────────────────────────────────────
  readonly algorithmId = linkedSignal(() => {
    const algorithms = this.targetOption()?.algorithms ?? [];
    const preferred = algorithms.find((a) => a.id === this.preferredAlgorithm());
    return (preferred ?? algorithms[0])?.id ?? null;
  });
  private readonly algorithmResource = resource({
    params: () => this.algorithmId() ?? undefined,
    loader: ({ params }) => this.api.invoke(getAlgorithm, { algorithm_id: params }),
  });
  readonly paramsSchema = computed(() =>
    this.algorithmResource.hasValue()
      ? (this.algorithmResource.value().params_schema as JsonSchema)
      : null,
  );
  /** Reset to the schema defaults whenever the algorithm changes. */
  readonly params = linkedSignal<SchemaValues>(() => {
    this.algorithmId();
    return {};
  });
  readonly paramsValid = signal(true);
  readonly testSize = signal(0.2);
  readonly seed = signal(42);
  readonly creatingRun = signal(false);
  readonly createRunError = signal<string | null>(null);

  // ── run ─────────────────────────────────────────────────────────────────
  private readonly runResource = resource({
    params: () => this.runId() ?? undefined,
    loader: ({ params }) => this.api.invoke(getRun, { run_id: params }),
  });
  readonly run = computed(() => (this.runResource.hasValue() ? this.runResource.value() : null));
  readonly runError = computed(() =>
    this.runResource.error() ? errorMessage(this.runResource.error()) : null,
  );
  readonly events = linkedSignal<readonly RunEventOut[]>(() => {
    this.runId();
    return [];
  });
  readonly progress = computed(
    () =>
      this.events()
        .filter((e) => e.type === 'progress')
        .at(-1) ?? null,
  );

  constructor() {
    // Profiling runs in the background: poll until the dataset is ready (or failed).
    effect((onCleanup) => {
      if (this.dataset()?.status !== 'profiling') return;
      const timer = setTimeout(() => this.datasetResource.reload(), PROFILING_POLL_MS);
      onCleanup(() => clearTimeout(timer));
    });
    // Follow an unfinished run live; fetch it again once the stream says it's over.
    effect((onCleanup) => {
      const run = this.run();
      if (!run || isFinished(run.status)) return;
      const source = this.follow(run.id);
      onCleanup(() => source.close());
    });
  }

  async createFromSample(sampleId: string): Promise<string | null> {
    return this.createDataset(() =>
      this.api.invoke(createDatasetFromSample, { sample_id: sampleId }),
    );
  }

  async upload(file: File): Promise<string | null> {
    // The generator types binary fields as string; its request builder sends a Blob as-is.
    const body = { file: file as Blob as unknown as string };
    return this.createDataset(() => this.api.invoke(uploadDataset, { body }));
  }

  toggleFeature(column: string, selected: boolean): void {
    this.features.update((features) =>
      selected ? [...features, column] : features.filter((f) => f !== column),
    );
  }

  /** Queue a run with the current settings; returns its id. */
  async train(): Promise<string | null> {
    const datasetId = this.datasetId();
    const target = this.target();
    const algorithmId = this.algorithmId();
    if (!datasetId || !target || !algorithmId) return null;
    this.creatingRun.set(true);
    this.createRunError.set(null);
    try {
      const run = await this.api.invoke(createExperiment, {
        body: {
          dataset_id: datasetId,
          target,
          features: [...this.features()],
          algorithm_id: algorithmId,
          params: this.params() as Record<string, JsonValue>,
          split: { test_size: this.testSize() },
          seed: this.seed(),
        },
      });
      return run.id;
    } catch (error) {
      this.createRunError.set(errorMessage(error));
      return null;
    } finally {
      this.creatingRun.set(false);
    }
  }

  private async createDataset(create: () => Promise<{ id: string }>): Promise<string | null> {
    this.creatingDataset.set(true);
    this.createDatasetError.set(null);
    try {
      return (await create()).id;
    } catch (error) {
      this.createDatasetError.set(errorMessage(error));
      return null;
    } finally {
      this.creatingDataset.set(false);
    }
  }

  private follow(runId: string): EventSource {
    const source = this.openEventSource(
      this.rootUrl + streamRunEvents.PATH.replace('{run_id}', encodeURIComponent(runId)),
    );
    source.onmessage = (message: MessageEvent<string>) => {
      const event = JSON.parse(message.data) as RunEventOut;
      this.events.update((events) => [...events, event]);
      if (event.type === 'done' || event.type === 'error') {
        source.close();
        this.runResource.reload();
      }
    };
    // Lost connection: fall back to fetching the run, which re-follows it if still going.
    source.onerror = () => {
      source.close();
      const timer = setTimeout(() => this.runResource.reload(), RECONNECT_MS);
      this.destroyRef.onDestroy(() => clearTimeout(timer));
    };
    return source;
  }
}
