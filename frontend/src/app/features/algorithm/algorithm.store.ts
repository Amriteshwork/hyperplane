import { Injectable, computed, inject, resource, signal } from '@angular/core';

import { Api } from '../../core/api/api';
import { getAlgorithm } from '../../core/api/fn/learn/get-algorithm';
import { getAlgorithmCode } from '../../core/api/fn/learn/get-algorithm-code';
import { errorMessage, isNotFound } from '../../core/http/http-errors';
import { toCodeAnnotations, toLearningResource } from '../../shared/learn/mappers';

/** State for one algorithm page and its tabs. Provided by the algorithm route. */
@Injectable()
export class AlgorithmStore {
  private readonly api = inject(Api);

  readonly algorithmId = signal<string | null>(null);
  readonly variant = signal<string | null>(null);

  private readonly detail = resource({
    params: () => this.algorithmId() ?? undefined,
    loader: ({ params }) => this.api.invoke(getAlgorithm, { algorithm_id: params }),
  });

  private readonly codeResource = resource({
    params: () => {
      const id = this.algorithmId();
      const variant = this.variant();
      return id && variant ? { algorithm_id: id, variant } : undefined;
    },
    loader: ({ params }) => this.api.invoke(getAlgorithmCode, params),
  });

  readonly algorithm = computed(() => (this.detail.hasValue() ? this.detail.value() : null));
  readonly loading = this.detail.isLoading;
  readonly notFound = computed(() => isNotFound(this.detail.error()));
  readonly error = computed(() => (this.detail.error() ? errorMessage(this.detail.error()) : null));

  readonly resources = computed(() => this.algorithm()?.resources.map(toLearningResource) ?? []);
  readonly papers = computed(() => this.resources().filter((r) => r.type === 'paper'));

  readonly code = computed(() => {
    if (!this.codeResource.hasValue()) return null;
    const code = this.codeResource.value();
    return { ...code, explained: toCodeAnnotations(code) };
  });
  readonly codeLoading = this.codeResource.isLoading;
  readonly codeError = computed(() =>
    this.codeResource.error() ? errorMessage(this.codeResource.error()) : null,
  );

  reload(): void {
    this.detail.reload();
  }
}
