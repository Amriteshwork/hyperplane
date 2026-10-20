import { Injectable, computed, inject, resource, signal } from '@angular/core';

import { Api } from '../../core/api/api';
import { getConcept } from '../../core/api/fn/learn/get-concept';
import { errorMessage, isNotFound } from '../../core/http/http-errors';
import { splitPapers, toLearningResource } from '../../shared/learn/mappers';

@Injectable()
export class ConceptStore {
  private readonly api = inject(Api);

  readonly conceptId = signal<string | null>(null);

  private readonly detail = resource({
    params: () => this.conceptId() ?? undefined,
    loader: ({ params }) => this.api.invoke(getConcept, { concept_id: params }),
  });

  readonly concept = computed(() => (this.detail.hasValue() ? this.detail.value() : null));
  private readonly split = computed(() =>
    splitPapers(this.concept()?.resources.map(toLearningResource) ?? []),
  );
  readonly resources = computed(() => this.split().others);
  readonly papers = computed(() => this.split().papers);
  readonly notFound = computed(() => isNotFound(this.detail.error()));
  readonly error = computed(() => (this.detail.error() ? errorMessage(this.detail.error()) : null));

  reload(): void {
    this.detail.reload();
  }
}
