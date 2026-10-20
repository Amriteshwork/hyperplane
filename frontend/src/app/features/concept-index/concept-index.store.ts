import { Injectable, computed, inject, resource, signal } from '@angular/core';

import { Api } from '../../core/api/api';
import { listConceptAreas } from '../../core/api/fn/learn/list-concept-areas';
import { ConceptAreaOut } from '../../core/api/models/concept-area-out';
import { errorMessage } from '../../core/http/http-errors';

/** Every concept by area, filtered by a search query. Provided by the index route. */
@Injectable()
export class ConceptIndexStore {
  private readonly api = inject(Api);

  private readonly page = resource({
    loader: () => this.api.invoke(listConceptAreas, { limit: 100 }),
  });

  readonly query = signal('');
  private readonly areas = computed(() => (this.page.hasValue() ? this.page.value().items : []));
  readonly total = computed(() => this.areas().reduce((n, a) => n + a.concepts.length, 0));
  readonly filtered = computed(() => filterAreas(this.areas(), this.query()));
  readonly matches = computed(() => this.filtered().reduce((n, a) => n + a.concepts.length, 0));
  readonly loading = this.page.isLoading;
  readonly error = computed(() => (this.page.error() ? errorMessage(this.page.error()) : null));

  reload(): void {
    this.page.reload();
  }
}

/** Concepts whose title or summary contains every word of the query; empty areas drop out. */
export function filterAreas(areas: readonly ConceptAreaOut[], query: string): ConceptAreaOut[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  return areas
    .map((area) => ({
      ...area,
      concepts: area.concepts.filter((c) => {
        const text = `${c.title} ${c.summary}`.toLowerCase();
        return words.every((w) => text.includes(w));
      }),
    }))
    .filter((area) => area.concepts.length);
}
