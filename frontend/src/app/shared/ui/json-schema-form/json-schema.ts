/**
 * The subset of JSON Schema that Pydantic v2 emits for a flat hyperparameter model, and its
 * normalisation into one `SchemaField` per property. Pure functions: no Angular.
 */

export type JsonPrimitive = string | number | boolean | null;
export type SchemaValues = Record<string, unknown>;

export interface JsonSchema {
  type?: string | string[];
  title?: string;
  description?: string;
  default?: unknown;
  format?: string;
  enum?: readonly JsonPrimitive[];
  const?: JsonPrimitive;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  multipleOf?: number;
  minLength?: number;
  maxLength?: number;
  properties?: Record<string, JsonSchema>;
  required?: readonly string[];
  anyOf?: readonly JsonSchema[];
  allOf?: readonly JsonSchema[];
  $ref?: string;
  $defs?: Record<string, JsonSchema>;
  /** Our extension (Pydantic `json_schema_extra`): beginner params show first. */
  level?: FieldLevel;
}

export type FieldLevel = 'beginner' | 'advanced';

export interface SchemaField {
  readonly key: string;
  readonly title: string;
  readonly description: string | null;
  /** JSON type after unwrapping `Optional[...]`; `null` if the schema doesn't say. */
  readonly type: string | null;
  readonly format: string | null;
  readonly enum: readonly JsonPrimitive[] | null;
  readonly required: boolean;
  /** `Optional[...]` in Pydantic: the user may leave it empty. */
  readonly nullable: boolean;
  readonly default: unknown;
  readonly level: FieldLevel;
  readonly minimum: number | null;
  readonly maximum: number | null;
  readonly exclusiveMinimum: number | null;
  readonly exclusiveMaximum: number | null;
  readonly minLength: number | null;
  readonly maxLength: number | null;
}

/** Flattens a Pydantic object schema into ordered fields (schema property order). */
export function schemaFields(schema: JsonSchema): SchemaField[] {
  const required = new Set(schema.required ?? []);
  return Object.entries(schema.properties ?? {}).map(([key, raw]) =>
    toField(key, raw, required.has(key), schema.$defs ?? {}),
  );
}

/** Default value for every property that declares one. */
export function schemaDefaults(schema: JsonSchema): SchemaValues {
  const values: SchemaValues = {};
  for (const field of schemaFields(schema)) {
    if (field.default !== undefined) values[field.key] = field.default;
  }
  return values;
}

function toField(
  key: string,
  raw: JsonSchema,
  required: boolean,
  defs: Record<string, JsonSchema>,
): SchemaField {
  const variants = (raw.anyOf ?? [raw]).map((s) => resolve(s, defs));
  const nullable = variants.some(isNullSchema);
  // Pydantic puts constraints on the non-null branch of `Optional[...]`.
  const inner = { ...(variants.find((s) => !isNullSchema(s)) ?? {}), ...strip(raw) };
  return {
    key,
    title: raw.title ?? humanise(key),
    description: raw.description ?? inner.description ?? null,
    type: firstType(inner.type),
    format: inner.format ?? null,
    // A one-value `Literal` is emitted as `const`: a choice of one (plus None, if nullable).
    enum: inner.enum ?? (inner.const !== undefined ? [inner.const] : null),
    required,
    nullable,
    default: raw.default,
    level: raw.level ?? inner.level ?? 'beginner',
    minimum: inner.minimum ?? null,
    maximum: inner.maximum ?? null,
    exclusiveMinimum: inner.exclusiveMinimum ?? null,
    exclusiveMaximum: inner.exclusiveMaximum ?? null,
    minLength: inner.minLength ?? null,
    maxLength: inner.maxLength ?? null,
  };
}

/** Inlines `$ref` and single-item `allOf` (how Pydantic references enums). */
function resolve(schema: JsonSchema, defs: Record<string, JsonSchema>): JsonSchema {
  const ref = schema.$ref ?? (schema.allOf?.length === 1 ? schema.allOf[0].$ref : undefined);
  if (!ref) return schema;
  const target = defs[ref.replace('#/$defs/', '')] ?? {};
  return { ...target, ...strip(schema) };
}

/** Field-level keys that must not be overwritten by a resolved definition. */
function strip(schema: JsonSchema): JsonSchema {
  const rest = { ...schema };
  delete rest.anyOf;
  delete rest.allOf;
  delete rest.$ref;
  delete rest.title;
  return rest;
}

function isNullSchema(schema: JsonSchema): boolean {
  return schema.type === 'null';
}

function firstType(type: JsonSchema['type']): string | null {
  if (Array.isArray(type)) return type.find((t) => t !== 'null') ?? null;
  return type ?? null;
}

function humanise(key: string): string {
  const words = key.replace(/_/g, ' ').trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}
