import { SchemaField, SchemaValues } from './json-schema';

/** A user-facing message for the first rule `value` breaks, or `null` if it is valid. */
export function validateField(field: SchemaField, value: unknown): string | null {
  if (value === undefined || value === null || value === '') {
    return field.required && !field.nullable ? 'This field is required.' : null;
  }
  if (field.enum && !field.enum.includes(value as never)) return 'Choose one of the options.';
  if (field.type === 'number' || field.type === 'integer') return validateNumber(field, value);
  if (field.type === 'string' && typeof value === 'string') return validateString(field, value);
  return null;
}

/** Messages keyed by field; empty when every field is valid. */
export function validateValues(
  fields: readonly SchemaField[],
  values: SchemaValues,
): Record<string, string> {
  const errors: Record<string, string> = {};
  for (const field of fields) {
    const message = validateField(field, values[field.key]);
    if (message) errors[field.key] = message;
  }
  return errors;
}

function validateNumber(field: SchemaField, value: unknown): string | null {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'Enter a number.';
  if (field.type === 'integer' && !Number.isInteger(value)) return 'Enter a whole number.';
  if (field.minimum !== null && value < field.minimum) return `Must be at least ${field.minimum}.`;
  if (field.maximum !== null && value > field.maximum) return `Must be at most ${field.maximum}.`;
  if (field.exclusiveMinimum !== null && value <= field.exclusiveMinimum) {
    return `Must be greater than ${field.exclusiveMinimum}.`;
  }
  if (field.exclusiveMaximum !== null && value >= field.exclusiveMaximum) {
    return `Must be less than ${field.exclusiveMaximum}.`;
  }
  return null;
}

function validateString(field: SchemaField, value: string): string | null {
  if (field.minLength !== null && value.length < field.minLength) {
    return `Use at least ${field.minLength} characters.`;
  }
  if (field.maxLength !== null && value.length > field.maxLength) {
    return `Use at most ${field.maxLength} characters.`;
  }
  return null;
}
