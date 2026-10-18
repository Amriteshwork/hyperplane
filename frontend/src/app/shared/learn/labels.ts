import { TaskType } from '../../core/api/models/task-type';

export const TASK_LABELS: Record<TaskType, string> = {
  regression: 'Regression',
  binary: 'Binary classification',
  multiclass: 'Multiclass classification',
};
