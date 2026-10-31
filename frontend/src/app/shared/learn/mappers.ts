import { CodeAnnotationOut } from '../../core/api/models/code-annotation-out';
import { ResourceOut } from '../../core/api/models/resource-out';
import { VideoOut } from '../../core/api/models/video-out';
import { CodeAnnotation } from '../ui/code-explainer/code-explainer';
import { LearningResource } from '../ui/resource-card/resource-card';
import { VideoClipData } from '../ui/video-clip/video-clip';

/** API videos → the lookup `::video[name]` reads in MarkdownView. */
export function toVideoClips(videos: readonly VideoOut[]): Record<string, VideoClipData> {
  return Object.fromEntries(
    videos.map((v): [string, VideoClipData] => [v.id, { title: v.title, youtubeId: v.youtube_id }]),
  );
}

/** API code annotations → the CodeExplainer's input shape. */
export function toCodeAnnotations(code: {
  annotations: readonly CodeAnnotationOut[];
}): CodeAnnotation[] {
  return code.annotations.map((a) => ({
    id: a.anchor,
    startLine: a.start_line,
    endLine: a.end_line,
    short: a.short,
    deep: a.deep,
    concepts: a.concepts,
  }));
}

/** Papers get their own section; everything else is a learning resource. */
export function splitPapers(resources: readonly LearningResource[]) {
  return {
    papers: resources.filter((r) => r.type === 'paper'),
    others: resources.filter((r) => r.type !== 'paper'),
  };
}

export function toLearningResource(resource: ResourceOut): LearningResource {
  return {
    type: resource.type,
    title: resource.title,
    url: resource.url,
    author: resource.author,
    level: resource.level,
    durationMin: resource.duration_min,
    why: resource.why,
    note: resource.note,
  };
}
