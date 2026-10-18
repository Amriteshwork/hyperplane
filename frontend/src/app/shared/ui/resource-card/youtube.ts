/** The 11-character video id for youtube.com / youtu.be links, or `null` for anything else. */
export function youtubeId(url: string): string | null {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return null;
  }
  const host = parsed.hostname.replace(/^(www|m)\./, '');
  let id: string | null = null;
  if (host === 'youtu.be') id = parsed.pathname.slice(1);
  else if (host === 'youtube.com' || host === 'youtube-nocookie.com') {
    id =
      parsed.searchParams.get('v') ??
      parsed.pathname.match(/^\/(?:embed|shorts)\/([^/]+)/)?.[1] ??
      null;
  }
  return id && /^[\w-]{11}$/.test(id) ? id : null;
}
