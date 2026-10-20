export interface NavItem {
  label: string;
  path: string;
}

/** Primary navigation, shared by the desktop bar and the mobile menu. */
export const NAV_ITEMS: readonly NavItem[] = [
  { label: 'Learn', path: '/learn' },
  { label: 'Concepts', path: '/concepts' },
  { label: 'Lab', path: '/lab' },
  { label: 'Runs', path: '/runs' },
];
