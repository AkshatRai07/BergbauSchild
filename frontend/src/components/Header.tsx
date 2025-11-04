'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const navItems = [
  { name: 'Dashboard', href: '/' },
  { name: 'Digital Officer (Chat)', href: '/chat' },
  { name: 'Classify Incident', href: '/classify' },
  { name: 'Audit Reports', href: '/reports' },
  { name: 'DGMS Updates', href: '/updates' },
];

export default function Header() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 w-full border-b border-gray-700 bg-gray-900/80 backdrop-blur-md">
      <nav className="mx-auto flex max-w-7xl items-center justify-between p-4">
        <div className="flex items-center gap-2">
          <span className="text-xl font-bold text-cyan-400">⛏️</span>
          <span className="font-semibold text-white">MineSafe AI</span>
        </div>
        <div className="flex gap-4">
          {navItems.map((item) => (
            <Link
              key={item.name}
              href={item.href}
              className={`rounded-md px-3 py-1 text-sm font-medium transition-colors
                ${
                  pathname === item.href
                    ? 'bg-cyan-600 text-white'
                    : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                }
              `}
            >
              {item.name}
            </Link>
          ))}
        </div>
      </nav>
    </header>
  );
}