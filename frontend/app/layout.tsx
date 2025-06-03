import './globals.css'
import { Inter } from 'next/font/google'
import { ThemeProvider } from '@/components/theme-provider'
import { Toaster } from 'react-hot-toast'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:3000'),
  title: 'Research Assistant | AI-Powered Research Tool',
  description: 'Advanced AI-powered research assistant with hybrid search RAG capabilities for academic research and paper analysis.',
  keywords: 'AI, research, RAG, machine learning, academic papers, arXiv',
  authors: [{ name: 'Research Assistant Team' }],
  openGraph: {
    title: 'Research Assistant | AI-Powered Research Tool',
    description: 'Advanced AI-powered research assistant with hybrid search RAG capabilities.',
    url: 'https://research-assistant-flx.streamlit.app/',
    siteName: 'Research Assistant',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Research Assistant | AI-Powered Research Tool',
    description: 'Advanced AI-powered research assistant with hybrid search RAG capabilities.',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          {children}
          <Toaster 
            position="bottom-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: 'hsl(var(--card))',
                color: 'hsl(var(--card-foreground))',
                border: '1px solid hsl(var(--border))',
              },
            }}
          />
        </ThemeProvider>
      </body>
    </html>
  )
}
