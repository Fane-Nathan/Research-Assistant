'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { Navigation } from '@/components/navigation'
import { HeroSection } from '@/components/hero-section'
import { FeaturesSection } from '@/components/features-section'
import { DemoSection } from '@/components/demo-section'
import { StatsSection } from '@/components/stats-section'
import { CTASection } from '@/components/cta-section'
import { Footer } from '@/components/footer'

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <main className="relative">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
        >
          <HeroSection />
          <FeaturesSection />
          <StatsSection />
          <DemoSection />
          <CTASection />
        </motion.div>
      </main>
      
      <Footer />
    </div>
  )
}
