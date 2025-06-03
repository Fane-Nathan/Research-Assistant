'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { Search, FileText, Brain, Zap, Globe, Database, MessageSquare, TrendingUp } from 'lucide-react'
import { Card } from '@/components/ui/card'

export function FeaturesSection() {
  const features = [
    {
      icon: Search,
      title: 'Hybrid Search',
      description: 'Combine semantic search with keyword search using advanced BM25 and vector embeddings for superior relevance.',
      color: 'from-blue-500 to-cyan-500',
    },
    {
      icon: Brain,
      title: 'RAG Technology',
      description: 'Retrieval-Augmented Generation with configurable modes - strict local data or hybrid with general knowledge.',
      color: 'from-purple-500 to-pink-500',
    },
    {
      icon: FileText,
      title: 'arXiv Integration',
      description: 'Direct access to academic papers from arXiv with intelligent parsing and content extraction.',
      color: 'from-green-500 to-emerald-500',
    },
    {
      icon: Globe,
      title: 'Web Crawling',
      description: 'Advanced web crawling with JavaScript rendering support using Playwright for dynamic content.',
      color: 'from-orange-500 to-red-500',
    },
    {
      icon: Database,
      title: 'Smart Processing',
      description: 'Intelligent document chunking, embedding generation, and efficient data storage with local persistence.',
      color: 'from-indigo-500 to-purple-500',
    },
    {
      icon: MessageSquare,
      title: 'Interactive Chat',
      description: 'Natural language interface powered by advanced LLMs with contextual understanding and streaming responses.',
      color: 'from-teal-500 to-blue-500',
    },
    {
      icon: Zap,
      title: 'Fast Performance',
      description: 'Optimized algorithms with caching, parallel processing, and efficient vector similarity search.',
      color: 'from-yellow-500 to-orange-500',
    },
    {
      icon: TrendingUp,
      title: 'Analytics & Insights',
      description: 'Comprehensive analytics on research patterns, document relevance scoring, and usage statistics.',
      color: 'from-pink-500 to-rose-500',
    },
  ]

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  }

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.6,
      },
    },
  }

  return (
    <section id="features" className="py-24 bg-muted/50">
      <div className="container mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-primary to-purple-600 bg-clip-text text-transparent">
            Powerful Features
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Discover the advanced capabilities that make our Research Assistant
            the perfect tool for academic research and document analysis.
          </p>
        </motion.div>

        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        >
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              variants={itemVariants}
              whileHover={{ 
                scale: 1.02, 
                y: -5,
                transition: { duration: 0.2 }
              }}
            >
              <Card className="p-6 h-full glass-effect hover:shadow-xl transition-all duration-300 border-0 relative overflow-hidden group">
                {/* Background Gradient */}
                <div 
                  className={`absolute inset-0 bg-gradient-to-br ${feature.color} opacity-5 group-hover:opacity-10 transition-opacity duration-300`}
                />
                
                <div className="relative z-10">
                  <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${feature.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}>
                    <feature.icon className="w-6 h-6 text-white" />
                  </div>
                  
                  <h3 className="text-xl font-semibold mb-3 group-hover:text-primary transition-colors duration-300">
                    {feature.title}
                  </h3>
                  
                  <p className="text-muted-foreground leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              </Card>
            </motion.div>
          ))}
        </motion.div>

        {/* Technical Details */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          viewport={{ once: true }}
          className="mt-20 text-center"
        >
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            <div className="p-6 rounded-lg glass-effect">
              <div className="text-3xl font-bold text-primary mb-2">3x</div>
              <div className="text-muted-foreground">Faster than traditional search</div>
            </div>
            <div className="p-6 rounded-lg glass-effect">
              <div className="text-3xl font-bold text-primary mb-2">99.9%</div>
              <div className="text-muted-foreground">Uptime reliability</div>
            </div>
            <div className="p-6 rounded-lg glass-effect">
              <div className="text-3xl font-bold text-primary mb-2">24/7</div>
              <div className="text-muted-foreground">Availability</div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
