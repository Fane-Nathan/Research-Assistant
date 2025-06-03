'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Play, Code, Database, MessageCircle, ArrowRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

export function DemoSection() {
  const [activeDemo, setActiveDemo] = useState('search')

  const demoTabs = [
    {
      id: 'search',
      label: 'Smart Search',
      icon: Database,
      title: 'Hybrid Search in Action',
      description: 'See how our AI combines semantic and keyword search for the most relevant results.',
      preview: (
        <div className="space-y-4">
          <div className="p-4 bg-muted rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="text-sm text-muted-foreground">Query Processing</span>
            </div>
            <p className="font-mono text-sm">"What are the latest advances in transformer architectures?"</p>
          </div>
          <div className="space-y-2">
            {[
              { title: "Attention Is All You Need", relevance: 98, type: "arXiv" },
              { title: "BERT: Pre-training of Deep Bidirectional Transformers", relevance: 95, type: "arXiv" },
              { title: "GPT-4 Technical Report", relevance: 92, type: "Web" },
            ].map((result, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.2 }}
                className="p-3 bg-card rounded-lg border flex justify-between items-center"
              >
                <div>
                  <h4 className="font-medium">{result.title}</h4>
                  <span className="text-xs text-muted-foreground">{result.type}</span>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium text-primary">{result.relevance}%</div>
                  <div className="text-xs text-muted-foreground">relevance</div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      )
    },
    {
      id: 'chat',
      label: 'AI Chat',
      icon: MessageCircle,
      title: 'Intelligent Conversations',
      description: 'Experience natural language interaction with contextual understanding.',
      preview: (
        <div className="space-y-4">
          <div className="space-y-3">
            <div className="flex justify-end">
              <div className="bg-primary text-primary-foreground p-3 rounded-lg max-w-xs">
                Explain the key differences between BERT and GPT models
              </div>
            </div>
            <div className="flex justify-start">
              <div className="bg-muted p-3 rounded-lg max-w-xs">
                Based on the research papers in your collection, here are the key differences:
                
                **BERT** (Bidirectional Encoder Representations from Transformers):
                - Uses bidirectional context
                - Encoder-only architecture
                - Designed for understanding tasks
                
                **GPT** (Generative Pre-trained Transformer):
                - Uses unidirectional (left-to-right) context
                - Decoder-only architecture  
                - Designed for generation tasks
                
                Would you like me to elaborate on any specific aspect?
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'analysis',
      label: 'Document Analysis',
      icon: Code,
      title: 'Deep Document Understanding',
      description: 'Watch how we extract insights and analyze academic papers automatically.',
      preview: (
        <div className="space-y-4">
          <div className="p-4 bg-muted rounded-lg">
            <h4 className="font-medium mb-2">Document: "Attention Is All You Need"</h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Authors:</span>
                <p>Vaswani, A. et al.</p>
              </div>
              <div>
                <span className="text-muted-foreground">Year:</span>
                <p>2017</p>
              </div>
              <div>
                <span className="text-muted-foreground">Citations:</span>
                <p>50,000+</p>
              </div>
              <div>
                <span className="text-muted-foreground">Impact:</span>
                <p>Revolutionary</p>
              </div>
            </div>
          </div>
          <div className="space-y-2">
            <div className="p-3 bg-card rounded-lg border">
              <h5 className="font-medium text-sm">Key Concepts Extracted:</h5>
              <div className="flex flex-wrap gap-1 mt-1">
                {["Self-attention", "Transformer", "Multi-head attention", "Position encoding"].map((concept) => (
                  <span key={concept} className="px-2 py-1 bg-primary/10 text-primary text-xs rounded">
                    {concept}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )
    }
  ]

  return (
    <section id="demo" className="py-24">
      <div className="container mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-primary to-purple-600 bg-clip-text text-transparent">
            See It In Action
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Experience the power of our Research Assistant through interactive demos
            showcasing real-world use cases and capabilities.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          viewport={{ once: true }}
          className="max-w-6xl mx-auto"
        >
          <Tabs value={activeDemo} onValueChange={setActiveDemo} className="w-full">
            <TabsList className="grid w-full grid-cols-3 mb-8">
              {demoTabs.map((tab) => (
                <TabsTrigger key={tab.id} value={tab.id} className="flex items-center space-x-2">
                  <tab.icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </TabsTrigger>
              ))}
            </TabsList>

            {demoTabs.map((tab) => (
              <TabsContent key={tab.id} value={tab.id}>
                <div className="grid lg:grid-cols-2 gap-8 items-center">
                  <motion.div
                    initial={{ opacity: 0, x: -30 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.6 }}
                    viewport={{ once: true }}
                  >
                    <h3 className="text-3xl font-bold mb-4">{tab.title}</h3>
                    <p className="text-lg text-muted-foreground mb-6 leading-relaxed">
                      {tab.description}
                    </p>
                    
                    <div className="space-y-4">
                      <Button className="bg-gradient-to-r from-primary to-purple-600 hover:from-primary/90 hover:to-purple-600/90">
                        <Play className="w-4 h-4 mr-2" />
                        Try Interactive Demo
                      </Button>
                      
                      <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                        <div className="flex items-center space-x-1">
                          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                          <span>Live Demo Available</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                          <span>No Registration Required</span>
                        </div>
                      </div>
                    </div>
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, x: 30 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.6, delay: 0.2 }}
                    viewport={{ once: true }}
                  >
                    <Card className="p-6 glass-effect">
                      {tab.preview}
                    </Card>
                  </motion.div>
                </div>
              </TabsContent>
            ))}
          </Tabs>
        </motion.div>

        {/* CTA Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          viewport={{ once: true }}
          className="mt-16 text-center"
        >
          <Card className="p-8 glass-effect max-w-2xl mx-auto">
            <h3 className="text-2xl font-bold mb-4">Ready to Transform Your Research?</h3>
            <p className="text-muted-foreground mb-6">
              Join thousands of researchers who are already using our AI-powered assistant
              to accelerate their academic work and discover new insights.
            </p>
            <Button 
              size="lg" 
              className="bg-gradient-to-r from-primary to-purple-600 hover:from-primary/90 hover:to-purple-600/90"
            >
              Start Your Free Trial
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
          </Card>
        </motion.div>
      </div>
    </section>
  )
}
