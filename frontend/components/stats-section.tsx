'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { TrendingUp, Users, Clock, Award } from 'lucide-react'

export function StatsSection() {
  const stats = [
    {
      icon: TrendingUp,
      number: '10,000+',
      label: 'Papers Processed',
      description: 'Academic papers analyzed and indexed',
      color: 'from-blue-500 to-cyan-500',
    },
    {
      icon: Users,
      number: '5,000+',
      label: 'Active Researchers',
      description: 'Scientists and academics using our platform',
      color: 'from-green-500 to-emerald-500',
    },
    {
      icon: Clock,
      number: '2.5s',
      label: 'Average Response Time',
      description: 'Lightning-fast search and analysis',
      color: 'from-purple-500 to-pink-500',
    },
    {
      icon: Award,
      number: '95%',
      label: 'Accuracy Rate',
      description: 'Precision in document relevance scoring',
      color: 'from-orange-500 to-red-500',
    },
  ]

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
      },
    },
  }

  const itemVariants = {
    hidden: { y: 30, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.6,
      },
    },
  }

  return (
    <section className="py-24 bg-muted/50">
      <div className="container mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-primary to-purple-600 bg-clip-text text-transparent">
            Trusted by Researchers Worldwide
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Our platform powers research across universities, institutions, and
            organizations globally, delivering measurable results.
          </p>
        </motion.div>

        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8"
        >
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              variants={itemVariants}
              whileHover={{ 
                scale: 1.05,
                y: -10,
                transition: { duration: 0.2 }
              }}
              className="text-center group"
            >
              <div className="relative">
                {/* Background Glow */}
                <div 
                  className={`absolute inset-0 bg-gradient-to-r ${stat.color} opacity-20 blur-xl group-hover:opacity-30 transition-opacity duration-300 rounded-full`}
                />
                
                {/* Icon Container */}
                <div className={`relative w-16 h-16 mx-auto mb-4 bg-gradient-to-r ${stat.color} rounded-full flex items-center justify-center group-hover:scale-110 transition-transform duration-300`}>
                  <stat.icon className="w-8 h-8 text-white" />
                </div>
                
                {/* Number */}
                <motion.div
                  initial={{ scale: 0 }}
                  whileInView={{ scale: 1 }}
                  transition={{ 
                    duration: 0.6, 
                    delay: index * 0.1,
                    type: "spring",
                    stiffness: 200
                  }}
                  viewport={{ once: true }}
                  className="text-4xl md:text-5xl font-bold mb-2 bg-gradient-to-r from-foreground to-foreground/80 bg-clip-text text-transparent"
                >
                  {stat.number}
                </motion.div>
                
                {/* Label */}
                <h3 className="text-xl font-semibold mb-2 group-hover:text-primary transition-colors duration-300">
                  {stat.label}
                </h3>
                
                {/* Description */}
                <p className="text-muted-foreground text-sm leading-relaxed">
                  {stat.description}
                </p>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Additional Metrics */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          viewport={{ once: true }}
          className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto"
        >
          <div className="text-center p-6 glass-effect rounded-lg">
            <div className="text-2xl font-bold text-primary mb-2">24/7</div>
            <div className="text-muted-foreground">System Availability</div>
          </div>
          <div className="text-center p-6 glass-effect rounded-lg">
            <div className="text-2xl font-bold text-primary mb-2">99.9%</div>
            <div className="text-muted-foreground">Uptime Guarantee</div>
          </div>
          <div className="text-center p-6 glass-effect rounded-lg">
            <div className="text-2xl font-bold text-primary mb-2">150+</div>
            <div className="text-muted-foreground">Countries Served</div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
