'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Search, 
  FileText, 
  Download, 
  Star, 
  ExternalLink, 
  Clock,
  Brain,
  Loader2,
  Filter,
  SortDesc,
  RefreshCw,
  BookOpen,
  Users,
  Calendar,
  Tag
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { toast } from 'react-hot-toast'

interface ResearchPaper {
  id: string
  title: string
  authors: string[]
  abstract: string
  publishedDate: string
  url: string
  relevanceScore: number
  citations: number
  categories: string[]
  keywords: string[]
}

interface SearchResult {
  papers: ResearchPaper[]
  totalResults: number
  searchTime: number
  query: string
}

export function ResearchInterface() {
  const [query, setQuery] = useState('')
  const [isSearching, setIsSearching] = useState(false)
  const [searchResults, setSearchResults] = useState<SearchResult | null>(null)
  const [selectedPaper, setSelectedPaper] = useState<ResearchPaper | null>(null)
  const [searchHistory, setSearchHistory] = useState<string[]>([])
  const [filters, setFilters] = useState({
    sortBy: 'relevance',
    dateRange: 'all',
    category: 'all'
  })

  const handleSearch = async (searchQuery: string) => {
    if (!searchQuery.trim()) return

    setIsSearching(true)
    try {
      // Mock API call - replace with actual backend integration
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      const mockResults: SearchResult = {
        papers: [
          {
            id: '1',
            title: 'Attention Is All You Need',
            authors: ['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar'],
            abstract: 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...',
            publishedDate: '2017-06-12',
            url: 'https://arxiv.org/abs/1706.03762',
            relevanceScore: 98,
            citations: 50000,
            categories: ['Machine Learning', 'Neural Networks'],
            keywords: ['transformers', 'attention', 'neural networks']
          },
          {
            id: '2',
            title: 'BERT: Pre-training of Deep Bidirectional Transformers',
            authors: ['Jacob Devlin', 'Ming-Wei Chang', 'Kenton Lee'],
            abstract: 'We introduce a new language representation model called BERT...',
            publishedDate: '2018-10-11',
            url: 'https://arxiv.org/abs/1810.04805',
            relevanceScore: 95,
            citations: 45000,
            categories: ['Natural Language Processing', 'Machine Learning'],
            keywords: ['BERT', 'transformers', 'language model']
          },
          {
            id: '3',
            title: 'GPT-4 Technical Report',
            authors: ['OpenAI'],
            abstract: 'We report the development of GPT-4, a large-scale, multimodal model...',
            publishedDate: '2023-03-15',
            url: 'https://arxiv.org/abs/2303.08774',
            relevanceScore: 92,
            citations: 8000,
            categories: ['Artificial Intelligence', 'Language Models'],
            keywords: ['GPT-4', 'multimodal', 'large language model']
          }
        ],
        totalResults: 3,
        searchTime: 1.2,
        query: searchQuery
      }

      setSearchResults(mockResults)
      setSearchHistory(prev => [searchQuery, ...prev.slice(0, 4)])
      toast.success(`Found ${mockResults.totalResults} relevant papers`)
    } catch (error) {
      toast.error('Search failed. Please try again.')
    } finally {
      setIsSearching(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch(query)
    }
  }

  return (
    <section className="py-24 bg-background">
      <div className="container mx-auto px-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-primary to-purple-600 bg-clip-text text-transparent">
            Interactive Research Interface
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Experience our AI-powered research assistant in action. Search through academic papers,
            analyze content, and discover insights with intelligent recommendations.
          </p>
        </motion.div>

        {/* Search Interface */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          viewport={{ once: true }}
          className="max-w-4xl mx-auto mb-8"
        >
          <div className="relative">
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-muted-foreground w-5 h-5" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Search for research papers, concepts, or authors..."
              className="pl-12 pr-32 py-6 text-lg glass-effect border-2 focus:border-primary"
            />
            <Button
              onClick={() => handleSearch(query)}
              disabled={isSearching || !query.trim()}
              className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-gradient-to-r from-primary to-purple-600 hover:from-primary/90 hover:to-purple-600/90"
            >
              {isSearching ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Search className="w-4 h-4" />
              )}
              Search
            </Button>
          </div>

          {/* Search History */}
          {searchHistory.length > 0 && (
            <div className="mt-4">
              <p className="text-sm text-muted-foreground mb-2">Recent searches:</p>
              <div className="flex flex-wrap gap-2">
                {searchHistory.map((historyQuery, index) => (
                  <motion.button
                    key={index}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.1 }}
                    onClick={() => setQuery(historyQuery)}
                    className="px-3 py-1 text-sm bg-muted hover:bg-muted/80 rounded-full transition-colors"
                  >
                    {historyQuery}
                  </motion.button>
                ))}
              </div>
            </div>
          )}
        </motion.div>

        {/* Search Results */}
        <AnimatePresence>
          {searchResults && (
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -30 }}
              transition={{ duration: 0.6 }}
              className="max-w-6xl mx-auto"
            >
              {/* Results Header */}
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 p-4 glass-effect rounded-lg">
                <div>
                  <h3 className="text-xl font-semibold">
                    {searchResults.totalResults} results for "{searchResults.query}"
                  </h3>
                  <p className="text-muted-foreground">
                    Search completed in {searchResults.searchTime}s
                  </p>
                </div>
                
                <div className="flex items-center gap-2 mt-4 md:mt-0">
                  <Button variant="outline" size="sm">
                    <Filter className="w-4 h-4 mr-2" />
                    Filter
                  </Button>
                  <Button variant="outline" size="sm">
                    <SortDesc className="w-4 h-4 mr-2" />
                    Sort
                  </Button>
                  <Button variant="outline" size="sm">
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Refresh
                  </Button>
                </div>
              </div>

              {/* Results Grid */}
              <div className="grid gap-6">
                {searchResults.papers.map((paper, index) => (
                  <motion.div
                    key={paper.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    whileHover={{ y: -5 }}
                  >
                    <Card className="p-6 glass-effect hover:shadow-xl transition-all duration-300 cursor-pointer"
                          onClick={() => setSelectedPaper(paper)}>
                      <div className="flex justify-between items-start mb-4">
                        <div className="flex-1">
                          <h4 className="text-xl font-semibold mb-2 hover:text-primary transition-colors">
                            {paper.title}
                          </h4>
                          <div className="flex items-center gap-4 text-sm text-muted-foreground mb-3">
                            <div className="flex items-center gap-1">
                              <Users className="w-4 h-4" />
                              {paper.authors.slice(0, 3).join(', ')}
                              {paper.authors.length > 3 && ` +${paper.authors.length - 3} more`}
                            </div>
                            <div className="flex items-center gap-1">
                              <Calendar className="w-4 h-4" />
                              {new Date(paper.publishedDate).getFullYear()}
                            </div>
                            <div className="flex items-center gap-1">
                              <Star className="w-4 h-4" />
                              {paper.citations.toLocaleString()} citations
                            </div>
                          </div>
                        </div>
                        
                        <div className="flex flex-col items-end gap-2">
                          <Badge variant="secondary" className="bg-gradient-to-r from-primary/10 to-purple-600/10">
                            {paper.relevanceScore}% relevance
                          </Badge>
                          <Button variant="ghost" size="sm">
                            <ExternalLink className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>

                      <p className="text-muted-foreground mb-4 line-clamp-3">
                        {paper.abstract}
                      </p>

                      <div className="flex flex-wrap gap-2 mb-4">
                        {paper.categories.map((category) => (
                          <Badge key={category} variant="outline" className="text-xs">
                            {category}
                          </Badge>
                        ))}
                      </div>

                      <div className="flex flex-wrap gap-1">
                        {paper.keywords.map((keyword) => (
                          <span key={keyword} className="px-2 py-1 text-xs bg-primary/10 text-primary rounded">
                            {keyword}
                          </span>
                        ))}
                      </div>
                    </Card>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Demo Actions */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          viewport={{ once: true }}
          className="text-center mt-12"
        >
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-3xl mx-auto">
            <Button
              variant="outline"
              className="p-6 h-auto flex flex-col items-center gap-2 glass-effect"
              onClick={() => handleSearch('transformer neural networks')}
            >
              <Brain className="w-8 h-8 text-primary" />
              <span className="font-medium">AI & ML</span>
              <span className="text-sm text-muted-foreground">Explore AI research</span>
            </Button>
            
            <Button
              variant="outline"
              className="p-6 h-auto flex flex-col items-center gap-2 glass-effect"
              onClick={() => handleSearch('quantum computing algorithms')}
            >
              <FileText className="w-8 h-8 text-primary" />
              <span className="font-medium">Quantum Computing</span>
              <span className="text-sm text-muted-foreground">Latest breakthroughs</span>
            </Button>
            
            <Button
              variant="outline"
              className="p-6 h-auto flex flex-col items-center gap-2 glass-effect"
              onClick={() => handleSearch('climate change research')}
            >
              <BookOpen className="w-8 h-8 text-primary" />
              <span className="font-medium">Climate Science</span>
              <span className="text-sm text-muted-foreground">Environmental studies</span>
            </Button>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
