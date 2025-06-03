'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  MessageCircle, 
  Send, 
  Bot, 
  User, 
  Loader2,
  FileText,
  Download,
  Copy,
  ThumbsUp,
  ThumbsDown,
  RefreshCw,
  Settings,
  Volume2,
  VolumeX
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { toast } from 'react-hot-toast'

interface Message {
  id: string
  type: 'user' | 'assistant'
  content: string
  timestamp: Date
  sources?: Source[]
  isLoading?: boolean
}

interface Source {
  title: string
  url: string
  relevance: number
  snippet: string
}

interface Settings {
  ragMode: boolean
  includeGeneralKnowledge: boolean
  maxSources: number
}

export function AIChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'assistant',
      content: 'Hello! I\'m your AI research assistant. I can help you explore academic papers, explain complex concepts, and provide insights based on the latest research. What would you like to know?',
      timestamp: new Date(),
    }
  ])
  const [inputMessage, setInputMessage] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [isSpeechEnabled, setIsSpeechEnabled] = useState(false)
  const [settings, setSettings] = useState<Settings>({
    ragMode: true,
    includeGeneralKnowledge: false,
    maxSources: 5
  })

  const suggestedQuestions = [
    "Explain the difference between BERT and GPT models",
    "What are the latest advances in quantum computing?",
    "How do transformer architectures work?",
    "What is retrieval-augmented generation?",
    "Compare different attention mechanisms",
    "Explain neural machine translation"
  ]

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsTyping(true)

    // Add loading message
    const loadingMessage: Message = {
      id: (Date.now() + 1).toString(),
      type: 'assistant',
      content: '',
      timestamp: new Date(),
      isLoading: true
    }
    setMessages(prev => [...prev, loadingMessage])

    try {
      // Mock API call - replace with actual backend integration
      await new Promise(resolve => setTimeout(resolve, 2000))

      const mockSources: Source[] = [
        {
          title: "Attention Is All You Need",
          url: "https://arxiv.org/abs/1706.03762",
          relevance: 95,
          snippet: "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks..."
        },
        {
          title: "BERT: Pre-training of Deep Bidirectional Transformers",
          url: "https://arxiv.org/abs/1810.04805",
          relevance: 88,
          snippet: "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers..."
        }
      ]

      const assistantMessage: Message = {
        id: (Date.now() + 2).toString(),
        type: 'assistant',
        content: `Based on the research papers in your collection, I can explain this concept in detail.\n\n**Transformer architectures** represent a fundamental shift in how we approach sequence-to-sequence tasks in machine learning. Unlike traditional RNNs or CNNs, transformers rely entirely on attention mechanisms to process input sequences.\n\n**Key Components:**\n1. **Self-Attention Mechanism**: Allows the model to weigh the importance of different parts of the input sequence\n2. **Multi-Head Attention**: Enables the model to focus on different aspects of the input simultaneously\n3. **Position Encoding**: Provides sequence order information since transformers don't have inherent sequential processing\n\n**Advantages:**\n- Parallelizable training (unlike RNNs)\n- Better long-range dependency modeling\n- More interpretable attention patterns\n\nThis architecture has become the foundation for models like BERT, GPT, and T5, revolutionizing NLP and beyond.`,
        timestamp: new Date(),
        sources: mockSources
      }

      // Remove loading message and add actual response
      setMessages(prev => prev.slice(0, -1).concat(assistantMessage))
      toast.success('Response generated successfully')
    } catch (error) {
      setMessages(prev => prev.slice(0, -1))
      toast.error('Failed to generate response. Please try again.')
    } finally {
      setIsTyping(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage(inputMessage)
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    toast.success('Copied to clipboard')
  }
  const speakText = (text: string) => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text)
      speechSynthesis.speak(utterance)
    }
  }

  return (
    <section className="py-24 bg-muted/50">
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
            AI Chat Assistant
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Have intelligent conversations with our AI assistant. Ask questions about research papers,
            get explanations of complex concepts, and discover new insights through natural language interaction.
          </p>
        </motion.div>

        <div className="max-w-6xl mx-auto">
          <div className="grid lg:grid-cols-4 gap-6">
            {/* Settings Panel */}
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="lg:col-span-1"
            >
              <Card className="glass-effect h-fit">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Settings className="w-5 h-5" />
                    Settings
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">RAG Mode</label>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Use local papers</span>
                      <Switch
                        checked={settings.ragMode}                        onCheckedChange={(checked: boolean) => 
                          setSettings(prev => ({ ...prev, ragMode: checked }))
                        }
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">General Knowledge</label>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Include web knowledge</span>                      <Switch
                        checked={settings.includeGeneralKnowledge}
                        onCheckedChange={(checked: boolean) => 
                          setSettings(prev => ({ ...prev, includeGeneralKnowledge: checked }))
                        }
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Speech</label>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Text-to-speech</span>
                      <Switch
                        checked={isSpeechEnabled}
                        onCheckedChange={setIsSpeechEnabled}
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Suggested Questions */}
              <Card className="glass-effect mt-6">
                <CardHeader>
                  <CardTitle className="text-lg">Suggested Questions</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {suggestedQuestions.map((question, index) => (
                    <motion.button
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      onClick={() => setInputMessage(question)}
                      className="w-full text-left p-2 text-sm hover:bg-muted rounded-lg transition-colors"
                    >
                      {question}
                    </motion.button>
                  ))}
                </CardContent>
              </Card>
            </motion.div>

            {/* Chat Interface */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              viewport={{ once: true }}
              className="lg:col-span-3"
            >
              <Card className="glass-effect h-[600px] flex flex-col">
                <CardHeader className="flex-shrink-0">
                  <CardTitle className="flex items-center gap-2">
                    <MessageCircle className="w-5 h-5" />
                    Research Chat
                    <Badge variant="secondary" className="ml-auto">
                      {settings.ragMode ? 'RAG Mode' : 'General Mode'}
                    </Badge>
                  </CardTitle>
                </CardHeader>

                {/* Messages Area */}
                <CardContent className="flex-1 overflow-y-auto space-y-4 p-4">
                  <AnimatePresence>
                    {messages.map((message) => (
                      <motion.div
                        key={message.id}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className={`flex gap-3 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        {message.type === 'assistant' && (
                          <div className="w-8 h-8 bg-gradient-to-r from-primary to-purple-600 rounded-full flex items-center justify-center flex-shrink-0">
                            <Bot className="w-4 h-4 text-white" />
                          </div>
                        )}

                        <div className={`max-w-[80%] ${message.type === 'user' ? 'order-1' : ''}`}>
                          <div
                            className={`p-3 rounded-lg ${
                              message.type === 'user'
                                ? 'bg-gradient-to-r from-primary to-purple-600 text-white'
                                : 'bg-muted'
                            }`}
                          >
                            {message.isLoading ? (
                              <div className="flex items-center gap-2">
                                <Loader2 className="w-4 h-4 animate-spin" />
                                <span>Thinking...</span>
                              </div>
                            ) : (
                              <div className="whitespace-pre-wrap">{message.content}</div>
                            )}
                          </div>

                          {/* Sources */}
                          {message.sources && message.sources.length > 0 && (
                            <div className="mt-2 space-y-2">
                              <p className="text-xs text-muted-foreground">Sources:</p>
                              {message.sources.map((source, index) => (
                                <motion.div
                                  key={index}
                                  initial={{ opacity: 0, scale: 0.9 }}
                                  animate={{ opacity: 1, scale: 1 }}
                                  transition={{ delay: index * 0.1 }}
                                  className="p-2 bg-card border rounded-lg text-sm"
                                >
                                  <div className="flex items-center justify-between mb-1">
                                    <span className="font-medium truncate">{source.title}</span>
                                    <Badge variant="outline" className="text-xs">
                                      {source.relevance}%
                                    </Badge>
                                  </div>
                                  <p className="text-xs text-muted-foreground line-clamp-2">
                                    {source.snippet}
                                  </p>
                                </motion.div>
                              ))}
                            </div>
                          )}

                          {/* Message Actions */}
                          {message.type === 'assistant' && !message.isLoading && (
                            <div className="flex items-center gap-2 mt-2">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => copyToClipboard(message.content)}
                              >
                                <Copy className="w-3 h-3" />
                              </Button>
                              {isSpeechEnabled && (
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => speakText(message.content)}
                                >
                                  <Volume2 className="w-3 h-3" />
                                </Button>
                              )}
                              <Button variant="ghost" size="sm">
                                <ThumbsUp className="w-3 h-3" />
                              </Button>
                              <Button variant="ghost" size="sm">
                                <ThumbsDown className="w-3 h-3" />
                              </Button>
                            </div>
                          )}
                        </div>

                        {message.type === 'user' && (
                          <div className="w-8 h-8 bg-muted rounded-full flex items-center justify-center flex-shrink-0">
                            <User className="w-4 h-4" />
                          </div>
                        )}
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </CardContent>

                {/* Input Area */}
                <div className="p-4 border-t">
                  <div className="flex gap-2">
                    <Input
                      value={inputMessage}
                      onChange={(e) => setInputMessage(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="Ask me anything about research..."
                      className="flex-1"
                      disabled={isTyping}
                    />
                    <Button
                      onClick={() => handleSendMessage(inputMessage)}
                      disabled={isTyping || !inputMessage.trim()}
                      className="bg-gradient-to-r from-primary to-purple-600 hover:from-primary/90 hover:to-purple-600/90"
                    >
                      {isTyping ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Send className="w-4 h-4" />
                      )}
                    </Button>
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">
                    Press Enter to send, Shift+Enter for new line
                  </p>
                </div>
              </Card>
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  )
}
