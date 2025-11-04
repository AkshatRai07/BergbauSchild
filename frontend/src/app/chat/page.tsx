'use client';

import { useState, useRef, useEffect, FormEvent } from 'react';
import { ChatMessage } from '@/lib/types';

// Simple Markdown-like renderer for chat bubbles
function ChatBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';
  
  return (
    <div
      className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      <div
        className={`max-w-xl rounded-lg px-4 py-3 shadow-md
          ${
            isUser
              ? 'bg-cyan-700 text-white'
              : 'bg-gray-700 text-gray-100'
          }
        `}
      >
        <p className="whitespace-pre-wrap">{message.content}</p>
      </div>
    </div>
  );
}


export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
    };
    
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Add a placeholder for the assistant's response
    const assistantMessageId = (Date.now() + 1).toString();
    setMessages((prev) => [
      ...prev,
      { id: assistantMessageId, role: 'assistant', content: '▍' },
    ]);
    
    try {
      const response = await fetch('http://localhost:8000/chat_stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: input }),
      });

      if (!response.body) {
        throw new Error('No response body');
      }

      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
      let assistantResponse = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        assistantResponse += value;
        
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId
              ? { ...msg, content: assistantResponse + '▍' }
              : msg
          )
        );
      }
      
      // Final update to remove the cursor
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? { ...msg, content: assistantResponse }
            : msg
        )
      );

    } catch (error) {
      console.error('Chat stream error:', error);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? { ...msg, content: 'Error: Could not get a response from the agent.' }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-[calc(100vh-100px)] flex-col">
      <h1 className="mb-4 text-3xl font-bold text-white">Digital Mine Safety Officer</h1>
      
      {/* Chat Messages Area */}
      <div className="grow space-y-4 overflow-y-auto rounded-lg border border-gray-700 bg-gray-800 p-4">
        {messages.length === 0 && (
          <div className="flex h-full items-center justify-center">
            <p className="text-gray-400">Ask me anything about Indian mine safety...</p>
          </div>
        )}
        {messages.map((msg) => (
          <ChatBubble key={msg.id} message={msg} />
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="mt-4 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about an accident, get statistics, or classify a new report..."
          className="grow rounded-lg border border-gray-600 bg-gray-700 px-4 py-2 text-white placeholder-gray-400 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
          disabled={isLoading}
        />
        <button
          type="submit"
          className="rounded-lg bg-cyan-600 px-5 py-2 font-semibold text-white transition-colors hover:bg-cyan-500 disabled:cursor-not-allowed disabled:bg-gray-600"
          disabled={isLoading}
        >
          {isLoading ? 'Thinking...' : 'Send'}
        </button>
      </form>
    </div>
  );
}