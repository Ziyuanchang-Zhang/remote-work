'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { FaChevronLeft, FaChevronRight } from 'react-icons/fa';
import Markdown from './Markdown';
import { useLanguage } from '@/contexts/LanguageContext';
import RepoInfo from '@/types/repoinfo';
import getRepoUrl from '@/utils/getRepoUrl';
import ModelSelectionModal from './ModelSelectionModal';
// import { Strait } from 'next/font/google';
import { collectSegmentData } from 'next/dist/server/app-render/collect-segment-data';

interface Model {
  id: string;
  name: string;
}

interface Provider {
  id: string;
  name: string;
  models: Model[];
  supportsCustomModel?: boolean;
}

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface ResearchStage {
  title: string;
  content: string;
  iteration: number;
  type: 'plan' | 'update' | 'conclusion';
}
interface ThinkingStage {
  content: string;
  iteration: number;
}

interface AskProps {
  repoInfo: RepoInfo;
  provider?: string;
  model?: string;
  isCustomModel?: boolean;
  customModel?: string;
  language?: string;
  onRef?: (ref: { clearConversation: () => void }) => void;
}

interface ChatCompletionRequest {
  repo_url: string;
  type: string;
  messages: { role: 'user' | 'assistant'; content: string }[];
  provider: string;
  model: string;
  language: string;
  token?: string;
}

const Ask: React.FC<AskProps> = ({
  repoInfo,
  provider = '',
  model = '',
  isCustomModel = false,
  customModel = '',
  language = 'en',
  onRef
}) => {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');
  const [streamingResponse, setStreamingResponse] = useState(''); // 新增：流式文本内容
  const [isStreaming, setIsStreaming] = useState(false); // 新增：是否正在流式传输
  const [isMermaidGenerating, setIsMermaidGenerating] = useState(false);// 新增：是否在生成mermaid

  const [mermaidBuffer, setMermaidBuffer] = useState('')// 新增：mermaid缓存
  const [thinkingContent, setThinkingContent] = useState(''); // 新增：存储思考内容
  const [thinkingStages, setThinkingStages] = useState<ThinkingStage[]>([]); // 存储每个阶段的思考内容
  const [isThinkingCollapsed, setIsThinkingCollapsed] = useState(false); // 新增：思考内容收起状态
  const [isLoading, setIsLoading] = useState(false);
  const [deepResearch, setDeepResearch] = useState(false);

  // Model selection state
  const [selectedProvider, setSelectedProvider] = useState(provider);
  const [selectedModel, setSelectedModel] = useState(model);
  const [isCustomSelectedModel, setIsCustomSelectedModel] = useState(isCustomModel);
  const [customSelectedModel, setCustomSelectedModel] = useState(customModel);
  const [isModelSelectionModalOpen, setIsModelSelectionModalOpen] = useState(false);
  const [isComprehensiveView, setIsComprehensiveView] = useState(true);

  // Get language context for translations
  const { messages } = useLanguage();

  // Research navigation state
  const [researchStages, setResearchStages] = useState<ResearchStage[]>([]);
  const [currentStageIndex, setCurrentStageIndex] = useState(0);
  const [conversationHistory, setConversationHistory] = useState<Message[]>([]);
  const [researchIteration, setResearchIteration] = useState(0);
  const [researchComplete, setResearchComplete] = useState(false);

  // 添加控制研究流程的状态
  const [isResearching, setIsResearching] = useState(false);
  const researchAbortController = useRef<AbortController | null>(null);

  const [getResponse, setGetReponse] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);
  const responseRef = useRef<HTMLDivElement>(null);
  const providerRef = useRef(provider);
  const modelRef = useRef(model);
  // 发送与取消状态
  const [isResponding, setIsResponding] = useState<boolean>(false);
  const abortController = useRef<AbortController | null>(null);

  const [queueStatus, setQueueStatus] = useState<{
    processing: number;
    queued: number;
  } | null>(null);

  // 获取队列状态的函数
  const fetchQueueStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/vllm/status');
      if (response.ok) {
        const data = await response.json();
        setQueueStatus(prev => {
          const newStages = {
            processing: data.processing || 0,
            queued: data.queued || 0,
          };
          if (!prev || prev.processing !== newStages.processing || prev.queued !== newStages.queued) {
            return newStages;
          }
          return prev;
        });
      }
    } catch (error) {
      console.error('Failed to fetch queue status:', error);
    }
  }, []);

  //检测mermaid的状态
  const detectMermaidState = (content: string): {
    hasMermaidStart: boolean;
    hasMermaidEnd: boolean;
    isInMermaidBlock: boolean;
    lastMermaidComplete: boolean;
  } => {
    const mermaidStartRegex = /```mermaid/gi;
    const mermaidEndRegex = /```(?!\w)/g;

    const startMatches = [...content.matchAll(mermaidStartRegex)];
    const endMatches = [...content.matchAll(mermaidEndRegex)];

    const hasMermaidStart = startMatches.length > 0;
    const hasMermaidEnd = endMatches.length > 0;

    let isInMermaidBlock = false;
    let lastMermaidComplete = true;

    if (hasMermaidStart) {
      const startCount = startMatches.length;
      const endCount = endMatches.length;

      isInMermaidBlock = startCount > endCount;
      lastMermaidComplete = startCount === endCount;
    }

    return {
      hasMermaidStart,
      hasMermaidEnd,
      isInMermaidBlock,
      lastMermaidComplete
    };
  }
  // 提取不含mermaid的内容
  const extractTextWithPlaceholder = (content: string): string => {
    const mermaidState = detectMermaidState(content);

    if (!mermaidState.hasMermaidStart) {
      return content;
    }

    let result = content;
    let placeholderCount = 1;
    while (true) {
      const mermaidStartIndex = result.indexOf('```mermaid');
      if (mermaidStartIndex === -1) break;
      const afterMermaidStart = mermaidStartIndex + 10;// ```mermaid 的长度
      let nextEndIndex = result.indexOf('\n```', afterMermaidStart);

      if (nextEndIndex === -1) {
        const beforeMermaid = result.substring(0, mermaidStartIndex);
        const placeholder = `\n\n[检测到 Mermaid 图表 ${placeholderCount}，将在传输完成后渲染...]\n\n`;
        result = beforeMermaid + placeholder;
        break;
      } else {
        const beforeMermaid = result.substring(0, mermaidStartIndex);
        const afterMermaid = result.substring(nextEndIndex + 4);//\n```的长度
        const placeholder = `\n\n[检测到 Mermaid 图表 ${placeholderCount}，将在传输完成后渲染...]\n\n`;
        result = beforeMermaid + placeholder + afterMermaid;
        placeholderCount++;
      }
    }

    return result;
  };

  // 新增：检测内容是否包含 Mermaid 图表
  const containsMermaid = (content: string): boolean => {
    // 检测 Mermaid 代码块
    const mermaidRegex = /```mermaid[\s\S]*?```/i;
    return mermaidRegex.test(content);
  };

  // 新增：提取不含 Mermaid 的文本内容用于流式显示
  const extractTextWithoutMermaid = (content: string): string => {
    // 将 Mermaid 图表替换为占位符
    let textContent = content.replace(/```mermaid[\s\S]*?```/gi, '\n```\n[Mermaid Diagram - Rendering after completion...]\n```\n');
    return textContent;
  };


  // 当组件挂载时获取队列状态，并设置定期更新
  useEffect(() => {
    let interval: NodeJS.Timeout;

    // 立即获取一次状态
    fetchQueueStatus();

    const startInterval = () => {
      interval = setInterval(() => {
        if (!getResponse) {
          fetchQueueStatus();
        }
      }, 5000);
    }

    startInterval();

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [fetchQueueStatus, getResponse]);
  // Focus input on component mount
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  // Expose clearConversation method to parent component
  useEffect(() => {
    if (onRef) {
      onRef({ clearConversation });
    }
  }, [onRef]);

  // Scroll to bottom of response when it changes
  useEffect(() => {
    if (responseRef.current) {
      responseRef.current.scrollTop = responseRef.current.scrollHeight;
    }
  }, [response]);

  // 清理研究流程
  useEffect(() => {
    return () => {
      if (researchAbortController.current) {
        researchAbortController.current.abort();
      }
    };
  }, []);

  useEffect(() => {
    providerRef.current = provider;
    modelRef.current = model;
  }, [provider, model]);

  useEffect(() => {
    const fetchModel = async () => {
      try {
        setIsLoading(true);

        const response = await fetch('/api/models/config');
        if (!response.ok) {
          throw new Error(`Error fetching model configurations: ${response.status}`);
        }

        const data = await response.json();

        // Initialize provider and model with defaults from API if not already set
        if (!provider && data.defaultProvider) {
          setSelectedProvider(data.defaultProvider);

          // Find the default provider and set its default model
          const selectedProvider = data.providers.find((p: Provider) => p.id === data.defaultProvider);
          if (selectedProvider && selectedProvider.models.length > 0) {
            setSelectedModel(data.defaultModel);
          }
        }
      } catch (err) {
        console.error('Failed to fetch model configurations:', err);
      } finally {
        setIsLoading(false);
      }
    };
    if (provider == '' || model == '') {
      fetchModel()
    }
  }, [provider, model]);

  const clearConversation = () => {
    // 停止正在进行的研究
    if (researchAbortController.current) {
      researchAbortController.current.abort();
    }

    setQuestion('');
    setResponse('');
    setStreamingResponse(''); // 清空流式内容
    setIsStreaming(false); // 重置流式状态
    setIsMermaidGenerating(false)
    setMermaidBuffer('')
    setThinkingContent(''); // 清空思考内容
    setThinkingStages([]); // 清空所有阶段的思考内容
    setIsThinkingCollapsed(false); // 重置思考内容状态
    setConversationHistory([]);
    setResearchIteration(0);
    setResearchComplete(false);
    setResearchStages([]);
    setCurrentStageIndex(0);
    setIsResearching(false);
    setGetReponse(false);

    if (inputRef.current) {
      inputRef.current.focus();
    }
  };
  const downloadCurrentResponse = () => {
    // 下载当前显示的思考内容和回复内容
    let content = '';

    if (thinkingContent) {
      content += '# Thinking Process\n\n';
      content += thinkingContent;
      content += '\n\n---\n\n';
    }

    if (response) {
      content += '# Response\n\n';
      content += response;
    }

    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `current-response-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const downloadAllResponses = () => {
    // 下载所有阶段的思考内容和回复内容
    let content = '';

    // 添加标题和概述
    content += `# Deep Research Complete Report\n\n`;
    content += `Generated on: ${new Date().toLocaleString()}\n\n`;
    content += `Original Question: ${question}\n\n`;
    content += `Total Research Stages: ${researchStages.length}\n\n`;
    content += `---\n\n`;

    // 遍历所有研究阶段
    researchStages.forEach((stage, index) => {
      content += `## Stage ${index + 1}: ${stage.title}\n\n`;
      content += `**Iteration:** ${stage.iteration}\n`;
      content += `**Type:** ${stage.type}\n\n`;

      // 添加该阶段的思考内容
      const stageThinking = thinkingStages[index].content;
      if (stageThinking) {
        content += `### Thinking Process\n\n`;
        content += `\`\`\`\n${stageThinking}\n\`\`\`\n\n`;
      }

      // 添加该阶段的回复内容
      content += `### Response Content\n\n`;
      content += stage.content;
      content += `\n\n---\n\n`;
    });

    // 添加总结
    content += `## Summary\n\n`;
    content += `This report contains ${researchStages.length} research stages with their complete thinking processes and responses.\n`;
    content += `Research Status: ${researchComplete ? 'Complete' : 'In Progress'}\n`;

    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `complete-research-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Function to check if research is complete based on response content
  const checkIfResearchComplete = (content: string): boolean => {
    // Check for explicit final conclusion markers
    if (content.includes('## Final Conclusion')) {
      return true;
    }

    // Check for conclusion sections that don't indicate further research
    if ((content.includes('## Conclusion') || content.includes('## Summary')) &&
      !content.includes('I will now proceed to') &&
      !content.includes('Next Steps') &&
      !content.includes('next iteration')) {
      return true;
    }

    // Check for phrases that explicitly indicate completion
    if (content.includes('This concludes our research') ||
      content.includes('This completes our investigation') ||
      content.includes('This concludes the deep research process') ||
      content.includes('Key Findings and Implementation Details') ||
      content.includes('In conclusion,') ||
      (content.includes('Final') && content.includes('Conclusion'))) {
      return true;
    }

    // Check for topic-specific completion indicators
    if (content.includes('Dockerfile') &&
      (content.includes('This Dockerfile') || content.includes('The Dockerfile')) &&
      !content.includes('Next Steps') &&
      !content.includes('In the next iteration')) {
      return true;
    }

    return false;
  };

  const extractThinkingStage = (content: string, iteration: number): ThinkingStage | null => {
    return {
      content: content,
      iteration: iteration,
    };
  };
  // Function to extract research stages from the response
  const extractResearchStage = (content: string, iteration: number): ResearchStage | null => {
    // Check for research plan (first iteration)
    if (iteration === 1 && content.includes('## Research Plan')) {
      const planMatch = content.match(/## Research Plan([\s\S]*?)(?:## Next Steps|$)/);
      if (planMatch) {
        return {
          title: 'Research Plan',
          content: content,
          iteration: 1,
          type: 'plan'
        };
      }
    }

    // Check for research updates (iterations 1-4)
    if (iteration >= 1 && iteration <= 4) {
      const updateMatch = content.match(new RegExp(`## Research Update ${iteration}([\\s\\S]*?)(?:## Next Steps|$)`));
      if (updateMatch) {
        return {
          title: `Research Update ${iteration}`,
          content: content,
          iteration: iteration,
          type: 'update'
        };
      }
    }

    // Check for final conclusion
    if (content.includes('## Final Conclusion')) {
      const conclusionMatch = content.match(/## Final Conclusion([\s\S]*?)$/);
      if (conclusionMatch) {
        return {
          title: 'Final Conclusion',
          content: content,
          iteration: iteration,
          type: 'conclusion'
        };
      }
    }

    return null;
  };

  // Function to navigate to a specific research stage
  const navigateToStage = (index: number) => {
    if (index >= 0 && index < researchStages.length) {
      setCurrentStageIndex(index);
      setResponse(researchStages[index].content);
      setThinkingContent(thinkingStages[index].content);
    }
  };

  // Function to navigate to the next research stage
  const navigateToNextStage = () => {
    if (currentStageIndex < researchStages.length - 1) {
      navigateToStage(currentStageIndex + 1);
    }
  };

  // Function to navigate to the previous research stage
  const navigateToPreviousStage = () => {
    if (currentStageIndex > 0) {
      navigateToStage(currentStageIndex - 1);
    }
  };

  // HTTP请求函数（替换WebSocket）
  const makeHttpRequest = async (requestBody: ChatCompletionRequest, abortSignal?: AbortSignal): Promise<{
    response: string;
    thinking: string;
  }> => {
    try {
      setIsStreaming(true); // 开始流式传输
      setStreamingResponse(''); // 清空流式内容
      setIsMermaidGenerating(false);
      setMermaidBuffer('');

      const apiResponse = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
        signal: abortSignal
      });

      if (!apiResponse.ok) {
        throw new Error(`API error: ${apiResponse.status}`);
      }

      // Process the streaming response
      const reader = apiResponse.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('Failed to get response reader');
      }
      setGetReponse(true)
      // Read the stream
      let fullResponse = '';
      let fullThinkingContent = ''; // 新增：存储完整的思考内容
      let buffer = '';
      let previousMermaidState = { isInMermaidBlock: false, lastMermaidComplete: true }
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (line.startsWith('data: ')) {
            try {
              const jsonStr = trimmedLine.slice(6).trim();
              if (!jsonStr || jsonStr === '') {
                continue;
              }

              const data = JSON.parse(jsonStr);
              switch (data.type) {
                case 'heartbeat':
                  // 处理心跳数据
                  if (data.data) {
                  }
                  break;

                case 'thinking':
                  // 处理思考数据
                  if (data.data) {
                    fullThinkingContent += data.data;
                    setThinkingContent(fullThinkingContent);
                  }
                  break;
                case 'content':
                  // 只有内容类型才拼接到回答中
                  if (data.data) {
                    fullResponse += data.data;

                    const currentMermaidState = detectMermaidState(fullResponse);

                    if (!previousMermaidState.isInMermaidBlock && currentMermaidState.isInMermaidBlock) {
                      setIsMermaidGenerating(true);
                    }

                    if (previousMermaidState.isInMermaidBlock && !currentMermaidState.isInMermaidBlock && currentMermaidState.lastMermaidComplete) {
                    }

                    if (currentMermaidState.hasMermaidStart) {
                      const fullResponseWithOutMermaid = extractTextWithPlaceholder(fullResponse);
                      setStreamingResponse(fullResponseWithOutMermaid);
                    } else {
                      setStreamingResponse(fullResponse);
                      setResponse(fullResponse);
                    }

                    previousMermaidState = currentMermaidState;
                    // // 检查当前累积的内容是否包含 Mermaid 图表
                    // if (containsMermaid(fullResponse)) {
                    //   // 如果包含 Mermaid，只更新不含 Mermaid 的文本部分用于流式显示
                    //   const textOnlyContent = extractTextWithoutMermaid(fullResponse);
                    //   setStreamingResponse(textOnlyContent);
                    //   // 不更新 response，避免 Mermaid 图表频繁重新渲染
                    // } else {
                    //   // 如果不包含 Mermaid，正常流式更新
                    //   setStreamingResponse(fullResponse);
                    //   setResponse(fullResponse);
                    // }
                    // setResponse(fullResponse); 
                  }
                  break;

                default:
                  console.warn('Unknown message type:', data.type, data);
              }
            } catch (error: any) {
              if (error.name === 'parseError') {
                console.warn('Failed to parse SSE data:', trimmedLine, error);
                continue;
              }
              if (error.name === 'AbortError') {
                console.warn('Fetch Abort:', trimmedLine, error);
              }
            }
          }
        }
      }

      // 流式传输完成，设置最终完整内容
      setIsStreaming(false);
      setResponse(fullResponse);

      return {
        response: fullResponse,
        thinking: fullThinkingContent
      };
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw error;
      }
      console.error('Error during HTTP request:', error);
      throw new Error('Failed to get a response. Please try again.');
    }
  };

  // 深度研究流程 - 统一控制所有迭代
  const conductDeepResearch = async (initialQuestion: string) => {
    setIsResearching(true);
    researchAbortController.current = new AbortController();

    let currentHistory: Message[] = [{
      role: 'user',
      content: `[DEEP RESEARCH] ${initialQuestion}`
    }];

    setConversationHistory(currentHistory);

    try {
      // 最多进行5次迭代
      for (let iteration = 1; iteration <= 5; iteration++) {
        setResearchIteration(iteration);

        // 准备请求
        const requestBody: ChatCompletionRequest = {
          repo_url: getRepoUrl(repoInfo),
          type: repoInfo.type,
          messages: currentHistory.map(msg => ({
            role: msg.role as 'user' | 'assistant',
            content: msg.content
          })),
          provider: selectedProvider,
          model: isCustomSelectedModel ? customSelectedModel : selectedModel,
          language: language
        };

        if (repoInfo?.token) {
          requestBody.token = repoInfo.token;
        }
        // 清空数据
        setResponse('')
        setThinkingContent('')
        setGetReponse(false)
        // 发送请求
        const streamOut = await makeHttpRequest(requestBody, researchAbortController.current.signal);

        // 检查是否完成
        const isComplete = checkIfResearchComplete(streamOut.response);

        // 更新研究阶段
        const thinkingStage = extractThinkingStage(streamOut.thinking, iteration);
        const researchStage = extractResearchStage(streamOut.response, iteration);
        if (researchStage) {
          setResearchStages(prev => [...prev, researchStage]);
          setCurrentStageIndex(iteration - 1);
        }
        if (thinkingStage) {
          setThinkingStages(prev => [...prev, thinkingStage]);
        }

        if (iteration == 5) {
          setResearchComplete(true);
          break;
        }
        if (iteration < 5) {
          // 添加AI回复到历史记录，准备下一次迭代
          currentHistory = [
            ...currentHistory,
            { role: 'assistant', content: streamOut.response },
            { role: 'user', content: '[DEEP RESEARCH] Continue the research' }
          ];
          setConversationHistory(currentHistory);
        }
        // 在迭代之间添加短暂延迟
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('Research was cancelled');
        return;
      }
      console.error('Error during deep research:', error);
      setResponse(prev => prev + '\n\nError: Failed to continue research. Please try again.');
      setResearchComplete(true);
    } finally {
      setIsResearching(false);
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsResponding(true);
    if (!question.trim() || isLoading) return;

    handleConfirmAsk();
  };

  const handleStopGen = () => {
    // TODO 让后端的模型真正结束推理
    // 切断后端的消息，停止fetch请求
    if (abortController.current) {
      abortController.current.abort();
    }
    setIsResponding(false);
  };

  // Handle confirm and send request
  const handleConfirmAsk = async () => {
    setIsLoading(true);
    setResponse('');
    setThinkingContent(''); // 清空当前思考内容
    setThinkingStages([]); // 清空所有阶段的思考内容
    setResearchIteration(0);
    setResearchComplete(false);
    setResearchStages([]);
    setCurrentStageIndex(0);
    setGetReponse(false)

    try {
      if (deepResearch) {
        // 启动深度研究流程
        await conductDeepResearch(question);
      } else {
        // 常规单次请求
        const initialMessage: Message = {
          role: 'user',
          content: question
        };

        const newHistory: Message[] = [initialMessage];
        setConversationHistory(newHistory);

        const requestBody: ChatCompletionRequest = {
          repo_url: getRepoUrl(repoInfo),
          type: repoInfo.type,
          messages: newHistory.map(msg => ({
            role: msg.role as 'user' | 'assistant',
            content: msg.content
          })),
          provider: selectedProvider,
          model: isCustomSelectedModel ? customSelectedModel : selectedModel,
          language: language
        };

        if (repoInfo?.token) {
          requestBody.token = repoInfo.token;
        }

        try {
          // 创建 AbortController 并保存引用
          const controller = new AbortController();
          abortController.current = controller;
          await makeHttpRequest(requestBody, controller.signal);
        } catch (error) {
          console.error('Error during API call:', error);
          setResponse('Error: Failed to get a response. Please try again.');
        } finally {
          setIsLoading(false);
        }
      }
    } catch (error) {
      console.error('Error during API call:', error);
      setResponse('Error: Failed to get a response. Please try again.');
      setIsLoading(false);
    }
  };

  const [buttonWidth, setButtonWidth] = useState(0);
  const buttonRef = useRef<HTMLButtonElement>(null);

  // Measure button width and update state
  useEffect(() => {
    if (buttonRef.current) {
      const width = buttonRef.current.offsetWidth;
      setButtonWidth(width);
    }
  }, [messages.ask?.askButton, isLoading]);

  return (
    <div className="flex flex-col min-h-0 h-full">
      <div className="flex-1 flex flex-col p-4 overflow-hidden">
        <div className="flex items-center justify-between mb-4 flex-shrink-0">
          {/* 队列状态显示 - 左侧 */}
          {queueStatus && (
            <div className="flex-1 mr-4">
              <div className="flex items-center justify-start text-xs text-[var(--muted)] bg-[var(--background)]/30 rounded-md px-3 py-2 border border-[var(--border-color)] w-fit">
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-[var(--accent-primary)] rounded-full"></div>
                    <span>
                      {language === 'zh' ? '正在处理' : 'Processing'}: {queueStatus.processing}
                    </span>
                  </div>
                  <div className="w-px h-3 bg-[var(--border-color)]"></div>
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-[var(--muted)] rounded-full"></div>
                    <span>
                      {language === 'zh' ? '排队等待' : 'Queued'}: {queueStatus.queued}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Model selection button - 右侧 */}
          <div className="flex-shrink-0">
            <button
              type="button"
              onClick={() => setIsModelSelectionModalOpen(true)}
              className="text-xs px-2.5 py-1 rounded border border-[var(--border-color)]/40 bg-[var(--background)]/10 text-[var(--foreground)]/80 hover:bg-[var(--background)]/30 hover:text-[var(--foreground)] transition-colors flex items-center gap-1.5"
            >
              <span>{isCustomSelectedModel ? customSelectedModel : selectedModel}</span>
              <svg className="h-3.5 w-3.5 text-[var(--accent-primary)]/70" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
            </button>
          </div>
        </div>

        {/* Question input - 固定高度 */}
        <form className="flex-shrink-0">
          {/* <form onSubmit={handleSubmit} className="flex-shrink-0"></form> */}
          <div className="relative">
            <input
              ref={inputRef}
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={messages.ask?.placeholder || 'What would you like to know about this codebase?'}
              className="block w-full rounded-md border border-[var(--border-color)] bg-[var(--input-bg)] text-[var(--foreground)] px-5 py-3.5 text-base shadow-sm focus:border-[var(--accent-primary)] focus:ring-2 focus:ring-[var(--accent-primary)]/30 focus:outline-none transition-all"
              style={{ paddingRight: `${buttonWidth + 24}px` }}
              disabled={isLoading}
            />
            {/* 点击提问以后显示停止回答按钮 */}
            <div className="button-container">
              {!isResponding && (
                <button
                  ref={buttonRef}
                  // type="submit"
                  type="button"
                  onClick={handleSubmit}
                  disabled={isLoading || !question.trim()}
                  className={`absolute right-3 top-1/2 transform -translate-y-1/2 px-4 py-2 rounded-md font-medium text-sm ${isLoading || !question.trim()
                    ? 'bg-[var(--button-disabled-bg)] text-[var(--button-disabled-text)] cursor-not-allowed'
                    : 'bg-[var(--accent-primary)] text-white hover:bg-[var(--accent-primary)]/90 shadow-sm'
                    } transition-all duration-200 flex items-center gap-1.5`}
                >
                  {isLoading ? (
                    <div className="w-4 h-4 rounded-full border-2 border-t-transparent border-white animate-spin" />
                  ) : (
                    <>
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                      </svg>
                      <span>{messages.ask?.askButton || 'Ask'}</span>
                    </>
                  )}
                </button>
              )}
              {/* 已经发出去展示停止按钮 */}
              {isResponding && (
                <button onClick={handleStopGen} className={`absolute right-3 top-1/2 transform -translate-y-1/2 px-4 py-2 rounded-md font-medium text-sm 
                  bg-[var(--accent-primary)] text-white hover:bg-[var(--accent-primary)]/90 shadow-sm transition-all duration-200 flex items-center gap-1.5`}>
                  <>
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                    </svg>
                    <span>{messages.ask?.stopGenerate || 'Stop'}</span>
                  </>
                </button>
              )}
            </div>
          </div>

          {/* Deep Research toggle */}
          <div className="flex items-center mt-2 justify-between">
            <div className="group relative">
              <label className="flex items-center cursor-pointer">
                <span className="text-xs text-gray-600 dark:text-gray-400 mr-2">Deep Research</span>
                <div className="relative">
                  <input
                    type="checkbox"
                    checked={deepResearch}
                    onChange={() => setDeepResearch(!deepResearch)}
                    className="sr-only"
                    disabled={isLoading || isResearching}
                  />
                  <div className={`w-10 h-5 rounded-full transition-colors ${deepResearch ? 'bg-purple-600' : 'bg-gray-300 dark:bg-gray-600'} ${isLoading ? 'opacity-50' : ''}`}></div>
                  <div className={`absolute left-0.5 top-0.5 w-4 h-4 rounded-full bg-white transition-transform transform ${deepResearch ? 'translate-x-5' : ''}`}></div>
                </div>
              </label>
              <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block bg-gray-800 text-white text-xs rounded p-2 w-72 z-10">
                <div className="relative">
                  <div className="absolute -bottom-2 left-4 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-800"></div>
                  <p className="mb-1">Deep Research conducts a single controlled multi-iteration investigation:</p>
                  <ul className="list-disc pl-4 text-xs">
                    <li><strong>Iteration 1:</strong> Creates research plan and initial findings</li>
                    <li><strong>Iteration 2-4:</strong> Progressively explores deeper aspects</li>
                    <li><strong>Final:</strong> Comprehensive conclusion with all findings</li>
                  </ul>
                  <p className="mt-1 text-xs italic">Optimized to use fewer API calls while maintaining depth</p>
                </div>
              </div>
            </div>
            {deepResearch && (
              <div className="text-xs text-purple-600 dark:text-purple-400">
                Controlled multi-iteration research
                {isResearching && ` (iteration ${researchIteration})`}
                {researchComplete && ` (complete)`}
              </div>
            )}
          </div>
        </form>
        {/* Loading indicator */}
        {isLoading && (
          <div className="flex-shrink-0 p-4 border-t border-gray-200 dark:border-gray-700">
            <div className="flex items-center space-x-2">
              <div className="animate-pulse flex space-x-1">
                <div className="h-2 w-2 bg-purple-600 rounded-full"></div>
                <div className="h-2 w-2 bg-purple-600 rounded-full"></div>
                <div className="h-2 w-2 bg-purple-600 rounded-full"></div>
              </div>
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {deepResearch
                  ? (researchIteration === 0
                    ? "Planning research approach..."
                    : `Research iteration ${researchIteration} in progress...`)
                  : "Thinking..."}
              </span>
            </div>
            {deepResearch && !thinkingContent && (
              <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 pl-5">
                <div className="flex flex-col space-y-1">
                  {researchIteration === 0 && (
                    <>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-blue-500 rounded-full mr-2"></div>
                        <span>Creating research plan...</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                        <span>Identifying key areas to investigate...</span>
                      </div>
                    </>
                  )}
                  {researchIteration === 1 && (
                    <>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-blue-500 rounded-full mr-2"></div>
                        <span>Exploring first research area in depth...</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                        <span>Analyzing code patterns and structures...</span>
                      </div>
                    </>
                  )}
                  {researchIteration === 2 && (
                    <>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-amber-500 rounded-full mr-2"></div>
                        <span>Investigating remaining questions...</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-purple-500 rounded-full mr-2"></div>
                        <span>Connecting findings from previous iterations...</span>
                      </div>
                    </>
                  )}
                  {researchIteration === 3 && (
                    <>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-indigo-500 rounded-full mr-2"></div>
                        <span>Exploring deeper connections...</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-blue-500 rounded-full mr-2"></div>
                        <span>Analyzing complex patterns...</span>
                      </div>
                    </>
                  )}
                  {researchIteration === 4 && (
                    <>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-teal-500 rounded-full mr-2"></div>
                        <span>Refining research conclusions...</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-cyan-500 rounded-full mr-2"></div>
                        <span>Addressing remaining edge cases...</span>
                      </div>
                    </>
                  )}
                  {researchIteration >= 5 && (
                    <>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-purple-500 rounded-full mr-2"></div>
                        <span>Finalizing comprehensive answer...</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                        <span>Synthesizing all research findings...</span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
        {/* Response area - 可滚动区域 */}
        {(response || thinkingContent) && (
          <div className="flex-1 flex flex-col border-t border-gray-200 dark:border-gray-700 mt-4 min-h-0">
            {/* Thinking content section */}
            {thinkingContent && (
              <div className="flex-shrink-0 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800/50">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-amber-500 rounded-full animate-pulse"></div>
                    <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
                      {language === 'zh' ? '思考过程' : 'Thinking Process'}
                    </span>
                  </div>
                  <button
                    onClick={() => setIsThinkingCollapsed(!isThinkingCollapsed)}
                    className="flex items-center space-x-1 text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
                  >
                    <span>{isThinkingCollapsed ? (language === 'zh' ? '展开' : 'Expand') : (language === 'zh' ? '收起' : 'Collapse')}</span>
                    <svg
                      className={`w-3 h-3 transform transition-transform ${isThinkingCollapsed ? 'rotate-180' : ''}`}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>
                </div>
                {!isThinkingCollapsed && (
                  <div className="p-3 max-h-[200px] overflow-y-auto bg-gray-50/50 dark:bg-gray-800/20">
                    <div className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed whitespace-pre-wrap font-mono">
                      {thinkingContent}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Main response content - 占用剩余空间 */}
            {(response || streamingResponse) && (
              <div
                ref={responseRef}
                className="flex-1 p-4 overflow-y-auto min-h-0"
              >
                {isStreaming ? (
                  <div className="space-y-4">
                    {/* 显示流式文本内容（不含 Mermaid） */}
                    <Markdown content={streamingResponse} />
                    {/* 如果检测到 Mermaid 内容，显示提示 */}
                    {containsMermaid(response || '') && (
                      <div className="flex items-center space-x-2 text-sm text-gray-500 dark:text-gray-400 border-l-4 border-blue-400 pl-3 py-2 bg-blue-50 dark:bg-blue-900/20">
                        <div className="animate-spin w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full"></div>
                        <span>
                          {language === 'zh'
                            ? '检测到 Mermaid 图表，将在传输完成后渲染...'
                            : 'Mermaid diagram detected, will render after streaming completes...'}
                        </span>
                      </div>
                    )}
                  </div>
                ) : (
                  <Markdown content={response} />
                )}
              </div>
            )}

            {/* Research navigation and clear button */}
            <div className="flex-shrink-0 p-2 flex justify-between items-center border-t border-gray-200 dark:border-gray-700">
              {/* Research navigation */}
              {deepResearch && researchStages.length > 0 && (
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => navigateToPreviousStage()}
                    disabled={currentStageIndex === 0 || isResearching}
                    className={`p-1 rounded-md 
                      ${currentStageIndex === 0 || isResearching
                        ? 'text-gray-400 dark:text-gray-600 cursor-not-allowed'
                        : 'text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'}`}
                    aria-label="Previous stage"
                    title={isResearching ? "Navigation disabled during research" : "Previous stage"}
                  >
                    <FaChevronLeft size={12} />
                  </button>

                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    {currentStageIndex + 1} / {Math.max(researchStages.length, researchIteration)}
                    {isResearching && (
                      <span className='ml-1 text-amber-500'>
                        (Research in progress...)
                      </span>
                    )}
                  </div>

                  <button
                    onClick={() => navigateToNextStage()}
                    disabled={currentStageIndex >= researchStages.length - 1 || isResearching}
                    className={`p-1 rounded-md ${currentStageIndex >= researchStages.length - 1 || isResearching
                      ? 'text-gray-400 dark:text-gray-600 cursor-not-allowed'
                      : 'text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'}`}
                    aria-label="Next stage"
                    title={isResearching ? "Navigation disabled during research" : "Previous stage"}
                  >
                    <FaChevronRight size={12} />
                  </button>

                  <div className="text-xs text-gray-600 dark:text-gray-400 ml-2">
                    {researchStages[currentStageIndex]?.title || `Stage ${currentStageIndex}`}
                  </div>
                </div>
              )}
              <div className="flex items-center space-x-2">
                {/* Download Current button */}
                {!isLoading && !isResearching && (
                  <button
                    onClick={downloadCurrentResponse}
                    className="text-xs text-gray-500 dark:text-gray-400 hover:text-green-600 dark:hover:text-green-400 px-2 py-1 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 flex items-center gap-1"
                    title={thinkingContent ? "Download current thinking process and response" : "Download current response as markdown file"}
                  >
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    Download Current
                  </button>
                )}
                {/* Download All button - only show when deep research is complete */}
                {deepResearch && researchComplete && researchStages.length > 1 && (
                  <button
                    onClick={downloadAllResponses}
                    className="text-xs text-gray-500 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 px-2 py-1 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 flex items-center gap-1"
                    title="Download complete research report with all stages"
                  >
                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                    </svg>
                    Download All
                  </button>
                )}

                {/* Clear button */}
                <button
                  id="ask-clear-conversation"
                  onClick={clearConversation}
                  className="text-xs text-gray-500 dark:text-gray-400 hover:text-purple-600 dark:hover:text-purple-400 px-2 py-1 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700"
                >
                  Clear conversation
                </button>
              </div>
            </div>
          </div>
        )}


      </div>

      {/* Model Selection Modal */}
      <ModelSelectionModal
        isOpen={isModelSelectionModalOpen}
        onClose={() => setIsModelSelectionModalOpen(false)}
        provider={selectedProvider}
        setProvider={setSelectedProvider}
        model={selectedModel}
        setModel={setSelectedModel}
        isCustomModel={isCustomSelectedModel}
        setIsCustomModel={setIsCustomSelectedModel}
        customModel={customSelectedModel}
        setCustomModel={setCustomSelectedModel}
        isComprehensiveView={isComprehensiveView}
        setIsComprehensiveView={setIsComprehensiveView}
        showFileFilters={false}
        onApply={() => {
          console.log('Model selection applied:', selectedProvider, selectedModel);
        }}
        showWikiType={false}
        authRequired={false}
        isAuthLoading={false}
      />
    </div>
  );
};

export default Ask;
