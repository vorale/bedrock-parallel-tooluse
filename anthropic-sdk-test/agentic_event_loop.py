"""
Agentic Event Loop with Reusable ThreadPoolExecutor for Parallel Tool Execution
"""
import os
import json
import asyncio
import concurrent.futures
import logging
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from dotenv import load_dotenv
from anthropic import AnthropicBedrock
from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

# Load environment variables
load_dotenv()

# Configure logging
def setup_agentic_logging():
    """Setup comprehensive logging for agentic event loop"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/agentic_event_loop_{timestamp}.log"
    
    # Configure logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    # Create logger for this module
    logger = logging.getLogger('AgenticEventLoop')
    logger.info(f"🚀 Agentic Event Loop logging initialized. Log file: {log_filename}")
    return logger

# Initialize logging
agentic_logger = setup_agentic_logging()

class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING_TOOLS = "executing_tools"
    PLANNING = "planning"
    RESPONDING = "responding"
    ERROR = "error"
    COMPLETED = "completed"

@dataclass
class AgentEvent:
    """Event in the agent execution loop"""
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    event_id: str
    parent_event_id: Optional[str] = None

@dataclass
class AgentContext:
    """Context maintained throughout agent execution"""
    conversation_id: str
    user_message: str
    current_state: AgentState
    events: List[AgentEvent]
    tool_results: List[Dict[str, Any]]
    messages: List[Dict[str, Any]]
    iteration_count: int = 0
    max_iterations: int = 10

class AgenticEventLoop:
    """
    Agentic Event Loop that reuses ThreadPoolExecutor for parallel tool execution
    """
    
    def __init__(self, aws_region: str = None, max_workers: int = None):
        """
        Initialize the agentic event loop
        
        Args:
            aws_region: AWS region for Bedrock
            max_workers: Maximum workers for the thread pool (None = auto-size)
        """
        self.aws_region = aws_region or os.getenv('AWS_REGION', 'us-east-1')
        self.logger = logging.getLogger('AgenticEventLoop')
        
        # Initialize AnthropicBedrock client
        self.client = AnthropicBedrock(
            aws_region=self.aws_region,
            timeout=300.0,
        )
        
        self.model = "us.anthropic.claude-sonnet-4-20250514-v1:0"
        
        # Reusable ThreadPoolExecutor for the entire agent lifecycle
        self.max_workers = max_workers
        self.executor = None
        
        # Event handlers for different agent states
        self.state_handlers = {
            AgentState.THINKING: self._handle_thinking,
            AgentState.EXECUTING_TOOLS: self._handle_tool_execution,
            AgentState.PLANNING: self._handle_planning,
            AgentState.RESPONDING: self._handle_responding,
        }
        
        self.logger.info(f"🤖 AgenticEventLoop initialized")
        self.logger.info(f"📍 AWS Region: {self.aws_region}")
        self.logger.info(f"🧠 Model: {self.model}")
        self.logger.info(f"🧵 Max Workers: {max_workers or 'auto-size'}")
        self.logger.info(f"🔧 Available tools: {list(TOOL_FUNCTIONS.keys())}")
        self.logger.info(f"📋 State handlers: {list(self.state_handlers.keys())}")
    
    def __enter__(self):
        """Context manager entry - initialize thread pool"""
        # Create thread pool that will be reused throughout agent execution
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="AgentToolExecutor"
        )
        
        actual_max_workers = self.executor._max_workers
        self.logger.info(f"🧵 ThreadPoolExecutor initialized")
        self.logger.info(f"   • Max Workers: {actual_max_workers}")
        self.logger.info(f"   • Thread Name Prefix: AgentToolExecutor")
        self.logger.info(f"   • Pool Status: Active and ready for reuse")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup thread pool"""
        if self.executor:
            self.logger.info("🧵 Shutting down ThreadPoolExecutor...")
            
            # Log any pending tasks
            try:
                # Get executor statistics if available
                self.logger.info("   • Waiting for all threads to complete...")
            except:
                pass
            
            self.executor.shutdown(wait=True)
            
            self.logger.info("✅ ThreadPoolExecutor shutdown completed")
            self.logger.info("   • All threads terminated gracefully")
            self.logger.info("   • Resources cleaned up successfully")
            
            if exc_type:
                self.logger.error(f"💥 Context manager exiting due to exception: {exc_type.__name__}: {exc_val}")
        else:
            self.logger.warning("⚠️ ThreadPoolExecutor was not initialized")
    
    async def run_agent(self, user_message: str) -> Dict[str, Any]:
        """
        Run the complete agentic event loop
        
        Args:
            user_message: Initial user message to process
            
        Returns:
            Complete agent execution result
        """
        agent_start_time = datetime.now()
        
        # Initialize agent context
        context = AgentContext(
            conversation_id=f"agent_{agent_start_time.strftime('%Y%m%d_%H%M%S_%f')}",
            user_message=user_message,
            current_state=AgentState.IDLE,
            events=[],
            tool_results=[],
            messages=[{"role": "user", "content": user_message}]
        )
        
        self.logger.info("=" * 80)
        self.logger.info(f"🚀 STARTING AGENTIC EVENT LOOP")
        self.logger.info("=" * 80)
        self.logger.info(f"🆔 Conversation ID: {context.conversation_id}")
        self.logger.info(f"👤 User Message: {user_message}")
        self.logger.info(f"🔄 Initial State: {context.current_state.value}")
        self.logger.info(f"📊 Max Iterations: {context.max_iterations}")
        self.logger.info(f"🧵 Thread Pool Status: {'Active' if self.executor else 'Not Initialized'}")
        
        try:
            # Main event loop
            while (context.current_state != AgentState.COMPLETED and 
                   context.current_state != AgentState.ERROR and
                   context.iteration_count < context.max_iterations):
                
                iteration_start_time = datetime.now()
                context.iteration_count += 1
                
                self.logger.info("-" * 60)
                self.logger.info(f"🔄 AGENT ITERATION {context.iteration_count}/{context.max_iterations}")
                self.logger.info(f"📍 Current State: {context.current_state.value}")
                self.logger.info(f"⏰ Iteration Start: {iteration_start_time.strftime('%H:%M:%S.%f')[:-3]}")
                
                # Determine next state based on current context
                next_state = await self._determine_next_state(context)
                self.logger.info(f"🎯 Next State Determined: {next_state.value}")
                
                if next_state != context.current_state:
                    await self._transition_state(context, next_state)
                
                # Execute state handler
                if context.current_state in self.state_handlers:
                    self.logger.info(f"🎬 Executing state handler: {context.current_state.value}")
                    handler_start = datetime.now()
                    
                    await self.state_handlers[context.current_state](context)
                    
                    handler_time = (datetime.now() - handler_start).total_seconds()
                    self.logger.info(f"✅ State handler completed in {handler_time:.2f}s")
                else:
                    self.logger.warning(f"⚠️ No handler for state: {context.current_state}")
                    self.logger.error(f"❌ Available handlers: {list(self.state_handlers.keys())}")
                    break
                
                iteration_time = (datetime.now() - iteration_start_time).total_seconds()
                self.logger.info(f"⏱️ Iteration {context.iteration_count} completed in {iteration_time:.2f}s")
            
            # Log completion reason
            if context.current_state == AgentState.COMPLETED:
                self.logger.info("🎉 Agent completed successfully")
            elif context.current_state == AgentState.ERROR:
                self.logger.error("💥 Agent terminated due to error")
            elif context.iteration_count >= context.max_iterations:
                self.logger.warning(f"⏰ Agent terminated due to max iterations ({context.max_iterations})")
            
            # Generate final result
            final_result = self._generate_final_result(context)
            
            total_time = (datetime.now() - agent_start_time).total_seconds()
            self.logger.info("=" * 80)
            self.logger.info(f"🏁 AGENT EXECUTION COMPLETED")
            self.logger.info(f"⏱️ Total Execution Time: {total_time:.2f}s")
            self.logger.info(f"🔄 Total Iterations: {context.iteration_count}")
            self.logger.info(f"📊 Total Events: {len(context.events)}")
            self.logger.info(f"✅ Success: {final_result.get('success', False)}")
            self.logger.info("=" * 80)
            
            return final_result
            
        except Exception as e:
            total_time = (datetime.now() - agent_start_time).total_seconds()
            
            self.logger.error("=" * 80)
            self.logger.error(f"💥 AGENT EXECUTION FAILED")
            self.logger.error(f"⏱️ Execution Time Before Failure: {total_time:.2f}s")
            self.logger.error(f"🔄 Iterations Completed: {context.iteration_count}")
            self.logger.error(f"📍 Final State: {context.current_state.value}")
            self.logger.error(f"❌ Error: {e}")
            self.logger.error("🔍 Full Exception:", exc_info=True)
            self.logger.error("=" * 80)
            
            context.current_state = AgentState.ERROR
            return self._generate_error_result(context, str(e))
    
    async def _determine_next_state(self, context: AgentContext) -> AgentState:
        """Determine the next state based on current context"""
        
        if context.current_state == AgentState.IDLE:
            return AgentState.THINKING
        
        elif context.current_state == AgentState.THINKING:
            # Check if Claude requested tools in the last response
            last_event = context.events[-1] if context.events else None
            if last_event and last_event.event_type == "claude_response":
                tool_calls = last_event.data.get("tool_calls", [])
                if tool_calls:
                    return AgentState.EXECUTING_TOOLS
                else:
                    return AgentState.RESPONDING
            return AgentState.THINKING
        
        elif context.current_state == AgentState.EXECUTING_TOOLS:
            return AgentState.PLANNING
        
        elif context.current_state == AgentState.PLANNING:
            # Decide whether to continue with more tools or respond
            if self._should_continue_execution(context):
                return AgentState.THINKING
            else:
                return AgentState.RESPONDING
        
        elif context.current_state == AgentState.RESPONDING:
            return AgentState.COMPLETED
        
        return AgentState.COMPLETED
    
    async def _transition_state(self, context: AgentContext, new_state: AgentState):
        """Transition to a new state and log the event"""
        transition_time = datetime.now()
        old_state = context.current_state
        context.current_state = new_state
        
        event = AgentEvent(
            event_type="state_transition",
            timestamp=transition_time,
            data={
                "from_state": old_state.value,
                "to_state": new_state.value,
                "iteration": context.iteration_count,
                "transition_reason": self._get_transition_reason(old_state, new_state, context)
            },
            event_id=f"transition_{len(context.events)}"
        )
        
        context.events.append(event)
        
        self.logger.info(f"🔄 STATE TRANSITION")
        self.logger.info(f"   From: {old_state.value}")
        self.logger.info(f"   To: {new_state.value}")
        self.logger.info(f"   Iteration: {context.iteration_count}")
        self.logger.info(f"   Reason: {event.data['transition_reason']}")
        self.logger.info(f"   Timestamp: {transition_time.strftime('%H:%M:%S.%f')[:-3]}")
    
    def _get_transition_reason(self, old_state: AgentState, new_state: AgentState, context: AgentContext) -> str:
        """Get human-readable reason for state transition"""
        if old_state == AgentState.IDLE and new_state == AgentState.THINKING:
            return "Initial agent startup"
        elif old_state == AgentState.THINKING and new_state == AgentState.EXECUTING_TOOLS:
            return "Claude requested tool execution"
        elif old_state == AgentState.THINKING and new_state == AgentState.RESPONDING:
            return "Claude provided direct response without tools"
        elif old_state == AgentState.EXECUTING_TOOLS and new_state == AgentState.PLANNING:
            return "Tool execution completed, planning next actions"
        elif old_state == AgentState.PLANNING and new_state == AgentState.THINKING:
            return "Planning decided to continue with more tool execution"
        elif old_state == AgentState.PLANNING and new_state == AgentState.RESPONDING:
            return "Planning decided to generate final response"
        elif old_state == AgentState.RESPONDING and new_state == AgentState.COMPLETED:
            return "Final response generated successfully"
        else:
            return f"Transition from {old_state.value} to {new_state.value}"
    
    async def _handle_thinking(self, context: AgentContext):
        """Handle the thinking state - send request to Claude"""
        thinking_start_time = datetime.now()
        
        self.logger.info("🧠 AGENT THINKING STATE")
        self.logger.info(f"📤 Sending request to Claude model: {self.model}")
        self.logger.info(f"📋 Available tools: {[tool['name'] for tool in TOOL_SCHEMAS]}")
        self.logger.info(f"💬 Messages in conversation: {len(context.messages)}")
        
        # Log the current messages being sent
        self.logger.info("📨 CURRENT CONVERSATION CONTEXT:")
        for i, msg in enumerate(context.messages, 1):
            self.logger.info(f"   Message {i}: Role={msg['role']}")
            if isinstance(msg['content'], str):
                content_preview = msg['content'][:200] + '...' if len(msg['content']) > 200 else msg['content']
                self.logger.info(f"      Content: {content_preview}")
            elif isinstance(msg['content'], list):
                self.logger.info(f"      Content: {len(msg['content'])} items (tool results)")
        
        try:
            claude_request_start = datetime.now()
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                tools=TOOL_SCHEMAS,
                messages=context.messages
            )
            
            claude_request_time = (datetime.now() - claude_request_start).total_seconds()
            
            self.logger.info(f"📥 Received response from Claude in {claude_request_time:.2f}s")
            self.logger.info(f"🔍 Response content blocks: {len(response.content)}")
            
            # Extract tool calls from Anthropic response
            tool_calls = []
            response_text_blocks = []
            
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_calls.append({
                        "id": content_block.id,
                        "name": content_block.name,
                        "input": content_block.input
                    })
                elif content_block.type == "text":
                    response_text_blocks.append(content_block.text)
            
            # Log Claude's response details
            self.logger.info("📄 CLAUDE'S RESPONSE ANALYSIS:")
            self.logger.info(f"   Text blocks: {len(response_text_blocks)}")
            self.logger.info(f"   Tool calls requested: {len(tool_calls)}")
            
            if response_text_blocks:
                combined_text = " ".join(response_text_blocks)
                text_preview = combined_text[:300] + '...' if len(combined_text) > 300 else combined_text
                self.logger.info(f"   Text content: {text_preview}")
            
            if tool_calls:
                self.logger.info("🔧 TOOL CALLS REQUESTED:")
                for i, tool_call in enumerate(tool_calls, 1):
                    self.logger.info(f"   Tool {i}: {tool_call['name']}")
                    self.logger.info(f"      ID: {tool_call['id']}")
                    self.logger.info(f"      Input: {json.dumps(tool_call['input'], indent=6)}")
            
            # Log the complete Claude response
            self.logger.info("📋 COMPLETE CLAUDE RESPONSE:")
            self.logger.info(json.dumps(response_text_blocks + [{'tool_calls': tool_calls}], indent=2, ensure_ascii=False))
            
            # Log the thinking event
            event = AgentEvent(
                event_type="claude_response",
                timestamp=datetime.now(),
                data={
                    "response_text_blocks": response_text_blocks,
                    "tool_calls": tool_calls,
                    "tools_requested": len(tool_calls),
                    "text_blocks": len(response_text_blocks),
                    "request_time_seconds": claude_request_time,
                    "model_used": self.model
                },
                event_id=f"thinking_{len(context.events)}"
            )
            
            context.events.append(event)
            context.messages.append({"role": "assistant", "content": response.content})
            
            thinking_total_time = (datetime.now() - thinking_start_time).total_seconds()
            
            self.logger.info(f"🧠 THINKING COMPLETED")
            self.logger.info(f"   Claude request time: {claude_request_time:.2f}s")
            self.logger.info(f"   Total thinking time: {thinking_total_time:.2f}s")
            self.logger.info(f"   Tools requested: {len(tool_calls)}")
            self.logger.info(f"   Next state: {'EXECUTING_TOOLS' if tool_calls else 'RESPONDING'}")
            
        except Exception as e:
            thinking_error_time = (datetime.now() - thinking_start_time).total_seconds()
            
            self.logger.error(f"❌ THINKING STATE FAILED")
            self.logger.error(f"   Error after: {thinking_error_time:.2f}s")
            self.logger.error(f"   Error type: {type(e).__name__}")
            self.logger.error(f"   Error message: {str(e)}")
            self.logger.error(f"   Model: {self.model}")
            self.logger.error(f"   Messages count: {len(context.messages)}")
            
            context.current_state = AgentState.ERROR
    
    async def _handle_tool_execution(self, context: AgentContext):
        """Handle tool execution using the reusable ThreadPoolExecutor"""
        execution_start_time = datetime.now()
        
        self.logger.info("🔧 AGENT TOOL EXECUTION STATE")
        self.logger.info(f"🧵 Using reusable ThreadPoolExecutor (max_workers: {self.executor._max_workers})")
        
        # Get tool calls from the last thinking event
        last_event = next((e for e in reversed(context.events) if e.event_type == "claude_response"), None)
        if not last_event:
            self.logger.error("❌ TOOL EXECUTION ERROR: No tool calls found")
            self.logger.error("   No claude_response event found in context")
            self.logger.error(f"   Available events: {[e.event_type for e in context.events]}")
            context.current_state = AgentState.ERROR
            return
        
        tool_calls = last_event.data.get("tool_calls", [])
        if not tool_calls:
            self.logger.info("ℹ️ No tools to execute - continuing to next state")
            return
        
        self.logger.info(f"⚡ STARTING PARALLEL TOOL EXECUTION")
        self.logger.info(f"   Tools to execute: {len(tool_calls)}")
        self.logger.info(f"   Thread pool status: Active")
        self.logger.info(f"   Execution method: Async with reusable ThreadPoolExecutor")
        
        # Log each tool that will be executed
        self.logger.info("📋 TOOLS TO EXECUTE:")
        for i, tool_call in enumerate(tool_calls, 1):
            self.logger.info(f"   Tool {i}/{len(tool_calls)}: {tool_call['name']}")
            self.logger.info(f"      ID: {tool_call['id']}")
            self.logger.info(f"      Input: {json.dumps(tool_call['input'], indent=6)}")
        
        # Execute tools in parallel using the reusable thread pool
        parallel_start_time = datetime.now()
        tool_results = await self._execute_tools_parallel_async(tool_calls)
        parallel_execution_time = (datetime.now() - parallel_start_time).total_seconds()
        
        # Analyze results
        successful_results = [r for r in tool_results if r.get('success')]
        failed_results = [r for r in tool_results if not r.get('success')]
        
        self.logger.info("📊 PARALLEL EXECUTION RESULTS:")
        self.logger.info(f"   Total execution time: {parallel_execution_time:.2f}s")
        self.logger.info(f"   Successful tools: {len(successful_results)}")
        self.logger.info(f"   Failed tools: {len(failed_results)}")
        
        # Log detailed results for each tool
        self.logger.info("📋 DETAILED TOOL RESULTS:")
        for i, (tool_call, tool_result) in enumerate(zip(tool_calls, tool_results), 1):
            status = "✅ SUCCESS" if tool_result.get('success') else "❌ FAILED"
            exec_time = tool_result.get('execution_time_seconds', 0)
            
            self.logger.info(f"   Tool {i}: {tool_call['name']} - {status} ({exec_time:.2f}s)")
            
            if tool_result.get('success'):
                result_data = tool_result.get('result', {})
                self.logger.info(f"      📤 Result: {json.dumps(result_data, indent=6)}")
            else:
                error_msg = tool_result.get('error', 'Unknown error')
                self.logger.info(f"      ❌ Error: {error_msg}")
        
        # Log the tool execution event
        event = AgentEvent(
            event_type="parallel_tool_execution",
            timestamp=datetime.now(),
            data={
                "tool_calls": tool_calls,
                "tool_results": tool_results,
                "execution_time_seconds": parallel_execution_time,
                "successful_tools": len(successful_results),
                "failed_tools": len(failed_results),
                "thread_pool_reused": True,
                "max_workers_used": self.executor._max_workers,
                "aggregated_results": {
                    "total_tools": len(tool_results),
                    "successful_tools": len(successful_results),
                    "failed_tools": len(failed_results),
                    "total_execution_time": parallel_execution_time,
                    "individual_results": [
                        {
                            "tool_name": result.get('tool_name'),
                            "success": result.get('success'),
                            "execution_time": result.get('execution_time_seconds'),
                            "thread_name": result.get('thread_name'),
                            "result_summary": str(result.get('result', result.get('error')))[:200] + "..." if len(str(result.get('result', result.get('error')))) > 200 else str(result.get('result', result.get('error'))),
                            "input_received": result.get('input_received'),
                            "exception_type": result.get('exception_type') if not result.get('success') else None
                        }
                        for result in tool_results
                    ]
                }
            },
            event_id=f"tools_{len(context.events)}"
        )
        
        context.events.append(event)
        context.tool_results.extend(tool_results)
        
        # Prepare tool results for Claude
        self.logger.info("📝 PREPARING TOOL RESULTS FOR CLAUDE:")
        
        # Log aggregated results summary
        successful_results = [r for r in tool_results if r.get('success')]
        failed_results = [r for r in tool_results if not r.get('success')]
        
        self.logger.info(f"📊 AGGREGATED TOOL RESULTS SUMMARY:")
        self.logger.info(f"   ✅ Successful: {len(successful_results)}")
        self.logger.info(f"   ❌ Failed: {len(failed_results)}")
        
        # Log detailed aggregated results
        self.logger.info("📋 DETAILED AGGREGATED RESULTS:")
        for i, tool_result in enumerate(tool_results, 1):
            status = "SUCCESS" if tool_result.get('success') else "FAILED"
            tool_name = tool_result.get('tool_name', 'unknown')
            exec_time = tool_result.get('execution_time_seconds', 0)
            thread_name = tool_result.get('thread_name', 'unknown')
            
            self.logger.info(f"   {i}. {tool_name} - {status} ({exec_time:.2f}s) {thread_name}")
            
            if tool_result.get('success'):
                result_data = tool_result.get('result', {})
                self.logger.info(f"      📤 Result: {json.dumps(result_data, indent=6)}")
            else:
                error_msg = tool_result.get('error', 'Unknown error')
                exception_type = tool_result.get('exception_type', 'Unknown')
                self.logger.info(f"      ❌ Error ({exception_type}): {error_msg}")
        
        # Prepare the complete messages array that will be sent to Claude
        for i, (tool_call, tool_result) in enumerate(zip(tool_calls, tool_results), 1):
            if tool_result["success"]:
                result_content = json.dumps(tool_result["result"], indent=2)
                self.logger.info(f"   ✅ Tool {i} ({tool_call['name']}): Result prepared for Claude")
            else:
                result_content = f"Error: {tool_result['error']}"
                self.logger.info(f"   ❌ Tool {i} ({tool_call['name']}): Error prepared for Claude")
            
            # Add to messages
            context.messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": result_content
                }]
            })
        
        # Log the complete messages array that will be sent to Claude
        self.logger.info("📨 COMPLETE MESSAGES ARRAY AFTER TOOL RESULTS:")
        self.logger.info("=" * 80)
        self.logger.info(f"📋 COMPLETE Messages Array ({len(context.messages)} messages):")
        self.logger.info("🔍 FULL PAYLOAD - NO DETAILS HIDDEN:")
        
        for i, msg in enumerate(context.messages, 1):
            self.logger.info(f"\n--- MESSAGE {i} ---")
            self.logger.info(f"Role: {msg['role']}")
            
            if isinstance(msg['content'], str):
                # Simple string content
                self.logger.info(f"Content Type: string")
                self.logger.info(f"Content: {msg['content']}")
                
            elif isinstance(msg['content'], list):
                # List content (tool results or Claude response blocks)
                self.logger.info(f"Content Type: list ({len(msg['content'])} items)")
                
                for j, content_item in enumerate(msg['content']):
                    self.logger.info(f"\n  Content Item {j+1}:")
                    
                    if hasattr(content_item, 'model_dump'):
                        # Claude response content block
                        content_dict = content_item.model_dump()
                        self.logger.info(f"    Type: Claude Response Block")
                        self.logger.info(f"    Full Content: {json.dumps(content_dict, indent=6)}")
                        
                    elif isinstance(content_item, dict):
                        # Tool result or other dict content
                        self.logger.info(f"    Type: {content_item.get('type', 'unknown')}")
                        
                        if 'toolResult' in content_item:
                            self.logger.info(f"    Tool Use ID: {content_item['toolResult'].get('toolUseId')}")
                            self.logger.info(f"    Tool Result Content:")
                            self.logger.info(f"      {content_item['toolResult'].get('content')}")
                        else:
                            self.logger.info(f"    Full Content: {json.dumps(content_item, indent=6)}")
                    else:
                        # Other content types
                        self.logger.info(f"    Raw Content: {content_item}")
            
            else:
                # Other content types (shouldn't happen but just in case)
                self.logger.info(f"Content Type: {type(msg['content'])}")
                self.logger.info(f"Content: {msg['content']}")
        
        self.logger.info("\n" + "=" * 80)
        
        # Also log the complete request payload that will be sent
        complete_request_payload = {
            "model": self.model,
            "max_tokens": 4000,
            "messages": context.messages  # Include the actual messages, not just count
        }
        
        self.logger.info("🔧 COMPLETE API REQUEST PAYLOAD FOR NEXT CLAUDE CALL:")
        self.logger.info("📦 This is the EXACT payload that will be sent to Claude:")
        
        # Create a serializable version for logging
        serializable_messages = []
        for msg in context.messages:
            if isinstance(msg['content'], str):
                serializable_messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
            elif isinstance(msg['content'], list):
                content_list = []
                for content_item in msg['content']:
                    if hasattr(content_item, 'model_dump'):
                        content_list.append(content_item.model_dump())
                    else:
                        content_list.append(content_item)
                serializable_messages.append({
                    "role": msg['role'],
                    "content": content_list
                })
            else:
                serializable_messages.append({
                    "role": msg['role'],
                    "content": str(msg['content'])
                })
        
        complete_payload_for_logging = {
            "model": self.model,
            "max_tokens": 4000,
            "messages": serializable_messages
        }
        
        self.logger.info(json.dumps(complete_payload_for_logging, indent=2, ensure_ascii=False))
        self.logger.info("=" * 80)
        
        total_execution_time = (datetime.now() - execution_start_time).total_seconds()
        
        self.logger.info("🔧 TOOL EXECUTION COMPLETED")
        self.logger.info(f"   Parallel execution: {parallel_execution_time:.2f}s")
        self.logger.info(f"   Total processing: {total_execution_time:.2f}s")
        self.logger.info(f"   Messages added to context: {len(tool_calls)}")
        self.logger.info(f"   Ready for next state: PLANNING")
    
    async def _execute_tools_parallel_async(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute tools in parallel using the reusable ThreadPoolExecutor
        This is the same parallel execution logic, but integrated into the event loop
        """
        if not self.executor:
            error_msg = "ThreadPoolExecutor not initialized. Use context manager."
            self.logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        parallel_start_time = datetime.now()
        
        self.logger.info(f"⚡ PARALLEL TOOL EXECUTION ENGINE")
        self.logger.info(f"   Tools to execute: {len(tool_calls)}")
        self.logger.info(f"   Thread pool max workers: {self.executor._max_workers}")
        self.logger.info(f"   Execution method: asyncio.gather with ThreadPoolExecutor")
        
        # Submit all tool executions to the reusable thread pool
        loop = asyncio.get_event_loop()
        futures = []
        
        self.logger.info("🚀 SUBMITTING TOOLS TO THREAD POOL:")
        for i, tool_call in enumerate(tool_calls, 1):
            self.logger.info(f"   Submitting tool {i}: {tool_call['name']}")
            
            future = loop.run_in_executor(
                self.executor,  # Reuse the same thread pool
                self._execute_single_tool,
                tool_call['name'],
                tool_call['input']
            )
            futures.append(future)
        
        self.logger.info(f"✅ All {len(futures)} tools submitted to thread pool")
        self.logger.info("⏳ Waiting for parallel execution to complete...")
        
        # Wait for all tools to complete
        gather_start_time = datetime.now()
        results = await asyncio.gather(*futures, return_exceptions=True)
        gather_time = (datetime.now() - gather_start_time).total_seconds()
        
        self.logger.info(f"🏁 Parallel execution completed in {gather_time:.2f}s")
        
        # Process results and handle exceptions
        processed_results = []
        successful_count = 0
        failed_count = 0
        
        self.logger.info("🔍 PROCESSING EXECUTION RESULTS:")
        for i, (result, tool_call) in enumerate(zip(results, tool_calls), 1):
            if isinstance(result, Exception):
                self.logger.error(f"   Tool {i} ({tool_call['name']}): Exception - {result}")
                processed_results.append({
                    "tool_name": tool_call['name'],
                    "success": False,
                    "error": str(result),
                    "execution_time_seconds": 0
                })
                failed_count += 1
            else:
                status = "✅ SUCCESS" if result.get('success') else "❌ FAILED"
                exec_time = result.get('execution_time_seconds', 0)
                self.logger.info(f"   Tool {i} ({tool_call['name']}): {status} ({exec_time:.2f}s)")
                processed_results.append(result)
                if result.get('success'):
                    successful_count += 1
                else:
                    failed_count += 1
        
        total_parallel_time = (datetime.now() - parallel_start_time).total_seconds()
        
        self.logger.info("📊 PARALLEL EXECUTION SUMMARY:")
        self.logger.info(f"   Total parallel time: {total_parallel_time:.2f}s")
        self.logger.info(f"   Asyncio gather time: {gather_time:.2f}s")
        self.logger.info(f"   Successful tools: {successful_count}")
        self.logger.info(f"   Failed tools: {failed_count}")
        self.logger.info(f"   Thread pool reused: ✅")
        
        return processed_results
    
    def _execute_single_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool (runs in thread pool)"""
        start_time = datetime.now()
        thread_name = f"[{threading.current_thread().name}]"
        
        # Note: We can't use self.logger here directly as it may not be thread-safe
        # Instead, we'll return detailed information that gets logged by the main thread
        
        try:
            if tool_name not in TOOL_FUNCTIONS:
                return {
                    "tool_name": tool_name, 
                    "success": False, 
                    "error": f"Unknown tool: {tool_name}",
                    "execution_time_seconds": 0,
                    "thread_name": thread_name
                }
            
            tool_function = TOOL_FUNCTIONS[tool_name]
            result = tool_function(**tool_input)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "tool_name": tool_name,
                "success": True,
                "result": result,
                "execution_time_seconds": execution_time,
                "thread_name": thread_name,
                "input_received": tool_input
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                "tool_name": tool_name,
                "success": False,
                "error": str(e),
                "execution_time_seconds": execution_time,
                "thread_name": thread_name,
                "input_received": tool_input,
                "exception_type": type(e).__name__
            }
    
    async def _handle_planning(self, context: AgentContext):
        """Handle the planning state - decide next actions"""
        planning_start_time = datetime.now()
        
        self.logger.info("📋 AGENT PLANNING STATE")
        self.logger.info("🤔 Analyzing tool results and deciding next actions...")
        
        # Analyze tool results and decide whether to continue
        last_tool_event = next((e for e in reversed(context.events) if e.event_type == "parallel_tool_execution"), None)
        
        if last_tool_event:
            successful_tools = last_tool_event.data.get("successful_tools", 0)
            failed_tools = last_tool_event.data.get("failed_tools", 0)
            execution_time = last_tool_event.data.get("execution_time_seconds", 0)
            
            self.logger.info("📊 TOOL EXECUTION ANALYSIS:")
            self.logger.info(f"   Successful tools: {successful_tools}")
            self.logger.info(f"   Failed tools: {failed_tools}")
            self.logger.info(f"   Execution time: {execution_time:.2f}s")
            self.logger.info(f"   Success rate: {(successful_tools/(successful_tools+failed_tools)*100):.1f}%" if (successful_tools+failed_tools) > 0 else "   Success rate: N/A")
            
            # Analyze individual tool results
            tool_results = last_tool_event.data.get("tool_results", [])
            self.logger.info("🔍 INDIVIDUAL TOOL ANALYSIS:")
            for i, result in enumerate(tool_results, 1):
                status = "✅" if result.get('success') else "❌"
                tool_name = result.get('tool_name', 'unknown')
                exec_time = result.get('execution_time_seconds', 0)
                self.logger.info(f"   Tool {i}: {tool_name} {status} ({exec_time:.2f}s)")
                
                if not result.get('success'):
                    error = result.get('error', 'Unknown error')
                    self.logger.info(f"      Error: {error}")
        
        # Decision logic
        should_continue = self._should_continue_execution(context)
        
        self.logger.info("🎯 PLANNING DECISION:")
        self.logger.info(f"   Should continue execution: {should_continue}")
        self.logger.info(f"   Current iteration: {context.iteration_count}/{context.max_iterations}")
        self.logger.info(f"   Next state: {'THINKING' if should_continue else 'RESPONDING'}")
        
        if should_continue:
            self.logger.info("   Reason: Planning decided to continue with more tool execution")
        else:
            self.logger.info("   Reason: Planning decided to generate final response")
        
        # Log planning decision
        event = AgentEvent(
            event_type="planning_decision",
            timestamp=datetime.now(),
            data={
                "successful_tools": successful_tools if last_tool_event else 0,
                "failed_tools": failed_tools if last_tool_event else 0,
                "should_continue": should_continue,
                "iteration": context.iteration_count,
                "decision_reason": "Continue with more tools" if should_continue else "Generate final response",
                "tool_analysis": {
                    "total_tools": len(tool_results) if last_tool_event else 0,
                    "success_rate": (successful_tools/(successful_tools+failed_tools)*100) if last_tool_event and (successful_tools+failed_tools) > 0 else 0
                }
            },
            event_id=f"planning_{len(context.events)}"
        )
        
        context.events.append(event)
        
        planning_time = (datetime.now() - planning_start_time).total_seconds()
        self.logger.info(f"📋 PLANNING COMPLETED in {planning_time:.2f}s")
    
    async def _handle_responding(self, context: AgentContext):
        """Handle the responding state - generate final response"""
        responding_start_time = datetime.now()
        
        self.logger.info("💬 AGENT RESPONDING STATE")
        self.logger.info("🎯 Generating final response with all tool results...")
        
        # Log current conversation state
        self.logger.info("📊 CONVERSATION STATE BEFORE FINAL RESPONSE:")
        self.logger.info(f"   Total messages: {len(context.messages)}")
        self.logger.info(f"   Total events: {len(context.events)}")
        self.logger.info(f"   Total tool results: {len(context.tool_results)}")
        
        # Count message types
        user_messages = sum(1 for msg in context.messages if msg['role'] == 'user')
        assistant_messages = sum(1 for msg in context.messages if msg['role'] == 'assistant')
        
        self.logger.info(f"   User messages: {user_messages}")
        self.logger.info(f"   Assistant messages: {assistant_messages}")
        
        # Log the complete conversation that will be sent to Claude
        self.logger.info("📨 FINAL REQUEST TO CLAUDE FOR RESPONSE GENERATION:")
        self.logger.info("=" * 80)
        self.logger.info(f"📋 COMPLETE Messages Array ({len(context.messages)} messages):")
        self.logger.info("🔍 FULL PAYLOAD - NO DETAILS HIDDEN:")
        
        for i, msg in enumerate(context.messages, 1):
            self.logger.info(f"\n--- MESSAGE {i} ---")
            self.logger.info(f"Role: {msg['role']}")
            
            if isinstance(msg['content'], str):
                # Simple string content
                self.logger.info(f"Content Type: string")
                self.logger.info(f"Content: {msg['content']}")
                
            elif isinstance(msg['content'], list):
                # List content (tool results or Claude response blocks)
                self.logger.info(f"Content Type: list ({len(msg['content'])} items)")
                
                for j, content_item in enumerate(msg['content']):
                    self.logger.info(f"\n  Content Item {j+1}:")
                    
                    if hasattr(content_item, 'model_dump'):
                        # Claude response content block
                        content_dict = content_item.model_dump()
                        self.logger.info(f"    Type: Claude Response Block")
                        self.logger.info(f"    Full Content: {json.dumps(content_dict, indent=6)}")
                        
                    elif isinstance(content_item, dict):
                        # Tool result or other dict content
                        self.logger.info(f"    Type: {content_item.get('type', 'unknown')}")
                        
                        if 'toolResult' in content_item:
                            self.logger.info(f"    Tool Use ID: {content_item['toolResult'].get('toolUseId')}")
                            self.logger.info(f"    Tool Result Content:")
                            self.logger.info(f"      {content_item['toolResult'].get('content')}")
                        else:
                            self.logger.info(f"    Full Content: {json.dumps(content_item, indent=6)}")
                    else:
                        # Other content types
                        self.logger.info(f"    Raw Content: {content_item}")
            
            else:
                # Other content types (shouldn't happen but just in case)
                self.logger.info(f"Content Type: {type(msg['content'])}")
                self.logger.info(f"Content: {msg['content']}")
        
        self.logger.info("\n" + "=" * 80)
        
        # Also log the complete request payload that will be sent
        self.logger.info("🔧 COMPLETE FINAL API REQUEST PAYLOAD:")
        self.logger.info("📦 This is the EXACT payload sent to Claude for final response:")
        
        # Create a serializable version for logging
        serializable_messages = []
        for msg in context.messages:
            if isinstance(msg['content'], str):
                serializable_messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
            elif isinstance(msg['content'], list):
                content_list = []
                for content_item in msg['content']:
                    if hasattr(content_item, 'model_dump'):
                        content_list.append(content_item.model_dump())
                    else:
                        content_list.append(content_item)
                serializable_messages.append({
                    "role": msg['role'],
                    "content": content_list
                })
            else:
                serializable_messages.append({
                    "role": msg['role'],
                    "content": str(msg['content'])
                })
        
        complete_payload_for_logging = {
            "model": self.model,
            "max_tokens": 4000,
            "messages": serializable_messages
        }
        
        self.logger.info(json.dumps(complete_payload_for_logging, indent=2, ensure_ascii=False))
        self.logger.info("=" * 80)
        
        # Log request statistics
        self.logger.info("📊 FINAL REQUEST STATISTICS:")
        self.logger.info(f"   Model: {self.model}")
        self.logger.info(f"   Max tokens: 4000")
        self.logger.info(f"   Total messages: {len(context.messages)}")
        self.logger.info(f"   User messages: {sum(1 for msg in context.messages if msg['role'] == 'user')}")
        self.logger.info(f"   Assistant messages: {sum(1 for msg in context.messages if msg['role'] == 'assistant')}")
        self.logger.info(f"   Tool result messages: {sum(1 for msg in context.messages if msg['role'] == 'user' and isinstance(msg.get('content'), list))}")
        
        try:
            claude_request_start = datetime.now()
            
            final_response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=context.messages
            )
            
            claude_request_time = (datetime.now() - claude_request_start).total_seconds()
            
            self.logger.info(f"📥 Received final response from Claude in {claude_request_time:.2f}s")
            
            final_text = final_response.content[0].text if final_response.content else "No response"
            response_content = [block.model_dump() for block in final_response.content]
            
            self.logger.info("📄 FINAL RESPONSE ANALYSIS:")
            self.logger.info(f"   Response blocks: {len(response_content)}")
            self.logger.info(f"   Response length: {len(final_text)} characters")
            self.logger.info(f"   Response preview: {final_text[:300]}{'...' if len(final_text) > 300 else ''}")
            
            # Log the complete final response
            self.logger.info("📋 COMPLETE FINAL RESPONSE:")
            self.logger.info(json.dumps(response_content, indent=2, ensure_ascii=False))
            
            # Log the final response event
            event = AgentEvent(
                event_type="final_response",
                timestamp=datetime.now(),
                data={
                    "response_content": response_content,
                    "final_text": final_text,
                    "response_length": len(final_text),
                    "response_blocks": len(response_content),
                    "request_time_seconds": claude_request_time,
                    "model_used": self.model
                },
                event_id=f"response_{len(context.events)}"
            )
            
            context.events.append(event)
            
            total_responding_time = (datetime.now() - responding_start_time).total_seconds()
            
            self.logger.info("💬 RESPONDING COMPLETED")
            self.logger.info(f"   Claude request time: {claude_request_time:.2f}s")
            self.logger.info(f"   Total responding time: {total_responding_time:.2f}s")
            self.logger.info(f"   Final response generated: ✅")
            self.logger.info(f"   Ready for completion")
            
        except Exception as e:
            responding_error_time = (datetime.now() - responding_start_time).total_seconds()
            
            self.logger.error("❌ RESPONDING STATE FAILED")
            self.logger.error(f"   Error after: {responding_error_time:.2f}s")
            self.logger.error(f"   Error type: {type(e).__name__}")
            self.logger.error(f"   Error message: {str(e)}")
            self.logger.error(f"   Model: {self.model}")
            self.logger.error(f"   Messages count: {len(context.messages)}")
            self.logger.error("🔍 Full Exception:", exc_info=True)
            
            context.current_state = AgentState.ERROR
    
    def _should_continue_execution(self, context: AgentContext) -> bool:
        """Decide whether the agent should continue with more iterations"""
        
        # Simple heuristics for continuation
        if context.iteration_count >= context.max_iterations:
            return False
        
        # Check if there were recent failures that might need retry
        recent_events = context.events[-3:] if len(context.events) >= 3 else context.events
        recent_failures = sum(1 for e in recent_events 
                            if e.event_type == "parallel_tool_execution" 
                            and e.data.get("failed_tools", 0) > 0)
        
        # Don't continue if too many recent failures
        if recent_failures >= 2:
            return False
        
        # For this demo, we'll typically complete after one tool execution cycle
        return False
    
    def _generate_final_result(self, context: AgentContext) -> Dict[str, Any]:
        """Generate the final result from agent execution"""
        
        # Get final response
        final_response_event = next((e for e in reversed(context.events) if e.event_type == "final_response"), None)
        final_text = final_response_event.data.get("final_text", "No response generated") if final_response_event else "No response generated"
        
        # Calculate total execution time
        start_time = context.events[0].timestamp if context.events else datetime.now()
        end_time = context.events[-1].timestamp if context.events else datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Count tool executions
        tool_events = [e for e in context.events if e.event_type == "parallel_tool_execution"]
        total_tools = sum(len(e.data.get("tool_calls", [])) for e in tool_events)
        successful_tools = sum(e.data.get("successful_tools", 0) for e in tool_events)
        
        return {
            "conversation_id": context.conversation_id,
            "final_response": final_text,
            "agent_state": context.current_state.value,
            "total_iterations": context.iteration_count,
            "total_execution_time_seconds": total_time,
            "total_tools_executed": total_tools,
            "successful_tools": successful_tools,
            "failed_tools": total_tools - successful_tools,
            "events": [
                {
                    "event_type": e.event_type,
                    "timestamp": e.timestamp.isoformat(),
                    "data": e.data
                }
                for e in context.events
            ],
            "thread_pool_reused": True,
            "success": context.current_state == AgentState.COMPLETED
        }
    
    def _generate_error_result(self, context: AgentContext, error_message: str) -> Dict[str, Any]:
        """Generate error result"""
        return {
            "conversation_id": context.conversation_id,
            "error": error_message,
            "agent_state": context.current_state.value,
            "iterations_completed": context.iteration_count,
            "events": [
                {
                    "event_type": e.event_type,
                    "timestamp": e.timestamp.isoformat(),
                    "data": e.data
                }
                for e in context.events
            ],
            "success": False
        }


# Demo function
async def demo_agentic_event_loop():
    """Demonstrate the agentic event loop with reusable ThreadPoolExecutor"""
    
    print("🤖 Agentic Event Loop Demo with Reusable ThreadPoolExecutor")
    print("=" * 70)
    print("📝 Comprehensive logging enabled - check logs/ directory for details")
    print("=" * 70)
    
    # Use context manager to ensure proper thread pool lifecycle
    with AgenticEventLoop(max_workers=8) as agent:
        
        # Test queries that will trigger the event loop
        demo_queries = [
            "What's the weather in New York and the stock price of AAPL?",
            "I'm planning a trip from Boston to Miami. Get the weather for both cities, calculate the distance, and find Italian restaurants in Miami.",
            "Show me stock prices for GOOGL, TSLA, and AAPL, plus weather in San Francisco.",
            "Get weather for Seattle, calculate distance from Seattle to Portland, and find Asian restaurants in Portland."
        ]
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n🔍 Demo Query {i}:")
            print(f"User: {query}")
            print("-" * 50)
            
            # Run the agent
            result = await agent.run_agent(query)
            
            # Display results
            if result["success"]:
                print(f"✅ Agent completed successfully!")
                print(f"🔄 Iterations: {result['total_iterations']}")
                print(f"⏱️  Total time: {result['total_execution_time_seconds']:.2f}s")
                print(f"🔧 Tools executed: {result['total_tools_executed']}")
                print(f"✅ Successful tools: {result['successful_tools']}")
                print(f"🧵 Thread pool reused: {result['thread_pool_reused']}")
                print(f"\n🤖 Final response:")
                print(f"{result['final_response'][:200]}...")
            else:
                print(f"❌ Agent failed: {result.get('error', 'Unknown error')}")
            
            print(f"\n📄 Detailed logs available in logs/agentic_event_loop_*.log")
            print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(demo_agentic_event_loop())
