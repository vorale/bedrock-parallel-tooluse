"""
Agentic Event Loop with Direct Bedrock Converse API (boto3)
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
import boto3
from botocore.config import Config
from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

# Load environment variables
load_dotenv()

# Configure logging
def setup_agentic_logging():
    """Setup comprehensive logging for agentic event loop"""
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/agentic_bedrock_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('AgenticBedrockLoop')
    logger.info(f"🚀 Agentic Bedrock Event Loop logging initialized. Log file: {log_filename}")
    return logger

agentic_logger = setup_agentic_logging()

class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING_TOOLS = "executing_tools"
    PLANNING = "planning"
    RESPONDING = "responding"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AgentEvent:
    """Event data structure for agent execution tracking"""
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    context: Optional['AgentContext'] = None

@dataclass
class AgentContext:
    """Context for agent execution state"""
    conversation_id: str
    messages: List[Dict[str, Any]]
    events: List[AgentEvent]
    current_state: AgentState
    tool_results: List[Dict[str, Any]]
    max_iterations: int = 10
    current_iteration: int = 0

class AgenticBedrockLoop:
    """
    Agentic Event Loop using direct Bedrock Converse API
    """
    
    def __init__(self, aws_region: str = "us-east-1", max_workers: int = 8):
        self.aws_region = aws_region
        self.logger = logging.getLogger('AgenticBedrockLoop')
        
        # Initialize direct boto3 Bedrock client with retry configuration
        retry_config = Config(
            retries={
                'max_attempts': 10,
                'mode': 'adaptive'
            },
            read_timeout=300,
            connect_timeout=60
        )
        
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=self.aws_region,
            config=retry_config
        )
        
        self.model = "us.anthropic.claude-sonnet-4-20250514-v1:0"
        self.max_workers = max_workers
        self.executor = None
        
        self.logger.info(f"🤖 AgenticBedrockLoop initialized")
        self.logger.info(f"📍 AWS Region: {self.aws_region}")
        self.logger.info(f"🧠 Model: {self.model}")
        self.logger.info(f"🧵 Max Workers: {self.max_workers}")
    
    def __enter__(self):
        """Context manager entry - initialize thread pool"""
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="BedrockToolExecutor"
        )
        
        self.logger.info(f"🧵 ThreadPoolExecutor initialized")
        self.logger.info(f"   • Max Workers: {self.executor._max_workers}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup thread pool"""
        if self.executor:
            self.logger.info("🧵 Shutting down ThreadPoolExecutor...")
            self.executor.shutdown(wait=True)
            self.logger.info("✅ ThreadPoolExecutor shutdown completed")
    
    async def run_agent(self, user_message: str) -> Dict[str, Any]:
        """Execute agent with Bedrock Converse API"""
        conversation_id = f"bedrock_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        context = AgentContext(
            conversation_id=conversation_id,
            messages=[{"role": "user", "content": user_message}],
            events=[],
            current_state=AgentState.IDLE,
            tool_results=[]
        )
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 STARTING BEDROCK AGENTIC EVENT LOOP")
        self.logger.info("=" * 80)
        self.logger.info(f"🆔 Conversation ID: {conversation_id}")
        self.logger.info(f"👤 User Message: {user_message}")
        
        start_time = datetime.now()
        
        try:
            while (context.current_state != AgentState.COMPLETED and 
                   context.current_state != AgentState.ERROR and 
                   context.current_iteration < context.max_iterations):
                
                context.current_iteration += 1
                self.logger.info(f"🔄 ITERATION {context.current_iteration}/{context.max_iterations}")
                
                if context.current_state == AgentState.IDLE:
                    context.current_state = AgentState.THINKING
                elif context.current_state == AgentState.THINKING:
                    await self._handle_thinking_bedrock(context)
                elif context.current_state == AgentState.EXECUTING_TOOLS:
                    await self._handle_tool_execution(context)
                elif context.current_state == AgentState.RESPONDING:
                    await self._handle_responding_bedrock(context)
                
                if context.current_iteration >= context.max_iterations:
                    self.logger.warning("⚠️ Max iterations reached")
                    break
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info("=" * 80)
            self.logger.info("🏁 BEDROCK AGENT EXECUTION COMPLETED")
            self.logger.info(f"⏱️ Total Execution Time: {total_time:.2f}s")
            self.logger.info(f"🔄 Total Iterations: {context.current_iteration}")
            
            # Get final response
            final_response = ""
            if context.events:
                last_event = context.events[-1]
                if last_event.event_type == "final_response":
                    final_response = last_event.data.get("final_text", "")
            
            return {
                "response": final_response,
                "conversation_id": conversation_id,
                "total_time": total_time,
                "iterations": context.current_iteration,
                "events": len(context.events)
            }
            
        except Exception as e:
            self.logger.error(f"💥 Agent execution failed: {str(e)}")
            return {"error": str(e), "conversation_id": conversation_id}
    
    async def _handle_thinking_bedrock(self, context: AgentContext):
        """Handle thinking state using Bedrock Converse API"""
        self.logger.info("🧠 BEDROCK THINKING STATE")
        
        try:
            # Convert messages to Bedrock format
            bedrock_messages = []
            for msg in context.messages:
                if msg['role'] == 'user':
                    if isinstance(msg['content'], str):
                        bedrock_messages.append({
                            'role': 'user',
                            'content': [{'text': msg['content']}]
                        })
                    else:
                        bedrock_messages.append({
                            'role': 'user',
                            'content': msg['content']
                        })
                elif msg['role'] == 'assistant':
                    bedrock_messages.append({
                        'role': 'assistant',
                        'content': msg['content']
                    })
            
            # Convert tool schemas to Bedrock format
            bedrock_tools = []
            for tool in TOOL_SCHEMAS:
                bedrock_tool = {
                    'toolSpec': {
                        'name': tool['name'],
                        'description': tool['description'],
                        'inputSchema': {'json': tool['input_schema']}
                    }
                }
                bedrock_tools.append(bedrock_tool)
            
            # Log detailed request
            self.logger.info("📤 [LLM REQUEST] Sending to Bedrock Converse API")
            self.logger.info(f"📤 [LLM REQUEST] Model: {self.model}")
            self.logger.info(f"📤 [LLM REQUEST] Messages count: {len(bedrock_messages)}")
            self.logger.info(f"📤 [LLM REQUEST] Tools available: {len(bedrock_tools)}")
            self.logger.info("📤 [LLM REQUEST] Full request payload:")
            self.logger.info(json.dumps({
                "modelId": self.model,
                "messages": bedrock_messages,
                "toolConfig": {"tools": bedrock_tools},
                "inferenceConfig": {"maxTokens": 4000}
            }, indent=2, default=str))
            
            start_time = datetime.now()
            response = self.client.converse(
                modelId=self.model,
                messages=bedrock_messages,
                toolConfig={'tools': bedrock_tools},
                inferenceConfig={'maxTokens': 4000}
            )
            request_time = (datetime.now() - start_time).total_seconds()
            
            # Log detailed response
            self.logger.info(f"📩 [LLM RESPONSE] Received in {request_time:.2f}s")
            self.logger.info("📩 [LLM RESPONSE] Full response payload:")
            self.logger.info(json.dumps(response, indent=2, default=str))
            
            # Extract tool calls and text from Bedrock response
            tool_calls = []
            response_text_blocks = []
            
            if 'output' in response and 'message' in response['output']:
                message = response['output']['message']
                content = message.get('content', [])
                
                self.logger.info(f"📩 [LLM RESPONSE] Content blocks: {len(content)}")
                
                for i, block in enumerate(content):
                    if block.get('text'):
                        text_content = block['text']
                        response_text_blocks.append(text_content)
                        self.logger.info(f"📩 [LLM RESPONSE] Text block {i+1}: {text_content}")
                    elif block.get('toolUse'):
                        tool_use = block['toolUse']
                        tool_call = {
                            'id': tool_use['toolUseId'],
                            'name': tool_use['name'],
                            'input': tool_use['input']
                        }
                        tool_calls.append(tool_call)
                        self.logger.info(f"📩 [LLM RESPONSE] Tool call {len(tool_calls)}: {tool_call['name']}")
                        self.logger.info(f"📩 [LLM RESPONSE] Tool ID: {tool_call['id']}")
                        self.logger.info(f"📩 [LLM RESPONSE] Tool input: {json.dumps(tool_call['input'], indent=2)}")
            
            # Convert response to assistant message format for Bedrock
            assistant_content = []
            for text in response_text_blocks:
                assistant_content.append({'text': text})
            for tool_call in tool_calls:
                assistant_content.append({
                    'toolUse': {
                        'toolUseId': tool_call['id'],
                        'name': tool_call['name'],
                        'input': tool_call['input']
                    }
                })
            
            context.messages.append({"role": "assistant", "content": assistant_content})
            
            if tool_calls:
                context.current_state = AgentState.EXECUTING_TOOLS
                self.logger.info(f"🔧 {len(tool_calls)} tools requested - moving to execution")
            else:
                context.current_state = AgentState.RESPONDING
                self.logger.info("📝 No tools requested - moving to response")
                
        except Exception as e:
            self.logger.error(f"❌ BEDROCK THINKING FAILED: {str(e)}")
            context.current_state = AgentState.ERROR
    
    async def _handle_tool_execution(self, context: AgentContext):
        """Handle tool execution using ThreadPoolExecutor"""
        self.logger.info("🔧 BEDROCK TOOL EXECUTION STATE")
        
        # Get tool calls from last assistant message
        tool_calls = []
        last_message = context.messages[-1]
        if last_message['role'] == 'assistant':
            for content in last_message['content']:
                if 'toolUse' in content:
                    tool_use = content['toolUse']
                    tool_calls.append({
                        'id': tool_use['toolUseId'],
                        'name': tool_use['name'],
                        'input': tool_use['input']
                    })
        
        if not tool_calls:
            self.logger.error("❌ No tool calls found")
            context.current_state = AgentState.ERROR
            return
        
        # Log tool execution details
        self.logger.info(f"🔧 [TOOL EXECUTION] Starting parallel execution of {len(tool_calls)} tools")
        for i, tool_call in enumerate(tool_calls, 1):
            self.logger.info(f"🔧 [TOOL EXECUTION] Tool {i}/{len(tool_calls)}: {tool_call['name']}")
            self.logger.info(f"🔧 [TOOL EXECUTION] Tool ID: {tool_call['id']}")
            self.logger.info(f"🔧 [TOOL EXECUTION] Tool input: {json.dumps(tool_call['input'], indent=2)}")
        
        # Execute tools in parallel
        tool_results = await self._execute_tools_parallel_async(tool_calls)
        
        # Log tool results
        self.logger.info(f"✅ [TOOL RESULTS] Completed {len(tool_results)} tools")
        for i, (tool_call, tool_result) in enumerate(zip(tool_calls, tool_results), 1):
            status = "✅ SUCCESS" if tool_result["success"] else "❌ FAILED"
            exec_time = tool_result.get("execution_time_seconds", 0)
            self.logger.info(f"✅ [TOOL RESULTS] Tool {i}/{len(tool_results)}: {tool_call['name']} - {status} ({exec_time:.2f}s)")
            
            if tool_result["success"]:
                self.logger.info(f"✅ [TOOL RESULTS] Result: {json.dumps(tool_result['result'], indent=2)}")
            else:
                self.logger.info(f"❌ [TOOL RESULTS] Error: {tool_result['error']}")
        
        # Add ALL tool results to a SINGLE user message in Bedrock format
        tool_result_content = []
        for tool_call, tool_result in zip(tool_calls, tool_results):
            if tool_result["success"]:
                result_content = json.dumps(tool_result["result"], indent=2)
            else:
                result_content = f"Error: {tool_result['error']}"
            
            tool_result_content.append({
                "toolResult": {
                    "toolUseId": tool_call["id"],
                    "content": [{"text": result_content}]
                }
            })
        
        # Add single user message with ALL tool results
        context.messages.append({
            "role": "user",
            "content": tool_result_content
        })
        
        self.logger.info(f"📤 [TOOL RESULTS] Added {len(tool_result_content)} tool results to conversation")
        
        context.current_state = AgentState.RESPONDING
    
    async def _handle_responding_bedrock(self, context: AgentContext):
        """Handle responding state using Bedrock Converse API"""
        self.logger.info("📝 BEDROCK RESPONDING STATE")
        
        try:
            # Convert messages to Bedrock format
            bedrock_messages = []
            for msg in context.messages:
                if msg['role'] == 'user':
                    if isinstance(msg['content'], str):
                        bedrock_messages.append({
                            'role': 'user',
                            'content': [{'text': msg['content']}]
                        })
                    else:
                        bedrock_messages.append({
                            'role': 'user',
                            'content': msg['content']
                        })
                elif msg['role'] == 'assistant':
                    bedrock_messages.append({
                        'role': 'assistant',
                        'content': msg['content']
                    })
            
            # Convert tool schemas for final response
            bedrock_tools = []
            for tool in TOOL_SCHEMAS:
                bedrock_tool = {
                    'toolSpec': {
                        'name': tool['name'],
                        'description': tool['description'],
                        'inputSchema': {'json': tool['input_schema']}
                    }
                }
                bedrock_tools.append(bedrock_tool)
            
            # Log detailed final request
            self.logger.info("📤 [FINAL REQUEST] Sending final request to Bedrock")
            self.logger.info(f"📤 [FINAL REQUEST] Model: {self.model}")
            self.logger.info(f"📤 [FINAL REQUEST] Messages count: {len(bedrock_messages)}")
            self.logger.info("📤 [FINAL REQUEST] Full conversation:")
            for i, msg in enumerate(bedrock_messages):
                self.logger.info(f"📤 [FINAL REQUEST] Message {i+1} ({msg['role']}):")
                if isinstance(msg['content'], list):
                    for j, content in enumerate(msg['content']):
                        if 'text' in content:
                            self.logger.info(f"📤 [FINAL REQUEST]   Text: {content['text']}")
                        elif 'toolUse' in content:
                            self.logger.info(f"📤 [FINAL REQUEST]   ToolUse: {content['toolUse']['name']} (ID: {content['toolUse']['toolUseId']})")
                        elif 'toolResult' in content:
                            self.logger.info(f"📤 [FINAL REQUEST]   ToolResult: ID {content['toolResult']['toolUseId']}")
            
            start_time = datetime.now()
            final_response = self.client.converse(
                modelId=self.model,
                messages=bedrock_messages,
                toolConfig={'tools': bedrock_tools},
                inferenceConfig={'maxTokens': 4000}
            )
            request_time = (datetime.now() - start_time).total_seconds()
            
            # Log detailed final response
            self.logger.info(f"📩 [FINAL RESPONSE] Received in {request_time:.2f}s")
            self.logger.info("📩 [FINAL RESPONSE] Full response payload:")
            self.logger.info(json.dumps(final_response, indent=2, default=str))
            
            # Extract final text from Bedrock response
            final_text = "No response"
            if 'output' in final_response and 'message' in final_response['output']:
                message = final_response['output']['message']
                content = message.get('content', [])
                
                self.logger.info(f"📩 [FINAL RESPONSE] Content blocks: {len(content)}")
                
                for i, block in enumerate(content):
                    if block.get('text'):
                        final_text = block['text']
                        self.logger.info(f"📩 [FINAL RESPONSE] Text block {i+1}: {final_text}")
                        break
            
            # Create final response event
            event = AgentEvent(
                event_type="final_response",
                timestamp=datetime.now(),
                data={
                    "final_text": final_text,
                    "response_length": len(final_text),
                    "request_time_seconds": request_time
                },
                context=context
            )
            
            context.events.append(event)
            context.current_state = AgentState.COMPLETED
            
            self.logger.info(f"✅ [FINAL RESPONSE] Generated: {len(final_text)} characters")
            self.logger.info(f"✅ [FINAL RESPONSE] Preview: {final_text[:200]}{'...' if len(final_text) > 200 else ''}")
            
        except Exception as e:
            self.logger.error(f"❌ BEDROCK RESPONDING FAILED: {str(e)}")
            context.current_state = AgentState.ERROR
    
    async def _execute_tools_parallel_async(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tools in parallel using ThreadPoolExecutor"""
        if not self.executor:
            raise RuntimeError("ThreadPoolExecutor not initialized")
        
        self.logger.info(f"⚡ PARALLEL TOOL EXECUTION: {len(tool_calls)} tools")
        
        loop = asyncio.get_event_loop()
        futures = []
        
        for tool_call in tool_calls:
            future = loop.run_in_executor(
                self.executor,
                self._execute_single_tool,
                tool_call['name'],
                tool_call['input']
            )
            futures.append(future)
        
        start_time = datetime.now()
        results = await asyncio.gather(*futures, return_exceptions=True)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        self.logger.info(f"🏁 Parallel execution completed in {execution_time:.2f}s")
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "tool_name": tool_calls[i]['name'],
                    "success": False,
                    "error": str(result),
                    "execution_time_seconds": 0
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _execute_single_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool function"""
        start_time = datetime.now()
        
        try:
            if tool_name not in TOOL_FUNCTIONS:
                return {
                    "tool_name": tool_name,
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                    "execution_time_seconds": 0
                }
            
            tool_function = TOOL_FUNCTIONS[tool_name]
            result = tool_function(**tool_input)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "tool_name": tool_name,
                "success": True,
                "result": result,
                "execution_time_seconds": execution_time,
                "thread_name": threading.current_thread().name
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                "tool_name": tool_name,
                "success": False,
                "error": str(e),
                "exception_type": type(e).__name__,
                "execution_time_seconds": execution_time,
                "thread_name": threading.current_thread().name
            }

# Demo function
async def demo_bedrock_agentic_loop():
    """Demonstrate the Bedrock agentic event loop"""
    
    print("🤖 Bedrock Agentic Event Loop Demo")
    print("=" * 50)
    
    demo_queries = [
        "What's the weather in New York and the stock price of AAPL?",
        "I'm planning a trip from Boston to Miami. Get the weather for both cities, calculate the distance, and find Italian restaurants in Miami.",
        "Show me stock prices for GOOGL, TSLA, and AAPL, plus weather in San Francisco.",
        "Get weather for Seattle, calculate distance from Seattle to Portland, and find Asian restaurants in Portland."
    ]
    
    with AgenticBedrockLoop(max_workers=8) as agent:
        for i, query in enumerate(demo_queries, 1):
            print(f"\n📋 Demo Query {i}:")
            print(f"❓ {query}")
            print("-" * 40)
            
            result = await agent.run_agent(query)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"🤖 Response: {result['response']}")
                print(f"⏱️ Time: {result['total_time']:.2f}s")
                print(f"🔄 Iterations: {result['iterations']}")
            
            print("=" * 50)

if __name__ == "__main__":
    asyncio.run(demo_bedrock_agentic_loop())
