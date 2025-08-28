"""
Parallel Tool Execution Demo using Anthropic SDK with Bedrock
"""
import os
import json
import asyncio
import concurrent.futures
import logging
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from anthropic import AnthropicBedrock
from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging():
    """Setup comprehensive logging for requests and responses"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/parallel_tool_execution_{timestamp}.log"
    
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
    logger = logging.getLogger('ParallelToolExecutor')
    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger

# Initialize logging
logger = setup_logging()


class ParallelToolExecutor:
    """
    Handles parallel execution of tools with AnthropicBedrock client
    """
    
    def __init__(self, aws_region: str = None):
        """
        Initialize the parallel tool executor
        
        Args:
            aws_region: AWS region for Bedrock (defaults to environment variable)
        """
        self.aws_region = aws_region or os.getenv('AWS_REGION', 'us-east-1')
        self.logger = logging.getLogger('ParallelToolExecutor')
        
        # Initialize AnthropicBedrock client with retry configuration
        import boto3
        from botocore.config import Config
        
        # Configure retry settings
        retry_config = Config(
            retries={
                'max_attempts': 10,  # Increase from default 4 to 10
                'mode': 'adaptive'   # Use adaptive retry mode
            },
            read_timeout=300,        # 5 minutes read timeout
            connect_timeout=60       # 1 minute connect timeout
        )
        
        # Create Bedrock client with retry configuration
        bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=self.aws_region,
            config=retry_config
        )
        
        self.client = AnthropicBedrock(
            client=bedrock_client,
            timeout=300.0,  # 5 minute timeout
            # AWS credentials will be automatically picked up from environment
            # or AWS credential chain (IAM roles, profiles, etc.)
        )
        
        self.model = "us.anthropic.claude-sonnet-4-20250514-v1:0"  # Latest Claude 4 Opus with cross-region inference
        
        # Alternative Claude 4 models available:
        # self.model = "anthropic.claude-sonnet-4-20250514-v1:0"  # Claude 4 Sonnet
        # self.model = "anthropic.claude-opus-4-1-20250805-v1:0"  # Claude 4.1 Opus
        # self.model = "anthropic.claude-3-7-sonnet-20250219-v1:0"  # Claude 3.7 Sonnet
        
        self.logger.info(f"ParallelToolExecutor initialized with model: {self.model}, region: {self.aws_region}")
        self.logger.info(f"Available tools: {list(TOOL_FUNCTIONS.keys())}")
    
    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single tool function
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool
            
        Returns:
            Dictionary containing tool execution result
        """
        start_time = datetime.now()
        self.logger.info(f"🔧 Starting tool execution: {tool_name}")
        self.logger.info(f"📥 Tool input: {json.dumps(tool_input, indent=2)}")
        
        try:
            if tool_name not in TOOL_FUNCTIONS:
                error_msg = f"Unknown tool: {tool_name}"
                self.logger.error(f"❌ {error_msg}")
                return {"error": error_msg}
            
            tool_function = TOOL_FUNCTIONS[tool_name]
            result = tool_function(**tool_input)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            response = {
                "tool_name": tool_name,
                "success": True,
                "result": result,
                "execution_time_seconds": execution_time
            }
            
            self.logger.info(f"✅ Tool {tool_name} completed successfully in {execution_time:.2f}s")
            self.logger.info(f"📤 Tool output: {json.dumps(result, indent=2)}")
            
            return response
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            
            response = {
                "tool_name": tool_name,
                "success": False,
                "error": error_msg,
                "execution_time_seconds": execution_time
            }
            
            self.logger.error(f"❌ Tool {tool_name} failed after {execution_time:.2f}s: {error_msg}")
            return response
    
    def execute_tools_parallel(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute multiple tools in parallel using ThreadPoolExecutor
        
        Args:
            tool_calls: List of tool call dictionaries with 'name' and 'input' keys
            
        Returns:
            List of tool execution results
        """
        start_time = datetime.now()
        self.logger.info(f"⚡ Starting parallel execution of {len(tool_calls)} tools")
        
        # Log all tool calls that will be executed
        for i, tool_call in enumerate(tool_calls, 1):
            self.logger.info(f"📋 Tool {i}/{len(tool_calls)}: {tool_call['name']} with input: {json.dumps(tool_call['input'])}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
            # Submit all tool executions
            future_to_tool = {
                executor.submit(self.execute_tool, tool_call['name'], tool_call['input']): tool_call
                for tool_call in tool_calls
            }
            
            # Collect results as they complete
            results = []
            completed_count = 0
            
            for future in concurrent.futures.as_completed(future_to_tool):
                result = future.result()
                results.append(result)
                completed_count += 1
                
                tool_name = result.get('tool_name', 'unknown')
                status = "✅ SUCCESS" if result.get('success') else "❌ FAILED"
                self.logger.info(f"🏁 Tool completed ({completed_count}/{len(tool_calls)}): {tool_name} - {status}")
        
        total_execution_time = (datetime.now() - start_time).total_seconds()
        successful_tools = sum(1 for r in results if r.get('success'))
        failed_tools = len(results) - successful_tools
        
        self.logger.info(f"🎯 Parallel execution completed in {total_execution_time:.2f}s")
        self.logger.info(f"📊 Results: {successful_tools} successful, {failed_tools} failed")
        
        return results
    
    def process_message_with_tools(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message that may require tool usage
        
        Args:
            user_message: The user's input message
            
        Returns:
            Dictionary containing the conversation flow and results
        """
        conversation_start_time = datetime.now()
        conversation_id = f"conv_{conversation_start_time.strftime('%Y%m%d_%H%M%S_%f')}"
        
        self.logger.info(f"🚀 Starting conversation {conversation_id}")
        self.logger.info(f"👤 User message: {user_message}")
        
        conversation_log = {
            "conversation_id": conversation_id,
            "user_message": user_message,
            "start_time": conversation_start_time.isoformat(),
            "steps": []
        }
        
        try:
            # Initial message to Claude
            messages = [{"role": "user", "content": user_message}]
            
            self.logger.info(f"📤 Sending initial request to Claude model: {self.model}")
            self.logger.info(f"📋 Available tools: {[tool['name'] for tool in TOOL_SCHEMAS]}")
            
            # First API call to get tool usage
            claude_request_start = datetime.now()
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=8000,  # Reasonable token limit to avoid timeout issues
                    tools=TOOL_SCHEMAS,
                    messages=messages
                )
            except ValueError as e:
                if "Streaming is required" in str(e):
                    self.logger.error("❌ Request too large, try reducing complexity or use streaming")
                    raise ValueError("Request too complex. Try asking for fewer tools or simpler queries.")
                raise
            claude_request_time = (datetime.now() - claude_request_start).total_seconds()
            
            self.logger.info(f"📥 Received initial response from Claude in {claude_request_time:.2f}s")
            self.logger.info(f"🔍 Response content blocks: {len(response.content)}")
            
            # Log the full response content
            response_content = [block.model_dump() for block in response.content]
            self.logger.info(f"📄 Claude initial response: {json.dumps(response_content, indent=2)}")
            
            conversation_log["steps"].append({
                "step": "initial_response",
                "timestamp": datetime.now().isoformat(),
                "request_time_seconds": claude_request_time,
                "response_content": response_content
            })
            
            # Check if Claude wants to use tools
            tool_calls = []
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_calls.append({
                        "id": content_block.id,
                        "name": content_block.name,
                        "input": content_block.input
                    })
            
            self.logger.info(f"🔧 Claude requested {len(tool_calls)} tool(s)")
            
            if not tool_calls:
                # No tools needed, return the response
                final_response = response.content[0].text if response.content else "No response"
                conversation_log["final_response"] = final_response
                conversation_log["tools_used"] = False
                
                total_time = (datetime.now() - conversation_start_time).total_seconds()
                conversation_log["total_time_seconds"] = total_time
                
                self.logger.info(f"✅ Conversation {conversation_id} completed without tools in {total_time:.2f}s")
                self.logger.info(f"🤖 Final response: {final_response}")
                
                return conversation_log
            
            # Execute tools in parallel
            self.logger.info(f"⚡ Executing {len(tool_calls)} tools in parallel...")
            tool_execution_start = datetime.now()
            tool_results = self.execute_tools_parallel(tool_calls)
            tool_execution_time = (datetime.now() - tool_execution_start).total_seconds()
            
            self.logger.info(f"🎯 All tools completed in {tool_execution_time:.2f}s")
            
            conversation_log["steps"].append({
                "step": "parallel_tool_execution",
                "timestamp": datetime.now().isoformat(),
                "execution_time_seconds": tool_execution_time,
                "tool_calls": tool_calls,
                "tool_results": tool_results,
                "aggregated_results": {
                    "total_tools": len(tool_results),
                    "successful_tools": len([r for r in tool_results if r.get('success')]),
                    "failed_tools": len([r for r in tool_results if not r.get('success')]),
                    "total_execution_time": tool_execution_time,
                    "individual_results": [
                        {
                            "tool_name": result.get('tool_name'),
                            "success": result.get('success'),
                            "execution_time": result.get('execution_time_seconds'),
                            "result_summary": str(result.get('result', result.get('error')))[:200] + "..." if len(str(result.get('result', result.get('error')))) > 200 else str(result.get('result', result.get('error')))
                        }
                        for result in tool_results
                    ]
                }
            })
            
            # Prepare tool result messages for Claude
            messages.append({"role": "assistant", "content": response.content})
            
            # Add tool results to the conversation
            self.logger.info("📝 Preparing tool results for Claude...")
            
            # Log aggregated results summary
            successful_results = [r for r in tool_results if r.get('success')]
            failed_results = [r for r in tool_results if not r.get('success')]
            
            self.logger.info(f"📊 Aggregated Tool Results Summary:")
            self.logger.info(f"   ✅ Successful: {len(successful_results)}")
            self.logger.info(f"   ❌ Failed: {len(failed_results)}")
            
            # Log detailed aggregated results
            self.logger.info("📋 Detailed Aggregated Results:")
            for i, tool_result in enumerate(tool_results, 1):
                status = "SUCCESS" if tool_result.get('success') else "FAILED"
                tool_name = tool_result.get('tool_name', 'unknown')
                exec_time = tool_result.get('execution_time_seconds', 0)
                
                self.logger.info(f"   {i}. {tool_name} - {status} ({exec_time:.2f}s)")
                
                if tool_result.get('success'):
                    result_data = tool_result.get('result', {})
                    self.logger.info(f"      📤 Result: {json.dumps(result_data, indent=6)}")
                else:
                    error_msg = tool_result.get('error', 'Unknown error')
                    self.logger.info(f"      ❌ Error: {error_msg}")
            
            # Prepare the complete messages array that will be sent to Claude
            for i, tool_call in enumerate(tool_calls):
                tool_result = tool_results[i]
                
                if tool_result["success"]:
                    result_content = json.dumps(tool_result["result"], indent=2)
                    self.logger.info(f"✅ Tool {tool_call['name']} result prepared for Claude")
                else:
                    result_content = f"Error: {tool_result['error']}"
                    self.logger.warning(f"❌ Tool {tool_call['name']} error prepared for Claude: {tool_result['error']}")
                
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": result_content
                    }]
                })
            
            # Log the complete request that will be sent to Claude
            self.logger.info("📨 Complete Request to Claude with Tool Results:")
            self.logger.info("=" * 80)
            
            # Log the COMPLETE messages array with ALL details - no folding or hiding
            self.logger.info(f"📋 COMPLETE Messages Array ({len(messages)} messages):")
            self.logger.info("🔍 FULL PAYLOAD - NO DETAILS HIDDEN:")
            
            for i, msg in enumerate(messages, 1):
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
                            
                            if content_item.get('type') == 'tool_result':
                                self.logger.info(f"    Tool Use ID: {content_item.get('tool_use_id')}")
                                self.logger.info(f"    Tool Result Content:")
                                self.logger.info(f"      {content_item.get('content')}")
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
            
            # Also log the complete request parameters that will be sent
            complete_request_payload = {
                "model": self.model,
                "max_tokens": 4000,
                "messages": messages  # Include the actual messages, not just count
            }
            
            self.logger.info("🔧 COMPLETE API REQUEST PAYLOAD:")
            self.logger.info("📦 This is the EXACT payload sent to Claude:")
            
            # Create a serializable version for logging
            serializable_messages = []
            for msg in messages:
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
            
            # Final response from Claude with tool results
            self.logger.info("📤 Sending tool results to Claude for final response...")
            
            # Log the complete final request details - no summaries
            self.logger.info(f"🔧 FINAL API REQUEST - COMPLETE DETAILS:")
            self.logger.info(f"   Model: {self.model}")
            self.logger.info(f"   Max Tokens: 4000")
            self.logger.info(f"   Total Messages in Request: {len(messages)}")
            self.logger.info(f"   Tool Results Being Sent: {len(tool_calls)}")
            
            # Log each message being sent in the final request
            self.logger.info(f"📋 FINAL REQUEST MESSAGE BREAKDOWN:")
            for i, msg in enumerate(messages, 1):
                self.logger.info(f"   Message {i}:")
                self.logger.info(f"     Role: {msg['role']}")
                if isinstance(msg['content'], str):
                    self.logger.info(f"     Content: {msg['content']}")
                elif isinstance(msg['content'], list):
                    self.logger.info(f"     Content: {len(msg['content'])} items")
                    for j, item in enumerate(msg['content']):
                        if hasattr(item, 'model_dump'):
                            self.logger.info(f"       Item {j+1}: Claude Response Block")
                        elif isinstance(item, dict) and item.get('type') == 'tool_result':
                            self.logger.info(f"       Item {j+1}: Tool Result (ID: {item.get('tool_use_id')})")
                        else:
                            self.logger.info(f"       Item {j+1}: {type(item)}")
            
            final_claude_request_start = datetime.now()
            
            try:
                final_response = self.client.messages.create(
                    model=self.model,
                    max_tokens=8000,  # Reasonable token limit to avoid timeout issues
                    messages=messages
                )
            except ValueError as e:
                if "Streaming is required" in str(e):
                    self.logger.error("❌ Final response too large, try reducing complexity")
                    raise ValueError("Final response too complex. Try asking for simpler queries.")
                raise
            
            final_claude_request_time = (datetime.now() - final_claude_request_start).total_seconds()
            
            self.logger.info(f"📥 Received final response from Claude in {final_claude_request_time:.2f}s")
            
            final_response_content = [block.model_dump() for block in final_response.content]
            self.logger.info(f"📄 Claude final response: {json.dumps(final_response_content, indent=2)}")
            
            conversation_log["steps"].append({
                "step": "final_response_with_results",
                "timestamp": datetime.now().isoformat(),
                "request_time_seconds": final_claude_request_time,
                "response_content": final_response_content,
                "complete_request_payload": {
                    "model": self.model,
                    "max_tokens": 4000,
                    "messages": [
                        {
                            "role": msg["role"],
                            "content": msg["content"] if isinstance(msg["content"], str) else [
                                item.model_dump() if hasattr(item, 'model_dump') else item
                                for item in msg["content"]
                            ] if isinstance(msg["content"], list) else str(msg["content"])
                        }
                        for msg in messages
                    ]
                },
                "request_statistics": {
                    "total_messages": len(messages),
                    "tool_results_count": len(tool_calls),
                    "message_types": [
                        {
                            "message_index": i,
                            "role": msg["role"],
                            "content_type": "string" if isinstance(msg["content"], str) else "list" if isinstance(msg["content"], list) else "other",
                            "content_items": len(msg["content"]) if isinstance(msg["content"], list) else 1
                        }
                        for i, msg in enumerate(messages)
                    ]
                }
            })
            
            final_text = final_response.content[0].text if final_response.content else "No final response"
            conversation_log["final_response"] = final_text
            conversation_log["tools_used"] = True
            conversation_log["tools_count"] = len(tool_calls)
            
            total_time = (datetime.now() - conversation_start_time).total_seconds()
            conversation_log["total_time_seconds"] = total_time
            
            # Log comprehensive conversation summary
            self.logger.info("🎯 CONVERSATION SUMMARY")
            self.logger.info("=" * 60)
            self.logger.info(f"📋 Conversation ID: {conversation_id}")
            self.logger.info(f"⏱️  Total Duration: {total_time:.2f}s")
            self.logger.info(f"🔧 Tools Executed: {len(tool_calls)}")
            self.logger.info(f"✅ Successful Tools: {len([r for r in tool_results if r.get('success')])}")
            self.logger.info(f"❌ Failed Tools: {len([r for r in tool_results if not r.get('success')])}")
            self.logger.info(f"📊 API Calls Made: 2 (initial + final)")
            self.logger.info(f"📝 Final Response Length: {len(final_text)} characters")
            
            # Log tool execution breakdown
            self.logger.info("🔧 Tool Execution Breakdown:")
            for i, (tool_call, tool_result) in enumerate(zip(tool_calls, tool_results), 1):
                status = "✅ SUCCESS" if tool_result.get('success') else "❌ FAILED"
                exec_time = tool_result.get('execution_time_seconds', 0)
                self.logger.info(f"   {i}. {tool_call['name']}: {status} ({exec_time:.2f}s)")
            
            self.logger.info("=" * 60)
            
            self.logger.info(f"✅ Conversation {conversation_id} completed with {len(tool_calls)} tools in {total_time:.2f}s")
            self.logger.info(f"🤖 Final response: {final_text}")
            
        except Exception as e:
            error_msg = str(e)
            total_time = (datetime.now() - conversation_start_time).total_seconds()
            
            conversation_log["error"] = error_msg
            conversation_log["total_time_seconds"] = total_time
            
            self.logger.error(f"💥 Conversation {conversation_id} failed after {total_time:.2f}s: {error_msg}")
            self.logger.error(f"🔍 Error details: {e}", exc_info=True)
        
        return conversation_log


async def main():
    """
    Main demo function showing parallel tool execution
    """
    print("🚀 Parallel Tool Execution Demo with AnthropicBedrock")
    print("=" * 60)
    
    # Initialize the executor
    executor = ParallelToolExecutor()
    
    # Show logging information
    print(f"📝 Logs are being written to the logs/ directory")
    print(f"🔍 Check the log files for detailed request/response information")
    print("=" * 60)
    
    # Example queries that will trigger multiple tool calls
    demo_queries = [
        "What's the weather like in New York and Los Angeles? Also, what's the current stock price of AAPL?",
        # "I'm planning a trip from San Francisco to Seattle. Can you get the weather for both cities, calculate the distance between them, and find some good Italian restaurants in Seattle?",
        # "Give me the stock prices for GOOGL and TSLA, and also tell me the weather in Chicago."
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n🔍 Demo Query {i}:")
        print(f"User: {query}")
        print("-" * 40)
        
        # Process the query
        result = executor.process_message_with_tools(query)
        
        # Display results
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue
        
        # Show tool execution details
        for step in result["steps"]:
            if step["step"] == "parallel_tool_execution":
                print(f"⚡ Executed {len(step['tool_calls'])} tools in parallel:")
                for tool_call, tool_result in zip(step['tool_calls'], step['tool_results']):
                    status = "✅" if tool_result['success'] else "❌"
                    execution_time = tool_result.get('execution_time_seconds', 0)
                    print(f"  {status} {tool_call['name']}: {tool_call['input']} ({execution_time:.2f}s)")
        
        # Show timing information
        total_time = result.get('total_time_seconds', 0)
        tools_used = result.get('tools_used', False)
        tools_count = result.get('tools_count', 0)
        
        print(f"\n⏱️  Total conversation time: {total_time:.2f}s")
        if tools_used:
            print(f"🔧 Tools used: {tools_count}")
        
        # Show final response
        print(f"\n🤖 Claude's Response:")
        print(result["final_response"])
        print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
