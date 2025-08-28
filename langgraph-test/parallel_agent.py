"""
LangGraph-based parallel tool execution agent
"""
import logging
import time
import os
import json
from datetime import datetime
from typing import Literal
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from tools import get_weather, get_stock_price, calculate_distance, search_restaurants

# Create logs directory
logs_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
log_filename = os.path.join(logs_dir, f'langgraph_parallel_execution_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.log')

# Clear any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Clear existing loggers
for logger_name in list(logging.Logger.manager.loggerDict.keys()):
    if logger_name.startswith('langchain') or logger_name.startswith('LangGraphParallelAgent'):
        logging.getLogger(logger_name).handlers.clear()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("LangGraphParallelAgent")

# Enable debug logging for LangChain/LangGraph
langchain_logger = logging.getLogger("langchain")
langchain_logger.setLevel(logging.DEBUG)
langchain_logger.propagate = True

class ParallelLangGraphAgent:
    def __init__(self):
        """Initialize the LangGraph agent with parallel-capable tools"""
        self.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Initialize Claude 4 Sonnet model with custom retry configuration
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
            region_name='us-west-2',
            config=retry_config
        )
        
        self.model = ChatBedrock(
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            client=bedrock_client
        )
        
        # Define tools
        self.tools = [get_weather, get_stock_price, calculate_distance, search_restaurants]
        
        # Bind tools to model
        self.model_with_tools = self.model.bind_tools(self.tools)
        
        # Create ToolNode for parallel execution
        self.tool_node = ToolNode(self.tools)
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info(f"🚀 Starting conversation {self.conversation_id}")
        logger.info("🚀 LangGraph Parallel Agent initialized with Claude 4 Sonnet")
        logger.info("📋 Available tools: weather, stock prices, distance calculation, restaurant search")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with parallel tool execution"""
        
        def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
            """Determine if we should continue to tools or end"""
            messages = state["messages"]
            last_message = messages[-1]
            
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                logger.info(f"⚡ [PARALLEL EXECUTION] Detected {len(last_message.tool_calls)} tool calls")
                logger.info(f"🔄 [WORKFLOW] Routing to tools node for parallel execution")
                for i, tool_call in enumerate(last_message.tool_calls, 1):
                    logger.info(f"🔧 [TOOL EXECUTION] Tool {i}/{len(last_message.tool_calls)}: {tool_call['name']} with input: {tool_call['args']}")
                return "tools"
            else:
                logger.info(f"🔄 [WORKFLOW] No tool calls detected, ending workflow")
                return "__end__"
        
        def call_model(state: MessagesState):
            """Call the model with tools"""
            messages = state["messages"]
            logger.info(f"📤 [LLM REQUEST] Sending {len(messages)} messages to Claude")
            logger.info(f"📤 [LLM REQUEST] Full messages: {json.dumps([msg.dict() if hasattr(msg, 'dict') else str(msg) for msg in messages], indent=2)}")
            
            start_time = time.time()
            response = self.model_with_tools.invoke(messages)
            execution_time = time.time() - start_time
            
            logger.info(f"📩 [LLM RESPONSE] Received response in {execution_time:.2f}s")
            logger.info(f"📩 [LLM RESPONSE] Full response: {response.dict() if hasattr(response, 'dict') else str(response)}")
            logger.info(f"📩 [LLM RESPONSE] Content: {response.content[:200]}...")
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                logger.info(f"📩 [LLM RESPONSE] Tool calls requested: {len(response.tool_calls)}")
                for i, tool_call in enumerate(response.tool_calls, 1):
                    logger.info(f"📩 [LLM RESPONSE] Tool call {i}: {tool_call}")
            
            return {"messages": [response]}
        
        def execute_tools(state: MessagesState):
            """Execute tools in parallel using ToolNode"""
            messages = state["messages"]
            last_message = messages[-1]
            
            logger.info(f"⚡ [PARALLEL EXECUTION] Starting parallel execution of {len(last_message.tool_calls)} tools")
            
            # Log individual tool executions
            for i, tool_call in enumerate(last_message.tool_calls, 1):
                logger.info(f"🔧 [TOOL EXECUTION] Tool {i}/{len(last_message.tool_calls)}: {tool_call['name']} with input: {tool_call['args']}")
            
            start_time = time.time()
            
            # ToolNode automatically executes tools in parallel
            result = self.tool_node.invoke(state)
            
            execution_time = time.time() - start_time
            logger.info(f"✅ [PARALLEL EXECUTION] Completed in {execution_time:.2f}s")
            
            # Log individual tool results with full details
            tool_results = []
            for i, tool_message in enumerate(result["messages"], 1):
                logger.info(f"✅ [TOOL RESULT] Tool {i}: {tool_message.name} completed successfully")
                logger.info(f"✅ [TOOL RESULT] Full result: {tool_message.content}")
                logger.info(f"✅ [TOOL RESULT] Tool call ID: {tool_message.tool_call_id}")
                
                tool_results.append({
                    'name': tool_message.name,
                    'result': tool_message.content,
                    'tool_call_id': tool_message.tool_call_id,
                    'status': 'success'
                })
            
            # Log aggregated results
            logger.info(f"📊 [AGGREGATED RESULTS] Total tools executed: {len(tool_results)}")
            for i, tool_result in enumerate(tool_results, 1):
                logger.info(f"📊 [AGGREGATED RESULTS] Tool {i}: {tool_result}")
            
            return result
        
        # Build the graph
        workflow = StateGraph(MessagesState)
        
        # Add nodes
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", execute_tools)
        
        # Add edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def run_query(self, query: str) -> str:
        """Execute a query using the LangGraph agent with automatic parallel tool execution"""
        logger.info(f"👤 [CONVERSATION {self.conversation_id}] User query: {query}")
        logger.info(f"📤 [AGENT REQUEST] Sending to agent: {query}")
        start_time = time.time()
        
        try:
            # LangGraph automatically handles parallel tool execution via ToolNode
            logger.info("📤 Sending request to LangGraph agent...")
            result = self.graph.invoke({"messages": [{"role": "user", "content": query}]})
            
            execution_time = time.time() - start_time
            logger.info(f"✅ Query completed in {execution_time:.2f} seconds")
            
            # Get the final response
            final_message = result["messages"][-1]
            response_text = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            logger.info(f"📩 [AGENT RESPONSE] Full response from agent: {response_text}")
            logger.info(f"📝 Final response length: {len(response_text)} characters")
            logger.info(f"📝 Final response preview: {response_text[:200]}...")
            
            # Log conversation flow
            logger.info(f"💬 [CONVERSATION FLOW] Total messages in conversation: {len(result['messages'])}")
            for i, msg in enumerate(result["messages"]):
                msg_type = type(msg).__name__
                content_preview = str(msg)[:100] if hasattr(msg, '__str__') else "No content"
                logger.info(f"💬 [CONVERSATION FLOW] Message {i+1} ({msg_type}): {content_preview}...")
            
            return response_text
            
        except Exception as e:
            logger.error(f"❌ Error executing query: {str(e)}")
            logger.error(f"❌ Error details: {type(e).__name__}: {str(e)}")
            return f"Error: {str(e)}"

def main():
    """Run demo queries to showcase parallel tool execution"""
    agent = ParallelLangGraphAgent()
    
    # Demo queries that will trigger parallel tool execution
    demo_queries = [
        "What's the weather in New York and the stock price of AAPL?",
        "I'm planning a trip from Boston to Miami. Get the weather for both cities, calculate the distance, and find Italian restaurants in Miami.",
        "Show me stock prices for GOOGL, TSLA, and AAPL, plus weather in San Francisco.",
        "Get weather for Seattle, calculate distance from Seattle to Portland, and find Asian restaurants in Portland."
    ]
    
    print("🎯 LangGraph Parallel Tool Execution Demo")
    print("=" * 50)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n📋 Demo Query {i}:")
        print(f"❓ {query}")
        print("-" * 40)
        
        response = agent.run_query(query)
        print(f"🤖 Response:\n{response}")
        print("=" * 50)

if __name__ == "__main__":
    main()
