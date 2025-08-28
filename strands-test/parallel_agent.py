"""
Strands-based parallel tool execution agent
"""
import logging
import time
import os
from datetime import datetime
from strands import Agent
from tools import get_weather, get_stock_price, calculate_distance, search_restaurants

# Create logs directory
logs_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Configure detailed logging
log_filename = os.path.join(logs_dir, f'strands_parallel_execution_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.log')

# Clear any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Clear existing loggers
for logger_name in list(logging.Logger.manager.loggerDict.keys()):
    if logger_name.startswith('strands') or logger_name.startswith('StrandsParallelAgent'):
        logging.getLogger(logger_name).handlers.clear()

# Configure main logging with debug level for comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("StrandsParallelAgent")

# Enable debug logging for Strands to capture LLM requests/responses
strands_logger = logging.getLogger("strands")
strands_logger.setLevel(logging.DEBUG)
# Don't add separate handler - use the basicConfig handler
strands_logger.propagate = True

class ParallelStrandsAgent:
    def __init__(self):
        """Initialize the Strands agent with parallel-capable tools"""
        self.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Custom callback handler for detailed logging
        self.logged_tools = set()
        self.tool_results = []
        self.tool_start_time = None
        
        def detailed_callback_handler(**kwargs):
            if "data" in kwargs:
                # Log streaming data to file only (reduce console verbosity)
                pass  # Skip streaming data logging to reduce noise
            elif "current_tool_use" in kwargs:
                tool = kwargs["current_tool_use"]
                tool_id = tool.get("toolUseId")
                if tool.get("name") and tool_id and tool_id not in self.logged_tools:
                    # Only log complete tool calls once
                    if tool.get("input") and isinstance(tool["input"], dict):
                        if self.tool_start_time is None:
                            self.tool_start_time = time.time()
                            logger.info("⚡ [PARALLEL EXECUTION] Starting parallel tool execution")
                        logger.info(f"🔧 [TOOL EXECUTION] Using tool: {tool['name']} with input: {tool['input']}")
                        self.logged_tools.add(tool_id)
            elif "tool_result" in kwargs:
                # Log detailed tool execution results
                result = kwargs["tool_result"]
                tool_name = result.get('name', 'unknown')
                tool_content = result.get('content', result)
                logger.info(f"✅ [TOOL RESULT] Tool '{tool_name}' completed successfully")
                logger.info(f"✅ [TOOL RESULT] Full result: {tool_content}")
                
                # Store for aggregation
                self.tool_results.append({
                    'name': tool_name,
                    'result': tool_content,
                    'status': 'success'
                })
                
            elif "tool_error" in kwargs:
                # Log tool execution errors
                error = kwargs["tool_error"]
                tool_name = error.get('name', 'unknown')
                error_msg = error.get('error', error)
                logger.error(f"❌ [TOOL ERROR] Tool '{tool_name}' failed with error: {error_msg}")
                
                # Store for aggregation
                self.tool_results.append({
                    'name': tool_name,
                    'error': error_msg,
                    'status': 'error'
                })
                
            # Log all other events for debugging
            else:
                event_type = list(kwargs.keys())[0] if kwargs else "unknown"
                logger.debug(f"🔍 [DEBUG EVENT] {event_type}: {kwargs}")
                
                # Specifically look for LLM request/response patterns
                if any(key in str(kwargs) for key in ['request', 'response', 'messages', 'model']):
                    logger.info(f"🤖 [LLM EVENT] {event_type}: {kwargs}")
        
        # Use Claude 4 Sonnet for parallel tool execution support
        self.agent = Agent(
            max_parallel_tools = 5,
            model="us.anthropic.claude-sonnet-4-20250514-v1:0",
            tools=[get_weather, get_stock_price, calculate_distance, search_restaurants],
            system_prompt="""You are a helpful assistant that can execute multiple tools simultaneously.
            When a user asks for information that requires multiple tools, use them all at once for faster responses.
            Always provide comprehensive answers based on all tool results.""",
            callback_handler=detailed_callback_handler
        )
        
        logger.info(f"🚀 Starting conversation {self.conversation_id}")
        logger.info("🚀 Strands Parallel Agent initialized with Claude 4 Sonnet")
        logger.info("📋 Available tools: weather, stock prices, distance calculation, restaurant search")
    
    def run_query(self, query: str) -> str:
        """Execute a query using the Strands agent with automatic parallel tool execution"""
        logger.info(f"👤 [CONVERSATION {self.conversation_id}] User query: {query}")
        logger.info(f"📤 [AGENT REQUEST] Sending to agent: {query}")
        start_time = time.time()
        
        # Track tool executions for aggregation logging
        self.tool_results = []
        self.tool_start_time = None
        
        try:
            # Strands automatically handles parallel tool execution when Claude requests multiple tools
            logger.info("📤 Sending request to Strands agent...")
            result = self.agent(query)
            
            execution_time = time.time() - start_time
            logger.info(f"✅ Query completed in {execution_time:.2f} seconds")
            
            response_text = str(result.message) if result.message else "No response"
            logger.info(f"📩 [AGENT RESPONSE] Full response from agent: {response_text}")
            logger.info(f"📝 Final response length: {len(response_text)} characters")
            logger.info(f"📝 Final response preview: {response_text[:200]}...")
            
            # Log aggregated tool results if any tools were used
            if self.tool_results:
                logger.info(f"📊 [AGGREGATED RESULTS] Total tools executed: {len(self.tool_results)}")
                for i, tool_result in enumerate(self.tool_results, 1):
                    logger.info(f"📊 [AGGREGATED RESULTS] Tool {i}: {tool_result}")
            
            return response_text
            
        except Exception as e:
            logger.error(f"❌ Error executing query: {str(e)}")
            return f"Error: {str(e)}"

def main():
    """Run demo queries to showcase parallel tool execution"""
    agent = ParallelStrandsAgent()
    
    # Demo queries that will trigger parallel tool execution
    demo_queries = [
        "What's the weather in New York and the stock price of AAPL?",
        "I'm planning a trip from Boston to Miami. Get the weather for both cities, calculate the distance, and find Italian restaurants in Miami.",
        "Show me stock prices for GOOGL, TSLA, and AAPL, plus weather in San Francisco.",
        "Get weather for Seattle, calculate distance from Seattle to Portland, and find Asian restaurants in Portland."
    ]
    
    print("🎯 Strands Parallel Tool Execution Demo")
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
