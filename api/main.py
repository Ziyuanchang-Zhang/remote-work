import uvicorn
import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from api.logging_config import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Add the current directory to the path so we can import the api package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for required environment variables
required_env_vars = ['GOOGLE_API_KEY', 'OPENAI_API_KEY']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
    logger.warning("Some functionality may not work correctly without these variables.")

# Configure Google Generative AI
import google.generativeai as genai
from api.config import GOOGLE_API_KEY

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    logger.warning("GOOGLE_API_KEY not configured")

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8001))

    # Import the app here to ensure environment variables are set first
    from api.api import app

    logger.info(f"Starting Streaming API on port {port}")

    # Run the FastAPI app with uvicorn
    # Disable reload in production/Docker environment
    is_development = os.environ.get("NODE_ENV") != "production"
    
    if is_development:
        # Prevent infinite logging loop caused by file changes triggering log writes
        logging.getLogger("watchfiles.main").setLevel(logging.WARNING)

    app_import_string = "api.api:app"

    # uvicorn_config = {
    #     "app": app_import_string,
    #     "host": "0.0.0.0",
    #     "port": port,
    #     "reload": is_development,
    #     "workers": 8 if is_development else 88,  
    #     "loop": "asyncio",
    #     "timeout_keep_alive": 9999,
    #     "http": "httptools",
    #     "backlog": 4096,  
    #     "limit_concurrency": 2000, 
    #     "log_level": "info" if is_development else "warning",
    # }

    # uvicorn.run(**uvicorn_config)
    uvicorn.run(
        "api.api:app",
        host = "0.0.0.0",
        port = port,
        reload = is_development
    )
