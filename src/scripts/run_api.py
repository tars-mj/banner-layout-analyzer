import uvicorn
import argparse
from pathlib import Path
import os

def main():
    parser = argparse.ArgumentParser(description='Run Banner Layout Analyzer API')
    parser.add_argument('--host', type=str, default="0.0.0.0",
                      help='Host to run the API on')
    parser.add_argument('--port', type=int, default=int(os.getenv('PORT', '8000')),
                      help='Port to run the API on')
    parser.add_argument('--reload', action='store_true',
                      help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    print("\nBanner Layout Analyzer API")
    print("=========================")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {'Enabled' if args.reload else 'Disabled'}")
    print("=========================\n")
    
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main() 