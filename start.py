"""
Loqo AI Agent v3 — Quick Start Script
Run this to set up and launch the application.

Usage:
    py -3.11 start.py
"""
import subprocess
import sys
import os

def main():
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("\n" + "=" * 50)
    print("  Loqo AI Agent v3 — Setup & Launch")
    print("=" * 50)
    print(f"  Python: {sys.version.split()[0]}")
    
    # Check for .env file
    if not os.path.exists(".env"):
        print("\n⚠ No .env file found. Creating template...")
        with open(".env", "w") as f:
            f.write("GEMINI_API_KEY=your_api_key_here\n")
        print("  → Created .env — please add your GEMINI_API_KEY")
        print("  → Get one free at: https://aistudio.google.com/apikey")
    
    # Check if GEMINI_API_KEY is set
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        print("\n❌ GEMINI_API_KEY not configured!")
        print("   Edit .env and add your API key, then run again.")
        print("   Get one free at: https://aistudio.google.com/apikey")
        sys.exit(1)
    
    # Use PORT from env (Railway/Render inject $PORT), fallback to 8001
    port = int(os.getenv("PORT", 8001))
    # Disable reload in production (Railway/Render)
    is_production = bool(os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RENDER"))
    
    print(f"\n✓ GEMINI_API_KEY configured")
    print(f"✓ Starting server on http://localhost:{port}")
    if is_production:
        print("  (production mode — reload disabled)")
    print("\n" + "-" * 50)
    print(f"  Open your browser to: http://localhost:{port}")
    print("-" * 50 + "\n")
    
    # Run the FastAPI server
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=not is_production)

if __name__ == "__main__":
    main()
