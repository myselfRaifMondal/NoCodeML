"""
NoCodeML Backend Server

This is the main FastAPI application that serves as the backend for NoCodeML.
It provides REST API endpoints for dataset upload, analysis, model training, and evaluation.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from pathlib import Path
import sys
import os

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backend.api.routes import dataset, model, analysis
from backend.models.schemas import DatasetInfo, ModelConfig, TrainingStatus

# Initialize FastAPI app
app = FastAPI(
    title="NoCodeML API",
    description="No-Code Machine Learning Platform API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(dataset.router, prefix="/api/dataset", tags=["Dataset"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(model.router, prefix="/api/model", tags=["Model"])

# Serve static files (for frontend in production)
if os.path.exists("frontend/dist"):
    app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - serves the frontend or API info"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NoCodeML</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                   margin: 0; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; min-height: 100vh; }
            .container { max-width: 800px; margin: 0 auto; text-align: center; }
            h1 { font-size: 3em; margin-bottom: 20px; }
            p { font-size: 1.2em; margin-bottom: 30px; opacity: 0.9; }
            .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                       gap: 20px; margin: 40px 0; }
            .feature { background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; }
            a { color: #ffd700; text-decoration: none; font-weight: bold; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 NoCodeML</h1>
            <p>No-Code Machine Learning Platform</p>
            <p>Democratizing AI/ML model building for everyone!</p>
            
            <div class="features">
                <div class="feature">
                    <h3>🔍 Dataset Analysis</h3>
                    <p>Intelligent data profiling and quality assessment</p>
                </div>
                <div class="feature">
                    <h3>🤖 AutoML Pipeline</h3>
                    <p>Automated model selection and optimization</p>
                </div>
                <div class="feature">
                    <h3>📊 Multi-Algorithm</h3>
                    <p>Classical ML, Deep Learning, and more</p>
                </div>
            </div>
            
            <p>
                <a href="/api/docs">📚 API Documentation</a> | 
                <a href="/api/redoc">📖 ReDoc</a>
            </p>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "NoCodeML API"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )

