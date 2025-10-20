import http.server
import socketserver
import os
import json
from dotenv import load_dotenv
from multipart import parse_form_data
from components.embeddings_llm.jina_embedding_model import JinaEmbeddingModel
from components.chroma_db import ChromaDb
from components.hierarchy_late_chunk import HierarchyLateChunk
from components.llm.gemini_llm import GeminiLLM

load_dotenv()
JINA_API_KEY = os.environ.get("JINA_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

PORT = 8000
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), 'frontend')

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Initialize the pipeline
try:
    emb = JinaEmbeddingModel(model_name="jina-embeddings-v2-base-en", api_key=JINA_API_KEY)
    llm = GeminiLLM(api_key=GOOGLE_API_KEY)
    vdb = ChromaDb(persist_directory="./chroma_store")
    pipeline = HierarchyLateChunk(llm=llm, embedding_model=emb, vectordb=vdb)
except (ImportError, ValueError) as e:
    print(f"Error: {e}")
    print("\nPlease ensure all required packages are installed and API keys are set. You can install packages using:\n  uv pip install jina chromadb langgraph docling google-generativeai requests multipart")
    pipeline = None

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def do_POST(self):
        if self.path == '/chat':
            if not pipeline:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'message': 'Pipeline not initialized. Check API keys.'}).encode('utf-8'))
                return

            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)

            answer = pipeline.run(data['message'])
            response = {'message': answer}

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            form, files = parse_form_data(self.headers, self.rfile)

            if 'file' in files:
                file_item = files['file']
                if file_item.filename:
                    file_path = os.path.join(UPLOAD_DIR, file_item.filename)
                    with open(file_path, 'wb') as f:
                        f.write(file_item.file.read())
                    
                    if pipeline:
                        pipeline.ingest_from_file(file_path)

                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'File uploaded successfully')
                else:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'No file selected')
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'No file field in form')

with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()
