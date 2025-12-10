import os
import shutil
import chainlit as cl
from dotenv import load_dotenv
from engineio.payload import Payload

from components.embeddings_llm.gemini_embedding_model import GeminiEmbeddingModel
from components.db.chroma_db import ChromaDb
from components.hierarchy_late_chunk import HierarchyLateChunk
from components.llm.gemini_llm import GeminiLLM

# Fix for "Too many packets in payload" error with engineio > 4.9.0
Payload.max_decode_packets = 2000

# Load environment variables
load_dotenv()

# Constants
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
CHROMA_HOST = os.environ.get("CHROMA_HOST")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", 8000))
CHROMA_TOKEN = os.environ.get("CHROMA_TOKEN")
CHROMA_SSL = os.environ.get("CHROMA_SSL", "False").lower() == "true"

def init_pipeline():
    """Initializes the HierarchyLateChunk pipeline."""
    try:
        emb = GeminiEmbeddingModel(api_key=GOOGLE_API_KEY)
        llm = GeminiLLM(api_key=GOOGLE_API_KEY)

        chroma_headers = None
        if CHROMA_TOKEN:
            chroma_headers = {"X-Chroma-Token": CHROMA_TOKEN}

        try:
            vdb = ChromaDb(
                persist_directory="./chroma_store",
                host=CHROMA_HOST,
                port=CHROMA_PORT,
                ssl=CHROMA_SSL,
                headers=chroma_headers
            )
            vdb.client.heartbeat()
        except Exception as e:
            print(f"Could not connect to remote ChromaDB ({e}). Falling back to local mode.")
            vdb = ChromaDb(
                persist_directory="./chroma_store",
                host=None,
                port=8000,
                ssl=False,
                headers=None
            )
        
        return HierarchyLateChunk(llm=llm, embedding_model=emb, vectordb=vdb)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return None

@cl.on_chat_start
async def start():
    """Called when the chat starts."""
    msg = cl.Message(content="Initializing pipeline...", author="System")
    await msg.send()
    
    pipeline = await cl.make_async(init_pipeline)()
    
    if pipeline:
        cl.user_session.set("pipeline", pipeline)
        # Initialize uploaded doc_ids list
        # Initialize uploaded doc_ids list with all available docs
        all_docs = await cl.make_async(pipeline.list_documents)()
        cl.user_session.set("doc_ids", all_docs)
        
        msg.content = "Pipeline initialized. You can upload a PDF or ask questions."
        await msg.update()
        
        await update_doc_settings()
    else:
        msg.content = "Failed to initialize pipeline. Please check logs and API keys."
        await msg.update()

async def update_doc_settings():
    pipeline = cl.user_session.get("pipeline")
    if not pipeline:
        return
    
    all_docs = await cl.make_async(pipeline.list_documents)()
    all_docs.sort()
    
    current_ids = cl.user_session.get("doc_ids", [])
    
    # Filter current_ids to ensure they exist in all_docs (cleanup)
    current_ids = [d for d in current_ids if d in all_docs]
    
    if not all_docs:
        await cl.ChatSettings([]).send()
        return

    await cl.ChatSettings(
        [
            cl.input_widget.MultiSelect(
                id="doc_selection",
                label="Selected Documents",
                initial=current_ids,
                items={d: d for d in all_docs},
                description="Select which documents to include in the context."
            )
        ]
    ).send()

@cl.on_settings_update
async def on_settings_update(settings):
    doc_ids = settings.get("doc_selection", [])
    cl.user_session.set("doc_ids", doc_ids)
    await cl.Message(content=f"Context updated. Using {len(doc_ids)} documents.", author="System").send()

@cl.on_message
async def main(message: cl.Message):
    pipeline = cl.user_session.get("pipeline")
    if not pipeline:
        await cl.Message(content="Pipeline not initialized.", author="System").send()
        return

    # Handle File Uploads
    if message.elements:
        processing_msg = cl.Message(content="Processing uploaded files...", author="System")
        await processing_msg.send()
        
        for element in message.elements:
            # Check for PDF or text
            target_path = os.path.join(UPLOAD_DIR, element.name)
            shutil.copy(element.path, target_path)
            
            # Check if document already exists
            doc_id = element.name
            try:
                # Use a dummy embedding to search (we only care about metadata filter)
                dummy_emb = [0.0] * 768 # Assuming 768 dims, or use embedding model
                # Correct way: Use the embedding model to be safe about dims
                # blocking call
                dummy_emb = await cl.make_async(pipeline.embedding_model.embed_text)("test")
                
                existing_docs = await cl.make_async(pipeline.vectordb.query_by_embedding)(
                    collection=pipeline.sections_collection,
                    query_embedding=dummy_emb,
                    n_results=1,
                    where={"doc_id": doc_id}
                )
                
                doc_exists = False
                if existing_docs and existing_docs.get("ids") and existing_docs["ids"][0]:
                     doc_exists = True
                
                if doc_exists:
                    await cl.Message(content=f"Document {element.name} already exists. Skipping ingestion.", author="System").send()
                    
                    # Even if it exists, we must add it to the current CLEAN session so we can query it
                    current_doc_ids = cl.user_session.get("doc_ids", [])
                    if doc_id not in current_doc_ids:
                         current_doc_ids.append(doc_id)
                         current_doc_ids.append(doc_id)
                         cl.user_session.set("doc_ids", current_doc_ids)
                    
                    await update_doc_settings()
                         
                    # We continue to next file or return
                    continue
            except Exception as e:
                print(f"Error checking for existence: {e}")
                # Proceed to ingest if check fails
            
            await cl.Message(content=f"Ingesting {element.name}...", author="System").send()
            
            # Ingest
            try:
                # We overwrite the status callback for this session/request
                # But since ingest_from_file doesn't use callbacks much explicitly as run(),
                # We just await it.
                await cl.make_async(pipeline.ingest_from_file)(target_path, doc_id=doc_id)
                # Track this doc_id
                current_doc_ids = cl.user_session.get("doc_ids", [])
                if doc_id not in current_doc_ids:
                    current_doc_ids.append(doc_id)
                    current_doc_ids.append(doc_id)
                    cl.user_session.set("doc_ids", current_doc_ids)
                
                await update_doc_settings()

                await cl.Message(content=f"Successfully ingested {element.name}.", author="System").send()
            except Exception as e:
                await cl.Message(content=f"Error ingesting {element.name}: {e}", author="System").send()
        
        await processing_msg.remove()
        
        # If there is also text in the message, we can choose to process it as a query or ignore.
        # Usually checking message.content
        if not message.content:
            return

    if not message.content:
        # Just an upload
        return

    # Handle Query
    query = message.content
    
    # Callback to stream thoughts to UI
    # We define a sync function that calls async code via cl.run_sync
    def sync_status_callback(msg_text):
        cl.run_sync(cl.Message(content=msg_text, author="Thought").send())

    pipeline.status_callback = sync_status_callback
    
    # Run pipeline
    # We use make_async to run the blocking pipeline.run in a thread
    async_run = cl.make_async(pipeline.run)
    
    try:
        # Provide visual feedback that we are working
        async with cl.Step(name="RAG Pipeline") as step:
            step.input = query
            
            # Retrieve scoped doc_ids
            doc_ids = cl.user_session.get("doc_ids", [])
            result_state = await async_run(query, doc_ids=doc_ids)
            
            final_answer = result_state.get('final_answer', "No answer generated.")
            
            step.output = final_answer

        # Send the final answer as a normal message
        await cl.Message(content=final_answer).send()
        
    except Exception as e:
        await cl.Message(content=f"Error occurred: {e}", author="System").send()
