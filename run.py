"""
Main entry point that recreates the chat interface functionality.
"""

import asyncio
import gradio as gr
from src.components.chat.interface import ChatInterface
from src.services.database import create_async_db_engine, create_async_session_maker
import logging
from datetime import datetime
from typing import Tuple, List
import shutil
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create documents directory
documents_dir = Path("documents")
documents_dir.mkdir(exist_ok=True)

def calculate_elapsed_time(start_time: datetime) -> float:
    """Calculate elapsed time in seconds"""
    return (datetime.now() - start_time).total_seconds()

async def main():
    """Main function to run the chat interface"""
    chat_interface = ChatInterface()
    await chat_interface.initialize()
    
    try:
        with gr.Blocks() as demo:
            gr.Markdown("# Document Q&A System")
            gr.Markdown("Ask questions about your documents. The system will search through the document collection and provide relevant answers.")
            
            # Document Upload Accordion
            with gr.Accordion("Document Upload", open=False):
                upload_button = gr.UploadButton(
                    "Click to Upload Files",
                    variant="huggingface",
                    size="sm",
                    file_types=[".pdf", ".txt", ".doc", ".docx"],
                    file_count="multiple"
                )
                
                with gr.Row():
                    process_btn = gr.Button("Process Documents", variant="primary", size="sm")
                    clear_btn = gr.Button("Clear", variant="stop", size="sm")
                
                progress_box = gr.Textbox(
                    label="Status",
                    placeholder="Upload files and click Process to begin...",
                    interactive=False
                )
            
            # Chat interface
            with gr.Column():
                chat = gr.Chatbot()
                msg = gr.Textbox(label="Message")
                with gr.Row():
                    submit = gr.Button("Submit", variant="huggingface", size="sm")
                    clear = gr.Button("Clear", variant="stop", size="sm")

                # Add example questions
                gr.Examples(
                    examples=[
                        "What are the main topics covered in the documents?",
                        "Can you summarize the key points about machine learning?",
                        "What are the best practices mentioned in the documents?"
                    ],
                    inputs=msg
                )

            # Vector Visualization Accordion
            with gr.Accordion("Vector Store Visualization", open=False):
                with gr.Row():
                    visualize_btn = gr.Button("Generate Visualization", variant="huggingface", size="sm")
                    plot_status = gr.Textbox(
                        label="Visualization Status",
                        placeholder="Click Generate Visualization to begin...",
                        interactive=False
                    )
                
                with gr.Column(elem_classes="center-plot"):
                    plot_output = gr.Plot(
                        label="Vector Store Visualization",
                        show_label=True,
                        container=True,
                        elem_classes="plot-container"
                    )

            # Add custom CSS
            gr.Markdown("""
                <style>
                    .center-plot {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        width: 100%;
                        min-height: 700px;
                    }
                    .plot-container {
                        width: 100%;
                        max-width: 1200px;
                        margin: 0 auto;
                    }
                    .plot-container > div {
                        display: flex;
                        justify-content: center;
                        width: 100%;
                    }
                    .js-plotly-plot {
                        margin: 0 auto !important;
                    }
                    @media (max-width: 768px) {
                        .plot-container {
                            max-width: 100%;
                            padding: 10px;
                        }
                    }
                </style>
            """)

            # Chat response handler
            async def chat_response(message, history):
                try:
                    # Get response from chat interface
                    response = await chat_interface.chat(message, history)
                    
                    # Format the response for the chatbot
                    new_history = history + [(message, response['content'])] if history else [(message, response['content'])]
                    
                    # Return both the updated history and empty message
                    return new_history, ""
                except Exception as e:
                    print(f"Error in chat response: {str(e)}")
                    return history + [(message, f"Error: {str(e)}")], ""

            # Connect the components
            submit.click(
                chat_response,  # Use the new chat_response handler
                inputs=[msg, chat],
                outputs=[chat, msg],
            )
            
            clear.click(lambda: None, None, chat)
            
            # Connect file upload components
            async def handle_file_upload(files: List[str]) -> str:
                """Handle file upload with duplicate checking"""
                try:
                    logger.info(f"Received {len(files)} files")
                    
                    successful_uploads = []
                    skipped_uploads = []
                    
                    for file_path in files:
                        try:
                            source_path = Path(file_path)
                            
                            # Check for duplicate before moving
                            existing_doc = await chat_interface.processor.check_duplicate_document(str(source_path))
                            if existing_doc:
                                logger.info(f"Skipping duplicate file: {source_path.name}")
                                skipped_uploads.append((source_path.name, "Duplicate content"))
                                continue
                            
                            # Move file if not duplicate
                            dest_path = documents_dir / source_path.name
                            counter = 1
                            while dest_path.exists():
                                stem = source_path.stem
                                suffix = source_path.suffix
                                new_name = f"{stem}_{counter}{suffix}"
                                dest_path = documents_dir / new_name
                                counter += 1
                            
                            shutil.move(str(source_path), str(dest_path))
                            successful_uploads.append(dest_path.name)
                            logger.info(f"Successfully moved {dest_path.name}")
                            
                        except Exception as e:
                            logger.error(f"Error processing {file_path}: {str(e)}")
                            continue

                    # Create status message
                    status_message = []
                    if successful_uploads:
                        status_message.append(f"Successfully uploaded {len(successful_uploads)} files:")
                        status_message.extend(f"- {f}" for f in successful_uploads)
                    
                    if skipped_uploads:
                        status_message.append("\nSkipped files:")
                        status_message.extend(f"- {f} ({reason})" for f, reason in skipped_uploads)
                    
                    if not successful_uploads and not skipped_uploads:
                        return "No files were successfully uploaded"
                        
                    status_message.append("\nClick 'Process Documents' to proceed with document processing.")
                    return "\n".join(status_message)
                    
                except Exception as e:
                    logger.error(f"Upload error: {str(e)}")
                    return f"Error uploading files: {str(e)}"

            upload_button.upload(
                fn=handle_file_upload,
                inputs=upload_button,
                outputs=progress_box,
            )
            
            process_btn.click(
                fn=chat_interface.process_documents,
                inputs=None,
                outputs=progress_box,
            )
            
            clear_btn.click(
                fn=lambda: "Upload files and click Process to begin...",
                inputs=None,
                outputs=progress_box,
            )
            
            # Add visualization handler
            async def handle_visualization():
                """Handle visualization generation with detailed logging"""
                try:
                    return await chat_interface.generate_visualization()
                except Exception as e:
                    error_msg = f"Critical error in visualization process: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    return None, error_msg

            # Update visualization button connection
            visualize_btn.click(
                fn=handle_visualization,
                inputs=None,
                outputs=[plot_output, plot_status],
            )

        await demo.launch(
            server_name="127.0.0.1",
            share=True,
            inbrowser=True
        )
    
    except Exception as e:
        error_msg = f"Application error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise
    finally:
        if chat_interface:
            await chat_interface.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application failed: {str(e)}", exc_info=True) 