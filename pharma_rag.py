
import os
import sys
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import warnings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PharmaRAG")

# -----------------------------------------------------------------------------
# Imports - Docling
# -----------------------------------------------------------------------------
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions, 
        TableStructureOptions, 
        AcceleratorOptions, 
        AcceleratorDevice
    )
    from docling.chunking import HybridChunker
except ImportError as e:
    logger.error(f"Failed to import Docling dependencies: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Imports - LangChain / Embeddings / VectorStore
# -----------------------------------------------------------------------------
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_groq import ChatGroq
    from langchain_core.documents import Document as LCDocument
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
except ImportError as e:
    logger.error(f"Failed to import LangChain/VectorStore dependencies: {e}")
    sys.exit(1)


# -----------------------------------------------------------------------------
# 1. Ingestion Engine
# -----------------------------------------------------------------------------
class IngestionEngine:
    """
    Handles extracting unstructured data (Text, Tables, Formulas, Images) 
    from PDFs using Docling and converting them to Markdown.
    """
    def __init__(self):
        # Configure pipeline options for better extraction
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # Enable OCR for scanned images
        pipeline_options.do_table_structure = True # Enable table structure extraction
        pipeline_options.generate_picture_images = True # Enable image extraction

        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4, device= AcceleratorDevice.AUTO
        )
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def process_file(self, file_path: str) -> Any:
        """
        Processes a single PDF file and returns the Docling document object.
        Also saves extracted images to 'data/extracted_images/<filename>'.
        """
        logger.info(f"Processing file: {file_path}")
        try:
            conversion_result = self.converter.convert(file_path)
            doc = conversion_result.document
            
            # Create directory for images
            filename_stem = Path(file_path).stem
            output_dir = Path(f"data/extracted_images/{filename_stem}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save images and update markdown
            # Note: Docling's export_to_markdown doesn't automatically link all saved images easily
            # We will manually save them to ensure they exist for the frontend
            
            # Simple approach: Identify pictures and save provided PIL images
            table_counter = 0
            picture_counter = 0
            
            # Check for pictures in the document structure
            # Depending on docling version, we might iterate differently.
            # Here we iterate over document images if exposed, strictly saving them.
            
            # Using basic approach: Export markdown with image placeholders is default.
            # We need to ensure the images exist at reference paths if Docling generated them.
            # Since Docling in-memory keeps them as PIL, we iterate and save.
            
            # NOTE: For RAG context, we want to know WHICH image is WHERE.
            # Docling insert '![Image](image_path)' in markdown.
            # We will rely on Docling's internal image generation for export if available,
            # or simply save valid pictures found in structure.
            
            # Populate image_map
            image_map = {} 

            for i, element in enumerate(doc.pictures):
                image = element.get_image(doc)
                if image:
                    image_filename = f"picture_{i}.png"
                    image_path = output_dir / image_filename
                    image.save(image_path)
                    logger.info(f"Saved image: {image_path}")
                    
                    # Store mapping: self_ref -> full path (or relative path for frontend)
                    # We store the relative path from the app root which the frontend can use
                    # Or just the filename if we reconstruct path later.
                    # Frontend expects keys relative to data/extracted_images/<stem>/
                    # Let's store the full relative path "data/extracted_images/<stem>/picture_{i}.png"
                    image_map[element.self_ref] = str(image_path).replace("\\", "/")
            
            # We re-export markdown but currently Docling might not autolink OUR saved paths.
            # For this 'Best Option', we want to make sure the markdown contains
            # references to these images so the chunker picks them up.
            
            # To do this robustly with Docling's current API can be tricky without a custom Backend.
            # For now, we will save the images so we have them. 
            # The Text-to-Image link in RAG is usually implicit (text mentions "Figure 1").
            
            markdown_content = doc.export_to_markdown()
            
            logger.info("Successfully converted document to Markdown.")
            # Refactored to return image_map instead of attaching it to doc
            return doc, markdown_content, image_map
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise


# -----------------------------------------------------------------------------
# 2. Chunking Engine
# -----------------------------------------------------------------------------
class ChunkingEngine:
    """
    Handles chunking of the Docling document using Docling's HybridChunker.
    """
    def __init__(self, tokenizer_model: str = "intfloat/multilingual-e5-large"):
        # We can use the same tokenizer as our embedding model for better alignment
        self.chunker = HybridChunker(
            tokenizer=tokenizer_model
        )

    def chunk_document(self, doc_object: Any, image_map: Optional[Dict[str, str]] = None) -> List[LCDocument]:
        """
        Chunks the Docling document object and converts chunks to LangChain Documents.
        """
        if image_map is None:
             image_map = {}

        logger.info("Chunking document...")
        chunk_iter = self.chunker.chunk(dl_doc=doc_object)
        
        lc_docs = []
        for i, chunk in enumerate(chunk_iter):
            # Serialize content to markdown for the embedding context
            page_content = self.chunker.serialize(chunk=chunk)
            
            # Extract metadata
            page_numbers = set() # Use set for unique pages
            chunk_images = []
            
            for item in chunk.meta.doc_items:
                if hasattr(item, 'page_no'):
                    page_numbers.add(item.page_no)
                elif hasattr(item, 'prov') and hasattr(item.prov, 'page_no'):
                    page_numbers.add(item.prov.page_no)
                
                # Check for images using image_map
                if hasattr(item, 'self_ref') and item.self_ref in image_map:
                    chunk_images.append(image_map[item.self_ref])

            # Sanitize metadata for ChromaDB (must be simple types: str, int, float, bool)
            origin_info = getattr(doc_object, "origin", "unknown")
            source_str = str(origin_info) if origin_info else "unknown"
            
            # If origin has a filename, use that for cleaner metadata
            if hasattr(origin_info, 'filename'):
                source_str = origin_info.filename

            metadata = {
                "source": str(source_str),
                "page_numbers": ",".join(map(str, sorted(list(page_numbers)))), # Convert list to string
                "images": ",".join(chunk_images), # Comma separated list of images
                "chunk_index": i
            }
            
            # Create LangChain Document
            lc_docs.append(LCDocument(page_content=page_content, metadata=metadata))
            
        logger.info(f"Generated {len(lc_docs)} chunks.")
        return lc_docs


# -----------------------------------------------------------------------------
# 3. Embedding Engine & Vector Store
# -----------------------------------------------------------------------------
class VectorDatabase:
    """
    Manages embedding generation (HuggingFace) and storage (ChromaDB).
    """
    def __init__(self, 
                 embedding_model_name: str = "intfloat/multilingual-e5-large",
                 persist_directory: str = "./chroma_db"):
        
        logger.info(f"Initializing Embedding Model: {embedding_model_name}")
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'}, # Use 'cuda' if GPU is available
            encode_kwargs={'normalize_embeddings': True}
        )
        self.persist_directory = persist_directory
        self.vector_store = None

    def create_or_update_vector_store(self, documents: List[LCDocument], collection_name: str = "pharma_data"):
        """
        Creates or updates the ChromaDB vector store with new documents.
        """
        logger.info("Creating/Updating Vector Store...")
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            collection_name=collection_name,
            persist_directory=self.persist_directory
        )
        logger.info(f"Vector store persisted at {self.persist_directory}")
        return self.vector_store

    def get_retriever(self, collection_name: str = "pharma_data", k: int = 5):
        """
        Returns a retriever object from the existing vector store.
        """
        if not self.vector_store:
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory
            )
        
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    def list_ingested_files(self) -> List[str]:
        """
        Returns a list of unique filenames (source) currently in the vector store.
        """
        if not self.vector_store:
             self.vector_store = Chroma(
                collection_name="pharma_data",
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory
            )
        
        try:
             # Get all metadata
             data = self.vector_store.get()
             if not data or 'metadatas' not in data:
                 return []
             
             # Extract unique sources
             sources = set()
             for meta in data['metadatas']:
                 if meta and 'source' in meta:
                     sources.add(meta['source'])
             
             return sorted(list(sources))
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []

    def delete_file(self, filename: str) -> bool:
        """
        Deletes all chunks associated with a specific filename.
        """
        if not self.vector_store:
             self.vector_store = Chroma(
                collection_name="pharma_data",
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory
            )
            
        try:
            logger.info(f"Deleting file: {filename}")
            self.vector_store.delete(where={"source": filename})
            logger.info("Deletion successful.")
            return True
        except Exception as e:
            logger.error(f"Error deleting file {filename}: {e}")
            return False
    
    def list_ingested_files(self) -> List[str]:
        """
        Returns a list of unique filenames (source) currently in the vector store.
        """
        if not self.vector_store:
             self.vector_store = Chroma(
                collection_name="pharma_data",
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory
            )
        
        try:
             # Get all metadata
             data = self.vector_store.get()
             if not data or 'metadatas' not in data:
                 return []
             
             # Extract unique sources
             sources = set()
             for meta in data['metadatas']:
                 if meta and 'source' in meta:
                     sources.add(meta['source'])
             
             return sorted(list(sources))
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []

    def delete_file(self, filename: str) -> bool:
        """
        Deletes all chunks associated with a specific filename.
        """
        if not self.vector_store:
             self.vector_store = Chroma(
                collection_name="pharma_data",
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory
            )
            
        try:
            logger.info(f"Deleting file: {filename}")
            self.vector_store.delete(where={"source": filename})
            logger.info("Deletion successful.")
            return True
        except Exception as e:
            logger.error(f"Error deleting file {filename}: {e}")
            return False


# -----------------------------------------------------------------------------
# 4. RAG Controller (Retrieval & Generation)
# -----------------------------------------------------------------------------
class RAGController:
    """
    Orchestrates the RAG process: Retrieval -> Groq Generation.
    """
    def __init__(self, vector_db: VectorDatabase, groq_api_key: str):
        self.vector_db = vector_db
        if not groq_api_key:
            raise ValueError("Groq API Key is required.")
        
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile", # Using a strong model for complex data
            temperature=0
        )
        
        self.retriever = self.vector_db.get_retriever()
        
        # Define Prompt Template
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are an expert Pharmaceutical Data Analyst. 
            Use the following pieces of retrieved context to answer the question detailed below.
            The context may contain Markdown tables, text, and descriptions of images.
            
            Context:
            {context}
            
            Question:
            {question}
            
            Answer the question comprehensively. If the answer is not in the context, say you don't know.
            Ensure you capture numeric data from tables accurately.
            """
        )

    def query(self, question: str) -> Dict[str, Any]:
        """
        Executes the RAG pipeline for a given question.
        Returns a dict with 'answer' and 'context'.
        """
        logger.info(f"Querying: {question}")
        
        # Retrieval
        docs = self.retriever.invoke(question)
        context_str = "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        response = rag_chain.invoke({"context": context_str, "question": question})
        
        return {
            "answer": response,
            "context": docs
        }

    def grade_documents(self, documents: List[LCDocument], question: str) -> float:
        """
        Grades the relevance of the retrieved documents to the question.
        Returns a score between 0.0 and 1.0.
        """
        logger.info("Grading documents...")
        grader_prompt = ChatPromptTemplate.from_template(
            """
            You are a grader assessing relevance of a retrieved document to a user question.
            
            Retrieved Document:
            {context}
            
            User Question:
            {question}
            
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as RELEVANT.
            Give a binary score 'YES' or 'NO'. Do not provide explanation.
            """
        )
        
        grader_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | grader_prompt
            | self.llm
            | StrOutputParser()
        )
        
        relevant_count = 0
        for doc in documents:
            score = grader_chain.invoke({"context": doc.page_content, "question": question})
            if "YES" in score.upper():
                relevant_count += 1
                
        relevance_score = relevant_count / len(documents) if documents else 0.0
        logger.info(f"Relevance Score: {relevance_score}")
        return relevance_score

    def transform_query(self, question: str) -> str:
        """
        Rewrites the query to be better optimized for retrieval.
        """
        logger.info("Rewriting query...")
        rewrite_prompt = ChatPromptTemplate.from_template(
            """
            You are a question re-writer that converts an input question to a better version that is optimized 
            for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
            
            Input Question: {question}
            
            Output only the improved question. Do not add any preamble.
            """
        )
        
        chain = (
            {"question": RunnablePassthrough()}
            | rewrite_prompt
            | self.llm
            | StrOutputParser()
        )
        
        new_query = chain.invoke({"question": question})
        logger.info(f"New Query: {new_query}")
        return new_query

    def query_corrective_rag(self, question: str):
        """
        Executes Corrective RAG (CRAG) pipeline.
        Yields events: {'type': 'status'|'token'|'context', 'content': ...}
        """
        logger.info(f"Querying (CRAG): {question}")
        
        # 1. Retrieval
        yield {"type": "status", "content": "🔍 Retrieving documents..."}
        docs = self.retriever.invoke(question)
        
        # 2. Grading
        yield {"type": "status", "content": "⚖️ Grading document relevance..."}
        score = self.grade_documents(docs, question)
        
        final_docs = docs
        
        # 3. Correction (if needed)
        # Threshold: 60% relevance
        if score < 0.6:
            yield {"type": "status", "content": f"⚠️ Relevance Low ({score:.0%}). Rewriting query..."}
            new_question = self.transform_query(question)
            
            yield {"type": "status", "content": f"🔄 Retrying with: '{new_question}'"}
            final_docs = self.retriever.invoke(new_question)
        else:
            yield {"type": "status", "content": f"✅ Relevance Good ({score:.0%}). Generating answer..."}
            
        # 4. Final Generation
        context_str = "\n\n".join(doc.page_content for doc in final_docs)
        
        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Emit context for the UI to use later (e.g. for images)
        yield {"type": "context", "content": final_docs}
        
        # Check hallucination / final answer generation
        # NOTE: For speed, we are combining generation and hallucination check implicitly 
        # by creating a strong context. A specialized Hallucination Grader can be added here
        # but requires generation first, then grading.
        # For this implementation, we will stream the generation directly.
        
        stream = rag_chain.stream({"context": context_str, "question": question})
        
        for chunk in stream:
            yield {"type": "token", "content": chunk}

    def query_stream(self, question: str) -> tuple:
        """
        Executes the RAG pipeline returning a stream and context.
        Returns: (generator, context_docs)
        """
        logger.info(f"Querying (Stream): {question}")
        
        # 1. Retrieval
        docs = self.retriever.invoke(question)
        context_str = "\n\n".join(doc.page_content for doc in docs)

        # 2. Chain
        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        # 3. Stream
        stream = rag_chain.stream({"context": context_str, "question": question})
        
        return stream, docs


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main():
    # 1. Setup
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Modal RAG for Pharma Data")
    parser.add_argument("--file", type=str, help="Path to PDF file to ingest")
    parser.add_argument("--query", type=str, help="Question to ask")
    args = parser.parse_args()

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("Please set the GROQ_API_KEY environment variable.")
        return

    # Initialize Components
    ingestion = IngestionEngine()
    chunking = ChunkingEngine()
    vector_db = VectorDatabase()

    # 2. Ingest & Index (if file provided)
    if args.file:
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            return
            
        doc_object, _, image_map = ingestion.process_file(args.file)
        chunks = chunking.chunk_document(doc_object, image_map)
        vector_db.create_or_update_vector_store(chunks)
    
    # 3. Query (if query provided)
    if args.query:
        rag = RAGController(vector_db, groq_api_key)
        answer = rag.query(args.query)
        print("\n\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(answer)
        print("="*80 + "\n")
    
    if not args.file and not args.query:
        print("Usage: python pharma_rag.py --file <path_to_pdf> --query <question>")

if __name__ == "__main__":
    main()
