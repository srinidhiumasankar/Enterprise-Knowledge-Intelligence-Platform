# app/services/langchain/parent_retriever.py
# ------------------------------------------
# LangChain Parent Document Retriever wrapping hybrid vector search and document expansions.

import logging
import time
from typing import Any, List, Dict, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import Field

try:
    from langchain.retrievers import ParentDocumentRetriever
except ImportError:
    from langchain_classic.retrievers import ParentDocumentRetriever

logger = logging.getLogger(__name__)


class DummyVectorStore(VectorStore):
    """
    Placeholder LangChain VectorStore subclass to satisfy ParentDocumentRetriever's constructor requirements.
    """

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[Any, Any]]] = None, **kwargs: Any) -> List[str]:
        return []

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Any,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        **kwargs: Any,
    ) -> "DummyVectorStore":
        return cls()

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        return []


class CustomParentDocumentRetriever(ParentDocumentRetriever):
    """
    Subclass of LangChain's ParentDocumentRetriever that uses the custom Hybrid Search retriever
    for child chunk matches instead of direct vector store queries.
    """
    hybrid_retriever: Any = Field(description="Underlying hybrid retriever instance")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        start_time = time.time()
        logger.info(f"CustomParentDocumentRetriever searching child chunks for: '{query}'")

        try:
            # 1. Search child chunks using the hybrid retriever
            child_docs = self.hybrid_retriever.invoke(query, run_manager=run_manager.get_child())
            child_search_latency = (time.time() - start_time) * 1000
            logger.info(f"Child search retrieved {len(child_docs)} chunks in {child_search_latency:.2f}ms")

            # 2. Load corresponding parent documents from the docstore
            parent_loading_start = time.time()
            parent_docs = []
            seen_parent_ids = set()

            for doc in child_docs:
                parent_id = doc.metadata.get("parent_document_id")
                if not parent_id:
                    logger.warning(f"No parent_document_id mapping found for chunk ID {doc.metadata.get('chunk_id')}. Returning child.")
                    parent_docs.append(doc)
                    continue

                if parent_id in seen_parent_ids:
                    continue
                seen_parent_ids.add(parent_id)

                logger.info(f"Loading parent document from store: {parent_id}")
                retrieved_parents = self.docstore.mget([parent_id])
                parent_doc = retrieved_parents[0] if retrieved_parents else None

                if parent_doc:
                    # Merge metadata from child to parent to preserve citation keys, page numbers, filenames, etc.
                    meta = parent_doc.metadata.copy()
                    meta.update(doc.metadata)
                    
                    # Retain child chunk details for downstream services (like citation service formatting)
                    meta["child_chunk_text"] = doc.page_content
                    meta["parent_document_id"] = parent_id
                    
                    expanded_doc = Document(page_content=parent_doc.page_content, metadata=meta)
                    parent_docs.append(expanded_doc)
                    logger.info(f"Returned parent document chunk size: {len(parent_doc.page_content)} characters")
                else:
                    logger.warning(f"Parent document ID {parent_id} not found in docstore. Falling back to child chunk.")
                    parent_docs.append(doc)

            parent_loading_latency = (time.time() - parent_loading_start) * 1000
            total_latency = (time.time() - start_time) * 1000
            logger.info(
                f"Parent retrieval completed. Loaded {len(parent_docs)} parent documents. "
                f"Parent loading latency: {parent_loading_latency:.2f}ms. Total retrieval latency: {total_latency:.2f}ms"
            )
            return parent_docs

        except Exception as e:
            logger.error(f"Parent retrieval failed, falling back to Hybrid Retriever: {e}", exc_info=True)
            # Safe fallback: return hybrid retriever results directly
            try:
                return self.hybrid_retriever.invoke(query, run_manager=run_manager.get_child())
            except Exception as ex:
                logger.error(f"Fallback retrieval also failed: {ex}", exc_info=True)
                return []


# Shared docstore singleton to persist parent mappings across retrieval sessions
_shared_docstore = InMemoryStore()


class ParentRetriever:
    """
    Wrapper managing Parent Document Retriever indexing and retrieval pipelines.
    """

    def __init__(
        self,
        owner_id: Optional[int] = None,
        document_id: Optional[int] = None,
        top_k: int = 5,
        child_chunk_size: int = 300,
        child_overlap: int = 30,
        parent_chunk_size: int = 1024,
        parent_overlap: int = 100,
        chroma_service: Optional[Any] = None,
        embedding_service: Optional[Any] = None,
        docstore: Optional[InMemoryStore] = None,
        base_retriever: Optional[BaseRetriever] = None,
    ):
        self.owner_id = owner_id
        self.document_id = document_id
        self.top_k = top_k
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        self.parent_chunk_size = parent_chunk_size
        self.parent_overlap = parent_overlap

        from app.embeddings.chroma_service import ChromaService
        from app.embeddings.embedding_service import EmbeddingService
        from app.services.langchain.retriever import get_retriever

        self.chroma_service = chroma_service or ChromaService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.docstore = docstore or _shared_docstore
        
        # Hybrid Retriever
        self.base_retriever = base_retriever or get_retriever(
            owner_id=owner_id,
            document_id=document_id,
            top_k=top_k
        )
        self.parent_document_retriever = None
        self.initialize()

    def initialize(self):
        """
        Configures the CustomParentDocumentRetriever.
        """
        logger.info("Initializing Parent Retriever...")
        
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_overlap
        )
        
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_overlap
        )

        self.parent_document_retriever = CustomParentDocumentRetriever(
            vectorstore=DummyVectorStore(),
            docstore=self.docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            hybrid_retriever=self.base_retriever,
            id_key="parent_document_id"
        )

    def add_documents(self, documents: List[Document]):
        """
        Splits parent documents into chunks, assigns parent IDs, updates mapping, and index.
        """
        logger.info(f"Creating parent documents (chunk_size={self.parent_chunk_size})...")
        
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_overlap
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_overlap
        )

        parent_docs_to_store = []
        child_docs_to_index = []

        for doc_idx, doc in enumerate(documents):
            # Split original documents into configured parent chunks
            split_parents = parent_splitter.split_documents([doc])
            
            for p_idx, p_doc in enumerate(split_parents):
                parent_id = f"parent-{self.document_id or 'doc'}-{doc_idx}-{p_idx}-{time.time()}"
                
                # Make sure we preserve metadata from original document
                p_doc.metadata = doc.metadata.copy()
                p_doc.metadata["parent_document_id"] = parent_id
                
                parent_docs_to_store.append((parent_id, p_doc))
                
                # Split parent chunk into smaller child chunks
                logger.info(f"Creating child chunks for parent {parent_id}...")
                sub_docs = child_splitter.split_documents([p_doc])
                for c_idx, c_doc in enumerate(sub_docs):
                    c_doc.metadata = p_doc.metadata.copy()
                    c_doc.metadata["chunk_index"] = c_idx
                    
                    child_id = f"child-{parent_id}-{c_idx}"
                    c_doc.metadata["chunk_id"] = child_id
                    
                    child_docs_to_index.append((child_id, c_doc))

        # Update docstore parent mappings
        logger.info("Saving parent documents to store...")
        self.docstore.mset(parent_docs_to_store)

        # Index child chunks in vector db
        if child_docs_to_index:
            logger.info("Indexing child chunks in vector store...")
            ids = [item[0] for item in child_docs_to_index]
            docs = [item[1] for item in child_docs_to_index]
            
            texts = [d.page_content for d in docs]
            if hasattr(self.embedding_service, "generate_batch_embeddings"):
                embeddings = self.embedding_service.generate_batch_embeddings(texts)
            elif hasattr(self.embedding_service, "embed_documents"):
                embeddings = self.embedding_service.embed_documents(texts)
            else:
                raise AttributeError("The embedding service does not support generate_batch_embeddings or embed_documents.")
            metadatas = [d.metadata for d in docs]
            
            self.chroma_service.add_documents(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            
        logger.info(f"Parent indexing complete. Added {len(parent_docs_to_store)} parents, {len(child_docs_to_index)} child chunks.")

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieves relevant parent documents matching query child chunks.
        """
        return self.parent_document_retriever.invoke(query)
