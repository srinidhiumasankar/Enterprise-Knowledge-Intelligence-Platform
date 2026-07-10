# app/api/conversation.py
# ------------------------
# FastAPI route handlers exposing API endpoints for managing conversations and message histories.

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.api.deps import get_current_active_user, get_conversation_service
from app.schemas.retrieval import CollectionRetrievalRequest
from app.models.user import User
from app.services.conversation.conversation_service import ConversationService
from app.schemas.conversation import (
    ConversationCreate,
    ConversationResponse,
    ConversationListResponse,
    ConversationDeleteResponse,
    ChatMessageCreate,
    ChatMessageResponse,
    ConversationRenameRequest,
    ConversationSearchResponse,
    PinnedConversationResponse,
    ArchivedConversationResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: ConversationCreate,
    current_user: User = Depends(get_current_active_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Initializes a new empty conversation thread under the user's active workspace.
    """
    try:
        conv = service.create_conversation(
            user_id=current_user.id,
            workspace_id=payload.workspace_id,
            title=payload.title
        )
        if conv:
            from app.utils.activity_logger import log_activity
            log_activity(service.repo.db, current_user.id, conv.workspace_id, "conversation_created", f"Created conversation: '{conv.title}'")
        return conv
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.get("", response_model=ConversationListResponse)
def list_conversations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Retrieves a paginated listing of non-deleted, non-archived conversations belonging to the current user.
    """
    try:
        from app.services.workspace.workspace_context_service import WorkspaceContextService
        ctx_service = WorkspaceContextService(service.repo.db)
        active_ws = ctx_service.get_active_workspace(current_user.id)

        items, total = service.list_conversations(
            user_id=current_user.id,
            workspace_id=active_ws.id,
            page=page,
            page_size=page_size
        )
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0
        return {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total_records": total,
            "total_pages": total_pages
        }
    except Exception as e:
        logger.error(f"Failed to list conversations: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.get("/pinned", response_model=PinnedConversationResponse)
def get_pinned_conversations(
    current_user: User = Depends(get_current_active_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Returns all active pinned conversations for the current user.
    """
    try:
        items = service.get_pinned_conversations(current_user.id)
        return {"items": items}
    except Exception as e:
        logger.error(f"Failed to retrieve pinned conversations: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.get("/archived", response_model=ArchivedConversationResponse)
def get_archived_conversations(
    current_user: User = Depends(get_current_active_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Returns all active archived conversations for the current user.
    """
    try:
        items = service.get_archived_conversations(current_user.id)
        return {"items": items}
    except Exception as e:
        logger.error(f"Failed to retrieve archived conversations: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.get("/search", response_model=ConversationSearchResponse)
def search_conversations(
    keyword: str = Query(..., min_length=1),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Searches active conversation titles and message bodies for the matching keyword.
    """
    try:
        from app.services.workspace.workspace_context_service import WorkspaceContextService
        ctx_service = WorkspaceContextService(service.repo.db)
        active_ws = ctx_service.get_active_workspace(current_user.id)
        
        items, total = service.search_conversations(
            user_id=current_user.id,
            keyword=keyword,
            page=page,
            page_size=page_size,
            workspace_id=active_ws.id if active_ws else None
        )
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0
        
        search_items = []
        for conv in items:
            search_items.append({
                "id": conv.id,
                "uuid": conv.uuid,
                "workspace_id": conv.workspace_id,
                "title": conv.title,
                "is_pinned": conv.is_pinned,
                "is_archived": conv.is_archived,
                "message_count": len(conv.messages),
                "created_at": conv.created_at,
                "updated_at": conv.updated_at
            })
            
        return {
            "items": search_items,
            "page": page,
            "page_size": page_size,
            "total_records": total,
            "total_pages": total_pages
        }
    except Exception as e:
        logger.error(f"Search request failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.get("/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Retrieves a specific conversation with all message elements loaded.
    """
    try:
        return service.get_conversation(conversation_id, current_user.id)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to fetch conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.post("/{conversation_id}/messages", response_model=ChatMessageResponse)
def append_message(
    conversation_id: int,
    payload: ChatMessageCreate,
    current_user: User = Depends(get_current_active_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Appends a user/assistant query or response message to a conversation thread.
    """
    try:
        msg = service.append_message(
            conversation_id=conversation_id,
            user_id=current_user.id,
            role=payload.role,
            content=payload.content,
            token_count=payload.token_count,
            model_name=payload.model_name,
            metadata_json=payload.metadata_json
        )
        return msg
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to append message to conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.patch("/{conversation_id}/rename", response_model=ConversationResponse)
def rename_conversation(
    conversation_id: int,
    payload: ConversationRenameRequest,
    current_user: User = Depends(get_current_active_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Renames an active conversation thread.
    """
    try:
        trimmed_title = payload.title.strip()
        if not trimmed_title:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Title cannot be empty")
        if len(trimmed_title) > 100:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Title cannot exceed 100 characters")
        res = service.rename_conversation(conversation_id, current_user.id, trimmed_title)
        if res:
            from app.utils.activity_logger import log_activity
            log_activity(service.repo.db, current_user.id, res.workspace_id, "conversation_renamed", f"Renamed conversation to: '{trimmed_title}'")
        return res
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to rename conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.patch("/{conversation_id}/pin", response_model=ConversationDeleteResponse)
def pin_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Pins a conversation thread.
    """
    try:
        success = service.pin_conversation(conversation_id, current_user.id)
        return {
            "success": success,
            "message": "Conversation pinned successfully" if success else "Conversation already pinned"
        }
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to pin conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.patch("/{conversation_id}/unpin", response_model=ConversationDeleteResponse)
def unpin_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Unpins a conversation thread.
    """
    try:
        success = service.unpin_conversation(conversation_id, current_user.id)
        return {
            "success": success,
            "message": "Conversation unpinned successfully" if success else "Conversation not pinned"
        }
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to unpin conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.patch("/{conversation_id}/archive", response_model=ConversationDeleteResponse)
def archive_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Archives a conversation thread.
    """
    try:
        success = service.archive_conversation(conversation_id, current_user.id)
        return {
            "success": success,
            "message": "Conversation archived successfully" if success else "Conversation already archived"
        }
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to archive conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.patch("/{conversation_id}/unarchive", response_model=ConversationDeleteResponse)
def unarchive_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Restores an archived conversation back to normal listing.
    """
    try:
        success = service.unarchive_conversation(conversation_id, current_user.id)
        return {
            "success": success,
            "message": "Conversation unarchived successfully" if success else "Conversation not archived"
        }
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to unarchive conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.delete("/{conversation_id}", response_model=ConversationDeleteResponse)
def soft_delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Soft-deletes a conversation thread.
    """
    try:
        conv = service.get_conversation(conversation_id, current_user.id)
        title = conv.title
        workspace_id = conv.workspace_id
        success = service.delete_conversation(conversation_id, current_user.id)
        if success:
            from app.utils.activity_logger import log_activity
            log_activity(service.repo.db, current_user.id, workspace_id, "conversation_deleted", f"Deleted conversation: '{title}'")
        return {
            "success": success,
            "message": "Conversation soft-deleted successfully" if success else "Conversation could not be deleted"
        }
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to soft delete conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.post("/{conversation_id}/restore", response_model=ConversationDeleteResponse)
def restore_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Restores a soft-deleted conversation thread.
    """
    try:
        success = service.restore_conversation(conversation_id, current_user.id)
        return {
            "success": success,
            "message": "Conversation restored successfully" if success else "Conversation could not be restored"
        }
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to restore conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.delete("/{conversation_id}/permanent", response_model=ConversationDeleteResponse)
def hard_delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_active_user),
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Permanently hard-deletes a conversation thread.
    """
    try:
        conv = service.get_conversation(conversation_id, current_user.id, include_deleted=True)
        title = conv.title
        workspace_id = conv.workspace_id
        success = service.permanently_delete(conversation_id, current_user.id)
        if success:
            from app.utils.activity_logger import log_activity
            log_activity(service.repo.db, current_user.id, workspace_id, "conversation_deleted", f"Permanently deleted conversation: '{title}'")
        return {
            "success": success,
            "message": "Conversation permanently deleted" if success else "Conversation could not be deleted permanently"
        }
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to permanently delete conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.post("/{conversation_id}/stream")
async def stream_assistant_response(
    conversation_id: int,
    request_data: CollectionRetrievalRequest,
    current_user: User = Depends(get_current_active_user),
):
    """
    Generate and stream assistant response inline with conversation history and workspace context.
    """
    from app.database.connection import SessionLocal
    import asyncio
    import json
    
    # Run retrieval synchronous setup to resolve workspace ID
    db_setup = SessionLocal()
    try:
        from app.services.workspace.workspace_service import WorkspaceService
        ws_service = WorkspaceService(db_setup)
        active_ws = ws_service.get_active_workspace(current_user.id)
        workspace_id = request_data.workspace_id or (active_ws.id if active_ws else None)
    finally:
        db_setup.close()

    async def event_generator():
        import time
        from app.database.connection import SessionLocal
        from app.services.conversation.conversation_service import ConversationService
        from app.services.conversation.conversation_memory_service import ConversationMemoryService
        from app.services.retrieval_service import RetrievalService
        from app.database.chunk_repository import ChunkRepository
        from app.database.document_repository import DocumentRepository
        from app.ai.prompt_builder import PromptBuilder
        from app.ai.gemini_service import GeminiService
        from app.utils.activity_logger import record_rag_search_history

        db = SessionLocal()
        try:
            query_str = request_data.query
            start_time = time.perf_counter()

            # 1. Load conversation history
            conversation_history = None
            memory_tokens = 0
            loaded_msg_count = 0
            try:
                mem_service = ConversationMemoryService(db)
                conversation_history, memory_tokens = mem_service.load_conversation_history(
                    conversation_id=conversation_id,
                    user_id=current_user.id,
                    max_messages=11
                )
                if conversation_history:
                    lines = conversation_history.strip().split("\n")
                    if lines and lines[-1].startswith("User: ") and lines[-1][6:].strip().lower() == query_str.strip().lower():
                        lines.pop()
                    conversation_history = "\n".join(lines) if lines else None
                if conversation_history:
                    loaded_msg_count = len(conversation_history.strip().split("\n"))
            except PermissionError as perm_e:
                logger.error(f"Unauthorized stream request: user {current_user.id} for conversation {conversation_id}")
                yield json.dumps({
                    "type": "error",
                    "detail": "Unauthorized access to conversation"
                }) + "\n"
                return
            except KeyError as key_e:
                logger.error(f"Conversation {conversation_id} not found for stream request: {key_e}")
                yield json.dumps({
                    "type": "error",
                    "detail": "Conversation not found"
                }) + "\n"
                return
            except Exception as mem_e:
                logger.error(f"Failed to load memory during stream setup: {mem_e}")

            # 2. Retrieve chunks
            chunks = []
            retrieval_latency = 0.0
            doc_repo = DocumentRepository(db)
            docs = doc_repo.get_user_documents(owner_id=current_user.id, workspace_id=workspace_id)
            
            if docs:
                chunk_repo = ChunkRepository(db)
                retrieval_service = RetrievalService(chunk_repo)
                retrieval_start = time.perf_counter()
                try:
                    chunks = await retrieval_service.retrieve(
                        user_id=current_user.id,
                        query=query_str,
                        top_k=request_data.top_k or 5,
                        workspace_id=workspace_id
                    )
                except Exception as ret_err:
                    logger.error(f"Retrieval failed in stream handler: {ret_err}")
                retrieval_latency = (time.perf_counter() - retrieval_start) * 1000

            # Yield metadata/citations first
            citations = []
            search_results = []
            seen_chunks = set()
            for r in chunks:
                chunk_id = r["chunk_id"]
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    citations.append({
                        "document_id": r["document_id"],
                        "filename": r["metadata"]["filename"],
                        "chunk_id": chunk_id,
                        "score": r["score"]
                    })
                search_results.append({
                    "chunk_id": chunk_id,
                    "document_id": r["document_id"],
                    "text": r["text"],
                    "score": r["score"],
                    "metadata": r["metadata"]
                })

            # Send metadata block
            yield json.dumps({
                "type": "metadata",
                "citations": citations,
                "results": search_results
            }) + "\n"

            # 3. Build prompt
            prompt = PromptBuilder.build_prompt(
                question=query_str,
                chunks=chunks,
                conversation_history=conversation_history
            )

            # 4. Stream generated answer from Gemini
            gemini_service = GeminiService()
            gemini_start = time.perf_counter()
            full_text = ""
            generation_latency = 0.0

            try:
                for chunk_text in gemini_service.generate_stream(prompt=prompt, model_name="gemini-2.5-flash"):
                    full_text += chunk_text
                    yield json.dumps({
                        "type": "text",
                        "content": chunk_text
                    }) + "\n"
                    await asyncio.sleep(0.01) # Yield execution thread control
                
                generation_latency = (time.perf_counter() - gemini_start) * 1000
                total_latency = (time.perf_counter() - start_time) * 1000

                # 5. Save final completed response to database
                final_content = full_text.strip()
                if not final_content:
                    final_content = "I couldn't find an answer in your uploaded documents."

                conv_service = ConversationService(db)
                saved_msg = conv_service.append_message(
                    conversation_id=conversation_id,
                    user_id=current_user.id,
                    role="assistant",
                    content=final_content,
                    token_count=len(final_content.split()),
                    model_name="gemini-2.5-flash",
                    metadata_json={
                        "simulated": False,
                        "citations": citations,
                        "search_results": search_results
                    }
                )

                # Log metrics
                logger.info(
                    f"RAG streaming metrics: conversation_id={conversation_id}, "
                    f"loaded_message_count={loaded_msg_count}, retrieved_chunk_count={len(chunks)}, "
                    f"retrieval_latency={retrieval_latency:.2f}ms, generation_latency={generation_latency:.2f}ms, "
                    f"total_latency={total_latency:.2f}ms, prompt_size={len(prompt)}, memory_size={memory_tokens}"
                )

                if workspace_id:
                    doc_names = list(set(r["metadata"]["filename"] for r in chunks if r.get("metadata", {}).get("filename")))
                    scores = [r["score"] for r in chunks]
                    from app.utils.activity_logger import record_rag_search_history
                    record_rag_search_history(
                        current_user.id,
                        workspace_id,
                        query_str,
                        retrieval_latency,
                        generation_latency,
                        total_latency,
                        len(chunks),
                        doc_names,
                        scores,
                        "Success"
                    )

                # Yield final done event
                yield json.dumps({
                    "type": "done",
                    "message_id": saved_msg.id,
                    "content": final_content,
                    "created_at": saved_msg.created_at.isoformat() if hasattr(saved_msg.created_at, "isoformat") else str(saved_msg.created_at)
                }) + "\n"

            except Exception as gemini_err:
                logger.error(f"Streaming generation error: {gemini_err}")
                yield json.dumps({
                    "type": "error",
                    "detail": str(gemini_err)
                }) + "\n"

        finally:
            db.close()

    return StreamingResponse(event_generator(), media_type="text/event-stream")
