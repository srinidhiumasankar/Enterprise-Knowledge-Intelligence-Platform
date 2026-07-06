# app/api/conversation.py
# ------------------------
# FastAPI route handlers exposing API endpoints for managing conversations and message histories.

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_current_active_user, get_conversation_service
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
        items, total = service.search_conversations(current_user.id, keyword, page, page_size)
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
        return service.rename_conversation(conversation_id, current_user.id, payload.title)
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
        success = service.delete_conversation(conversation_id, current_user.id)
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
        success = service.permanently_delete(conversation_id, current_user.id)
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
