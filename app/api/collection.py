# app/api/collection.py
# --------------------
# FastAPI route handlers exposing API endpoints for Document Collections management.

import logging
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_current_active_user, get_collection_service
from app.models.user import User
from app.services.collection.collection_service import CollectionService
from app.schemas.collection import (
    CollectionCreate,
    CollectionUpdate,
    CollectionResponse,
    CollectionListResponse,
    AddDocumentRequest,
    CollectionStatistics
)
from app.schemas.upload import DocumentResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/collections", tags=["collections"])


@router.post("", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
def create_collection(
    payload: CollectionCreate,
    current_user: User = Depends(get_current_active_user),
    service: CollectionService = Depends(get_collection_service)
):
    """
    Creates a new collection under the designated or default workspace.
    """
    try:
        return service.create_collection(
            owner_id=current_user.id,
            name=payload.name,
            description=payload.description,
            workspace_id=payload.workspace_id
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create collection: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.get("", response_model=CollectionListResponse)
def list_collections(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    service: CollectionService = Depends(get_collection_service)
):
    """
    Lists collections owned by the current user.
    """
    try:
        items, total = service.list_collections(current_user.id, page, page_size)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0
        return {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total_records": total,
            "total_pages": total_pages
        }
    except Exception as e:
        logger.error(f"Failed to list collections: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.get("/{collection_id}", response_model=CollectionResponse)
def get_collection(
    collection_id: int,
    current_user: User = Depends(get_current_active_user),
    service: CollectionService = Depends(get_collection_service)
):
    """
    Retrieves collection details.
    """
    try:
        return service.get_collection(collection_id, current_user.id)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to fetch collection {collection_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.patch("/{collection_id}", response_model=CollectionResponse)
def update_collection(
    collection_id: int,
    payload: CollectionUpdate,
    current_user: User = Depends(get_current_active_user),
    service: CollectionService = Depends(get_collection_service)
):
    """
    Updates collection title name or description description.
    """
    try:
        return service.update_collection(
            collection_id=collection_id,
            owner_id=current_user.id,
            name=payload.name,
            description=payload.description
        )
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update collection {collection_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.delete("/{collection_id}", status_code=status.HTTP_200_OK)
def delete_collection(
    collection_id: int,
    current_user: User = Depends(get_current_active_user),
    service: CollectionService = Depends(get_collection_service)
):
    """
    Deletes the collection metadata record.
    """
    try:
        success = service.delete_collection(collection_id, current_user.id)
        return {
            "success": success,
            "message": "Collection deleted successfully" if success else "Collection could not be deleted"
        }
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete collection {collection_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.post("/{collection_id}/documents", status_code=status.HTTP_200_OK)
def add_document_to_collection(
    collection_id: int,
    payload: AddDocumentRequest,
    current_user: User = Depends(get_current_active_user),
    service: CollectionService = Depends(get_collection_service)
):
    """
    Links a document to the collection.
    """
    try:
        success = service.add_document(collection_id, current_user.id, payload.document_id)
        return {
            "success": success,
            "message": "Document added to collection successfully" if success else "Link could not be established"
        }
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to link document {payload.document_id} to collection {collection_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.delete("/{collection_id}/documents/{document_id}", status_code=status.HTTP_200_OK)
def remove_document_from_collection(
    collection_id: int,
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    service: CollectionService = Depends(get_collection_service)
):
    """
    Unlinks a document from the collection.
    """
    try:
        success = service.remove_document(collection_id, current_user.id, document_id)
        return {
            "success": success,
            "message": "Document removed from collection successfully" if success else "Link could not be unlinked"
        }
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to unlink document {document_id} from collection {collection_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.get("/{collection_id}/documents", response_model=List[DocumentResponse])
def list_collection_documents(
    collection_id: int,
    current_user: User = Depends(get_current_active_user),
    service: CollectionService = Depends(get_collection_service)
):
    """
    Lists all documents linked to the collection.
    """
    try:
        return service.get_documents(collection_id, current_user.id)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to retrieve documents for collection {collection_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.get("/{collection_id}/statistics", response_model=CollectionStatistics)
def get_collection_statistics(
    collection_id: int,
    current_user: User = Depends(get_current_active_user),
    service: CollectionService = Depends(get_collection_service)
):
    """
    Retrieves statistical details for the collection.
    """
    try:
        return service.get_statistics(collection_id, current_user.id)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load statistics for collection {collection_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")
