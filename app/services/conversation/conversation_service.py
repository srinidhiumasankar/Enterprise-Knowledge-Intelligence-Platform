# app/services/conversation/conversation_service.py
# -----------------------------------------------
# Core service orchestrating conversational thread validations, ownership checks, and fallback logic.

import logging
from typing import List, Tuple, Optional, Any, Dict
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.conversation import Conversation
from app.models.chat_message import ChatMessage
from app.models.workspace import Workspace
from app.models.user_preference import UserPreference
from app.repositories.conversation_repository import ConversationRepository

logger = logging.getLogger(__name__)


class ConversationService:
    """
    Service layer providing logic validations for conversations and messages lifecycle.
    """
    def __init__(self, db: Session):
        self.db = db
        self.repo = ConversationRepository(db)

    def _validate_workspace(self, user_id: int, workspace_id: Optional[int]) -> int:
        """
        Validates workspace access, or constructs a default workspace for the user if none exists.
        """
        if workspace_id is not None:
            workspace = self.db.scalar(select(Workspace).where(Workspace.id == workspace_id))
            if not workspace:
                logger.warning(f"Workspace {workspace_id} not found.")
                raise ValueError("Workspace not found")
            if workspace.owner_id != user_id:
                logger.warning(f"Unauthorized access check failed for user {user_id} on workspace {workspace_id}")
                raise PermissionError("Unauthorized access to workspace")
            return workspace.id

        # Fallback 1: Check user preference default workspace
        pref = self.db.scalar(select(UserPreference).where(UserPreference.user_id == user_id))
        if pref and pref.default_workspace:
            ws_id = pref.default_workspace
            # Validate preference workspace exists
            ws = self.db.scalar(select(Workspace).where(Workspace.id == ws_id))
            if ws and ws.owner_id == user_id:
                return ws_id

        # Fallback 2: Check first workspace owned by user
        first_ws = self.db.scalars(select(Workspace).where(Workspace.owner_id == user_id)).first()
        if first_ws:
            return first_ws.id

        # Fallback 3: Automatically initialize a new default workspace
        new_ws = Workspace(
            owner_id=user_id,
            name="Default Workspace",
            description="Auto-generated default workspace",
            is_default=True,
            is_active=True
        )
        self.db.add(new_ws)
        self.db.commit()
        self.db.refresh(new_ws)
        
        # Save to preference as default
        if pref:
            pref.default_workspace = new_ws.id
            self.db.add(pref)
            self.db.commit()
            
        logger.info(f"Auto-created default Workspace {new_ws.id} for user {user_id}")
        return new_ws.id

    def create_conversation(
        self,
        user_id: int,
        workspace_id: Optional[int] = None,
        title: Optional[str] = None
    ) -> Conversation:
        """
        Creates a new conversation under the designated or default workspace context.
        """
        # Validate or default workspace
        validated_ws_id = self._validate_workspace(user_id, workspace_id)
        
        conv = self.repo.create(
            workspace_id=validated_ws_id,
            user_id=user_id,
            title=title or "New Conversation"
        )
        logger.info(f"Conversation created successfully: id={conv.id}, user_id={user_id}, workspace_id={validated_ws_id}")
        return conv

    def get_conversation(self, conversation_id: int, user_id: int, include_deleted: bool = False) -> Conversation:
        """
        Retrieves a conversation thread with message listings, validating ownership permissions.
        """
        conv = self.repo.get_by_id(conversation_id, include_deleted=include_deleted)
        if not conv:
            logger.warning(f"Conversation {conversation_id} not found.")
            raise KeyError("Conversation not found")
        if conv.user_id != user_id:
            logger.warning(f"User {user_id} unauthorized to access conversation {conversation_id}")
            raise PermissionError("Unauthorized access to conversation")
        return conv

    def list_conversations(self, user_id: int, page: int = 1, page_size: int = 20) -> Tuple[List[Conversation], int]:
        """
        Retrieves a paginated list of non-deleted conversations belonging to a user.
        """
        return self.repo.list_user_conversations(user_id, page, page_size, include_deleted=False)

    def append_message(
        self,
        conversation_id: int,
        user_id: int,
        role: str,
        content: str,
        token_count: Optional[int] = None,
        model_name: Optional[str] = None,
        metadata_json: Optional[Dict[str, Any]] = None
    ) -> ChatMessage:
        """
        Appends a new query or response message element to an existing conversation thread.
        """
        # Retrieve conversation (validates ownership checks)
        self.get_conversation(conversation_id, user_id)
        
        msg = self.repo.add_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            token_count=token_count,
            model_name=model_name,
            metadata_json=metadata_json
        )
        logger.info(f"Message successfully appended to conversation: id={msg.id}, role={role}")
        return msg

    def delete_conversation(self, conversation_id: int, user_id: int) -> bool:
        """
        Soft-deletes a conversation thread.
        """
        self.get_conversation(conversation_id, user_id)
        res = self.repo.soft_delete(conversation_id)
        if res:
            logger.info(f"Conversation soft-deleted: id={conversation_id}")
        return res

    def restore_conversation(self, conversation_id: int, user_id: int) -> bool:
        """
        Restores a previously soft-deleted conversation thread.
        """
        # Retrieve soft-deleted conversation for validation
        self.get_conversation(conversation_id, user_id, include_deleted=True)
        res = self.repo.restore(conversation_id)
        if res:
            logger.info(f"Conversation restored successfully: id={conversation_id}")
        return res

    def permanently_delete(self, conversation_id: int, user_id: int) -> bool:
        """
        Permanently hard-deletes a conversation thread.
        """
        self.get_conversation(conversation_id, user_id, include_deleted=True)
        res = self.repo.hard_delete(conversation_id)
        if res:
            logger.info(f"Conversation permanently dropped: id={conversation_id}")
        return res

    def rename_conversation(self, conversation_id: int, user_id: int, new_title: str) -> Conversation:
        """
        Renames the conversation title after validating ownership.
        """
        self.get_conversation(conversation_id, user_id)
        conv = self.repo.rename(conversation_id, new_title)
        if not conv:
            raise KeyError("Conversation not found")
        logger.info(f"Conversation renamed: id={conversation_id}, new_title={new_title}")
        return conv

    def pin_conversation(self, conversation_id: int, user_id: int) -> bool:
        """
        Pins a conversation after validating ownership.
        """
        self.get_conversation(conversation_id, user_id)
        res = self.repo.pin(conversation_id)
        if res:
            logger.info(f"Conversation pinned: id={conversation_id}")
        return res

    def unpin_conversation(self, conversation_id: int, user_id: int) -> bool:
        """
        Unpins a conversation after validating ownership.
        """
        self.get_conversation(conversation_id, user_id)
        res = self.repo.unpin(conversation_id)
        if res:
            logger.info(f"Conversation unpinned: id={conversation_id}")
        return res

    def archive_conversation(self, conversation_id: int, user_id: int) -> bool:
        """
        Archives a conversation after validating ownership.
        """
        self.get_conversation(conversation_id, user_id)
        res = self.repo.archive(conversation_id)
        if res:
            logger.info(f"Conversation archived: id={conversation_id}")
        return res

    def unarchive_conversation(self, conversation_id: int, user_id: int) -> bool:
        """
        Restores an archived conversation back to normal listing after validating ownership.
        """
        self.get_conversation(conversation_id, user_id)
        res = self.repo.unarchive(conversation_id)
        if res:
            logger.info(f"Conversation unarchived: id={conversation_id}")
        return res

    def get_pinned_conversations(self, user_id: int) -> List[Conversation]:
        """
        Retrieves all pinned active conversations for the user.
        """
        return self.repo.list_pinned(user_id)

    def get_archived_conversations(self, user_id: int) -> List[Conversation]:
        """
        Retrieves all archived active conversations for the user.
        """
        return self.repo.list_archived(user_id)

    def search_conversations(
        self,
        user_id: int,
        keyword: str,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[Conversation], int]:
        """
        Executes paginated database query search matching keyword across titles and message bodies.
        """
        logger.info(f"Search request: user={user_id}, keyword='{keyword}', page={page}, page_size={page_size}")
        return self.repo.search(user_id, keyword, page, page_size)
