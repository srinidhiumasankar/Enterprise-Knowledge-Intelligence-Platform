# app/repositories/conversation_repository.py
# ------------------------------------------
# Data access repository layer for chat conversations and message history.

import logging
from typing import List, Tuple, Optional, Any, Dict
from sqlalchemy import select, func
from sqlalchemy.orm import Session, selectinload

from app.models.conversation import Conversation
from app.models.chat_message import ChatMessage, MessageRole

logger = logging.getLogger(__name__)


class ConversationRepository:
    """
    Repository encapsulating DB access and session transactions for chat histories.
    """
    def __init__(self, db: Session):
        self.db = db

    def create(self, workspace_id: int, user_id: int, title: Optional[str] = None) -> Conversation:
        """
        Initializes and registers a new empty Conversation thread record.
        """
        conv = Conversation(
            workspace_id=workspace_id,
            user_id=user_id,
            title=title
        )
        self.db.add(conv)
        self.db.commit()
        self.db.refresh(conv)
        logger.info(f"Conversation registered in DB: id={conv.id}, uuid={conv.uuid}")
        return conv

    def get_by_id(self, conversation_id: int, include_deleted: bool = False) -> Optional[Conversation]:
        """
        Retrieves conversation with loaded chat messages history.
        """
        query = select(Conversation).where(Conversation.id == conversation_id).options(
            selectinload(Conversation.messages)
        )
        if not include_deleted:
            query = query.where(Conversation.deleted_at.is_(None))
        return self.db.scalars(query).first()

    def list_user_conversations(
        self,
        user_id: int,
        page: int = 1,
        page_size: int = 20,
        include_deleted: bool = False,
        workspace_id: Optional[int] = None
    ) -> Tuple[List[Conversation], int]:
        """
        Lists user conversations with paginated counts, excluding archived ones.
        """
        query = select(Conversation).where(
            Conversation.user_id == user_id,
            Conversation.is_archived.is_(False)
        )
        if workspace_id is not None:
            query = query.where(Conversation.workspace_id == workspace_id)
        if not include_deleted:
            query = query.where(Conversation.deleted_at.is_(None))

        # Perform count
        count_query = select(func.count()).select_from(query.subquery())
        total_records = self.db.scalar(count_query) or 0

        # Sort order (newest updated first)
        query = query.order_by(Conversation.updated_at.desc(), Conversation.id.desc())

        # Offsets
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)

        items = list(self.db.scalars(query).all())
        return items, total_records

    def rename(self, conversation_id: int, new_title: str) -> Optional[Conversation]:
        """
        Renames the conversation title and updates updated_at.
        """
        conv = self.db.get(Conversation, conversation_id)
        if conv and conv.deleted_at is None:
            conv.title = new_title
            conv.updated_at = func.now()
            self.db.add(conv)
            self.db.commit()
            self.db.refresh(conv)
            logger.info(f"Conversation renamed in DB: id={conversation_id}, title={new_title}")
            return conv
        return None

    def pin(self, conversation_id: int) -> bool:
        """
        Sets conversation is_pinned flag to True.
        """
        conv = self.db.get(Conversation, conversation_id)
        if conv and conv.deleted_at is None and not conv.is_pinned:
            conv.is_pinned = True
            conv.updated_at = func.now()
            self.db.add(conv)
            self.db.commit()
            logger.info(f"Conversation pinned in DB: id={conversation_id}")
            return True
        return False

    def unpin(self, conversation_id: int) -> bool:
        """
        Sets conversation is_pinned flag to False.
        """
        conv = self.db.get(Conversation, conversation_id)
        if conv and conv.deleted_at is None and conv.is_pinned:
            conv.is_pinned = False
            conv.updated_at = func.now()
            self.db.add(conv)
            self.db.commit()
            logger.info(f"Conversation unpinned in DB: id={conversation_id}")
            return True
        return False

    def archive(self, conversation_id: int) -> bool:
        """
        Sets conversation is_archived flag to True.
        """
        conv = self.db.get(Conversation, conversation_id)
        if conv and conv.deleted_at is None and not conv.is_archived:
            conv.is_archived = True
            conv.updated_at = func.now()
            self.db.add(conv)
            self.db.commit()
            logger.info(f"Conversation archived in DB: id={conversation_id}")
            return True
        return False

    def unarchive(self, conversation_id: int) -> bool:
        """
        Sets conversation is_archived flag to False.
        """
        conv = self.db.get(Conversation, conversation_id)
        if conv and conv.deleted_at is None and conv.is_archived:
            conv.is_archived = False
            conv.updated_at = func.now()
            self.db.add(conv)
            self.db.commit()
            logger.info(f"Conversation unarchived in DB: id={conversation_id}")
            return True
        return False

    def list_pinned(self, user_id: int, workspace_id: Optional[int] = None) -> List[Conversation]:
        """
        Lists active pinned conversations for a user.
        """
        query = select(Conversation).where(
            Conversation.user_id == user_id,
            Conversation.is_pinned.is_(True),
            Conversation.deleted_at.is_(None),
            Conversation.is_archived.is_(False)
        )
        if workspace_id is not None:
            query = query.where(Conversation.workspace_id == workspace_id)
        query = query.order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        return list(self.db.scalars(query).all())

    def list_archived(self, user_id: int, workspace_id: Optional[int] = None) -> List[Conversation]:
        """
        Lists active archived conversations for a user.
        """
        query = select(Conversation).where(
            Conversation.user_id == user_id,
            Conversation.is_archived.is_(True),
            Conversation.deleted_at.is_(None)
        )
        if workspace_id is not None:
            query = query.where(Conversation.workspace_id == workspace_id)
        query = query.order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        return list(self.db.scalars(query).all())

    def search(
        self,
        user_id: int,
        keyword: str,
        page: int = 1,
        page_size: int = 20,
        workspace_id: Optional[int] = None
    ) -> Tuple[List[Conversation], int]:
        """
        Performs case-insensitive partial keyword search over titles and message bodies.
        """
        from sqlalchemy import or_, and_
        
        base_filter = and_(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None)
        )
        if workspace_id is not None:
            base_filter = and_(base_filter, Conversation.workspace_id == workspace_id)
            
        kw_filter = or_(
            Conversation.title.ilike(f"%{keyword}%"),
            ChatMessage.content.ilike(f"%{keyword}%")
        )
        
        # Outer join to match empty conversations too if they match title
        query = select(Conversation).outerjoin(ChatMessage).where(base_filter).where(kw_filter).distinct()
        
        count_query = select(func.count(func.distinct(Conversation.id))).select_from(
            Conversation
        ).outerjoin(ChatMessage).where(base_filter).where(kw_filter)
        
        total_records = self.db.scalar(count_query) or 0
        
        # Sort and offsets
        query = query.order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        
        items = list(self.db.scalars(query).all())
        return items, total_records

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        token_count: Optional[int] = None,
        model_name: Optional[str] = None,
        metadata_json: Optional[Dict[str, Any]] = None
    ) -> ChatMessage:
        """
        Appends a chat message to a conversation and updates updated_at timestamp.
        """
        role_enum = MessageRole[role.upper()]
        msg = ChatMessage(
            conversation_id=conversation_id,
            role=role_enum,
            content=content,
            token_count=token_count,
            model_name=model_name,
            metadata_json=metadata_json
        )
        self.db.add(msg)

        # Update parent conversation timestamp
        conv = self.db.get(Conversation, conversation_id)
        if conv:
            conv.updated_at = func.now()
            self.db.add(conv)

        self.db.commit()
        self.db.refresh(msg)
        logger.info(f"ChatMessage stored in DB: id={msg.id}, conversation_id={conversation_id}, role={role}")
        return msg

    def soft_delete(self, conversation_id: int) -> bool:
        """
        Marks conversation with a deleted_at timestamp.
        """
        conv = self.db.get(Conversation, conversation_id)
        if conv and conv.deleted_at is None:
            conv.deleted_at = func.now()
            self.db.add(conv)
            self.db.commit()
            logger.info(f"Conversation soft-deleted: id={conversation_id}")
            return True
        return False

    def restore(self, conversation_id: int) -> bool:
        """
        Clears soft-deleted_at timestamp.
        """
        conv = self.db.get(Conversation, conversation_id)
        if conv and conv.deleted_at is not None:
            conv.deleted_at = None
            self.db.add(conv)
            self.db.commit()
            logger.info(f"Conversation restored: id={conversation_id}")
            return True
        return False

    def hard_delete(self, conversation_id: int) -> bool:
        """
        Permanently drops record from database.
        """
        conv = self.db.get(Conversation, conversation_id)
        if conv:
            self.db.delete(conv)
            self.db.commit()
            logger.info(f"Conversation hard-deleted: id={conversation_id}")
            return True
        return False

    def exists(self, conversation_id: int, include_deleted: bool = False) -> bool:
        """
        Checks presence of conversation record.
        """
        query = select(Conversation.id).where(Conversation.id == conversation_id)
        if not include_deleted:
            query = query.where(Conversation.deleted_at.is_(None))
        return self.db.scalar(query) is not None
