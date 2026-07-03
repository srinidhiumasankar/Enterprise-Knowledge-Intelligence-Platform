# tests/verify_phase_8_1.py
# -------------------------
# Verification script for Phase 8.1 (LangChain Foundation).
# Initializes the LangChain LLM wrapper and invokes it with a simple prompt.

import os
import sys
import logging

# Add workspace directory to sys.path to ensure local app imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.langchain import get_llm, get_embeddings, create_basic_chain

# Configure lightweight logging for verification
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_1")


def verify_langchain_setup():
    """
    Verifies the LangChain foundation layer components:
    1. Loads environmental configurations (API Key)
    2. Initializes LLM using the get_llm() dependency
    3. Invokes the model with a simple prompt
    4. Initializes embeddings using get_embeddings()
    5. Construct and runs a basic LCEL chain
    """
    logger.info("==========================================================")
    logger.info("STARTING PHASE 8.1 LANGCHAIN FOUNDATION VERIFICATION")
    logger.info("==========================================================")

    # 1. Initialize LangChain LLM wrapper
    logger.info("Initializing LangChain LLM wrapper...")
    llm = get_llm()
    logger.info(f"✓ ChatGoogleGenerativeAI successfully initialized: {llm}")

    # 2. Send a simple prompt directly
    prompt_text = "Say Hello from LangChain"
    logger.info(f"Sending prompt to LLM: '{prompt_text}'...")
    try:
        response = llm.invoke(prompt_text)
        logger.info("✓ Prompt execution succeeded.")
        print("\n--- LLM Direct Response ---")
        print(response.content)
        print("----------------------------\n")
    except Exception as e:
        logger.error(f"❌ Failed during direct LLM invocation: {e}", exc_info=True)
        sys.exit(1)

    # 3. Initialize Embeddings wrapper
    logger.info("Initializing LangChain Embeddings wrapper...")
    try:
        embeddings = get_embeddings()
        logger.info(f"✓ GoogleGenerativeAIEmbeddings successfully initialized: {embeddings}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Embeddings: {e}", exc_info=True)
        sys.exit(1)

    # 4. Construct and invoke basic chain
    logger.info("Constructing and testing basic LCEL chain...")
    try:
        chain = create_basic_chain()
        chain_res = chain.invoke({"instruction": "confirm LangChain LCEL chain works!"})
        logger.info("✓ LCEL basic chain execution succeeded.")
        print("\n--- LCEL Chain Response ---")
        print(chain_res)
        print("---------------------------\n")
    except Exception as e:
        logger.error(f"❌ Failed during LCEL chain invocation: {e}", exc_info=True)
        sys.exit(1)

    logger.info("==========================================================")
    logger.info("ALL PHASE 8.1 LANGCHAIN FOUNDATION VERIFICATION PASSED!")
    logger.info("==========================================================")


if __name__ == "__main__":
    verify_langchain_setup()
