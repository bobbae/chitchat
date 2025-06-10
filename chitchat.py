#!/usr/bin/env python3
import asyncio
import datetime
import json
import os
from io import BytesIO
import traceback # For detailed error logging
import tempfile # For handling uploaded PDF files, used in process_documents_for_rag

import openai # Import the openai library for its exception types
import requests 
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mcp_use import MCPClient, MCPAgent # For MCP integration

DEFAULT_PROVIDER_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
    "sambanova": "https://api.sambanova.ai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "anthropic": "https://api.anthropic.com/v1" # Or your specific Anthropic endpoint
}

PROVIDER_API_KEY_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_API_KEY", # For a compatible endpoint
    "sambanova": "SAMBANOVA_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY"
}

DEFAULT_PROVIDER_MODELS = {
    "openai": "gpt-4o", 
    "ollama": "qwen3:8b", 
    "openrouter": "meta-llama/llama-4-maverick:free",
    "sambanova": "DeepSeek-R1",
    "gemini": "gemini-1.5-flash",
    "anthropic": "claude-3-opus-20240229" # Example Anthropic model
}

st.set_page_config(page_title="Chit-Chat with Models", layout="wide")

def try_initialize_mcp_agent():
    """Initializes or updates the MCPAgent if both LLM and MCPClient are available."""
    if st.session_state.openai_client and st.session_state.mcp_client:
        try:
            st.session_state.mcp_agent = MCPAgent(
                llm=st.session_state.openai_client,
                client=st.session_state.mcp_client,
                memory_enabled=False,
                max_steps=10  # Default max steps for the agent
            )
            # st.sidebar.info("MCPAgent initialized/updated.") # Can be noisy, enable if needed
        except Exception as e:
            st.sidebar.error(f"Failed to initialize MCPAgent: {e}")
            st.session_state.mcp_agent = None
    else:
        st.session_state.mcp_agent = None # Ensure it's None if dependencies are missing

def initialize_client(provider, model_name, api_key_input, base_url_input):
    """Initializes the OpenAI client based on selected provider and settings."""
    current_api_key = api_key_input
    provider_env_var_name = PROVIDER_API_KEY_ENV_VARS.get(provider)

    if not current_api_key and provider_env_var_name:
        current_api_key = os.environ.get(provider_env_var_name, "ollama" if provider == "ollama" else None)

    if provider == "ollama" and current_api_key == "ollama":
        st.sidebar.info("Using placeholder API key for Ollama.")
    elif not current_api_key and provider != "ollama": # Ollama can proceed with "ollama" key
        st.sidebar.error(f"API key for {provider} is missing. Please set it in the sidebar or as an environment variable ({provider_env_var_name}).")
        return None

    current_base_url = base_url_input or DEFAULT_PROVIDER_URLS.get(provider)

    if not current_base_url and provider != "ollama":
        st.sidebar.error(f"Base URL for {provider} could not be determined. Please specify it.")
        return None
    
    if provider == "gemini" and (not current_base_url or "generativelanguage.googleapis.com" not in current_base_url):
        st.sidebar.warning(f"For Gemini, ensure the Base URL points to an OpenAI-compatible endpoint if not using a direct proxy like `https://generativelanguage.googleapis.com/v1beta/models` with `openai/` suffix for compatibility.")

    try:
        client = ChatOpenAI(
            model=model_name, # model_name is now part of constructor
            openai_api_key=current_api_key,
            openai_api_base=current_base_url,
            temperature=0.7 # You can set a default temperature or make it configurable
        )
        st.session_state.show_connection_toast = {"provider": provider, "model": model_name}
        return client
    except Exception as e:
        st.sidebar.error(f"Failed to initialize client: {e}")
        return None

# Initialize session state variables
if "histories" not in st.session_state:
    st.session_state.histories = []
if "current_history" not in st.session_state:
    st.session_state.current_history = None
if "openai_client" not in st.session_state:
    st.session_state.openai_client: ChatOpenAI | None = None
if "current_provider" not in st.session_state:
    st.session_state.current_provider = "openai"
if "current_model" not in st.session_state:
    st.session_state.current_model = DEFAULT_PROVIDER_MODELS.get(st.session_state.current_provider, "")
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "base_url" not in st.session_state:
    st.session_state.base_url = DEFAULT_PROVIDER_URLS.get(st.session_state.current_provider, "")
# 'messages' is part of each history entry. 'pending_loaded_histories' and 'activated_from_pending' are no longer needed.
if "pending_history_activation_index" not in st.session_state:
    st.session_state.pending_history_activation_index = None
if "history_file_path" not in st.session_state:
    st.session_state.history_file_path = "chat_histories.json" # Default path for loading histories
if "force_process_prompt" not in st.session_state: # For redo functionality
    st.session_state.force_process_prompt = None
if "retriever" not in st.session_state: # For RAG
    st.session_state.retriever = None
if "processed_doc_names" not in st.session_state: # For RAG
    st.session_state.processed_doc_names = []
if "rag_enabled_by_user" not in st.session_state: # User toggle for RAG
    st.session_state.rag_enabled_by_user = False

if "mcp_config" not in st.session_state: # For MCPClient configuration (stores the full JSON structure)
    st.session_state.mcp_config = {}
if "mcp_client" not in st.session_state: # MCPClient instance
    st.session_state.mcp_client: MCPClient | None = None
if "mcp_agent" not in st.session_state: # MCPAgent instance
    st.session_state.mcp_agent: MCPAgent | None = None
if "mcp_enabled_by_user" not in st.session_state: # User toggle for MCP
    st.session_state.mcp_enabled_by_user = False
if "show_connection_toast" not in st.session_state: # For deferring connection toast
    st.session_state.show_connection_toast = None
if "show_past_rag_usage_toast" not in st.session_state: # For past RAG usage notification
    st.session_state.show_past_rag_usage_toast = False
if "past_rag_documents_to_restore" not in st.session_state: # Store doc names for potential restore
    st.session_state.past_rag_documents_to_restore = []
if "show_mcp_restore_info" not in st.session_state: # For MCP restore info
    st.session_state.show_mcp_restore_info = False

_pending_idx = st.session_state.get("pending_history_activation_index") # For history activation

if _pending_idx is not None:
    _histories = st.session_state.get("histories", [])
    if 0 <= _pending_idx < len(_histories):
        _active_history_obj = _histories[_pending_idx]
        if isinstance(_active_history_obj, dict) and \
           _active_history_obj.get("provider") is not None and \
           _active_history_obj.get("model") is not None:
            st.session_state.current_history = _pending_idx
            st.session_state.current_provider = _active_history_obj["provider"]
            st.session_state.current_model = _active_history_obj["model"]
            st.session_state.base_url = DEFAULT_PROVIDER_URLS.get(st.session_state.current_provider, "")

# Display connection toast if a connection was just made
if st.session_state.get("show_connection_toast"):
    toast_info = st.session_state.show_connection_toast
    st.info(f"Successfully connected to {toast_info['provider']} with model {toast_info['model']}!", icon="✅")
    st.session_state.show_connection_toast = None # Clear the flag

# Display past RAG usage toast if a history with past RAG was just activated
if st.session_state.get("show_past_rag_usage_toast") and st.session_state.current_history is not None:
    toast_message = "Note: This chat previously used RAG."
    if st.session_state.past_rag_documents_to_restore:
        toast_message += f" It referenced documents like: {', '.join(st.session_state.past_rag_documents_to_restore[:2])}{'...' if len(st.session_state.past_rag_documents_to_restore) > 2 else ''}."
    toast_message += " To use RAG now, ensure documents are processed and RAG is enabled in the sidebar."
    st.toast(toast_message, icon="💡")
    st.session_state.show_past_rag_usage_toast = False # Clear the flag
    # The actual prompt to restore will be in the sidebar after it's fully rendered.

# Dynamically set the main page title based on connection status and current/target LLM
provider_for_title = st.session_state.current_provider
model_for_title = st.session_state.current_model
st.title(f"Chat with {provider_for_title} {model_for_title}") # Always display the title


def convert_dict_to_langchain_message(msg_dict: dict) -> BaseMessage:
    role = msg_dict.get("role")
    content = msg_dict.get("content", "")  # Ensure content is not None

    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        lc_tool_calls = []
        raw_tool_calls = msg_dict.get("tool_calls")
        if raw_tool_calls:
            for tc_dict in raw_tool_calls:  # tc_dict is the stored format
                try:
                    args_dict = json.loads(tc_dict.get("function", {}).get("arguments", "{}"))
                except json.JSONDecodeError:
                    args_dict = {} 
                
                lc_tool_calls.append(
                    ToolCall(
                        name=tc_dict.get("function", {}).get("name",""),
                        args=args_dict,
                        id=tc_dict.get("id","")
                    )
                )
        # Add metadata if present in storage
        metadata = msg_dict.get("metadata", {})
        return AIMessage(
            content=content, 
            tool_calls=lc_tool_calls if lc_tool_calls else [],
            metadata=metadata # Pass metadata to LangChain message if needed by other parts
        )
    elif role == "tool":
        return ToolMessage(content=content, tool_call_id=msg_dict.get("tool_call_id",""))
    elif role == "system":
        return SystemMessage(content=content)
    else:
        st.warning(f"Unknown role '{role}' in message history. Treating as HumanMessage.")
        return HumanMessage(content=f"[{role}] {content}")

def convert_aimessage_to_storage_dict(aimsg: AIMessage) -> dict:
    storage_dict = {"role": "assistant", "content": aimsg.content if aimsg.content is not None else ""}
    if aimsg.tool_calls:
        storage_dict["tool_calls"] = []
        for tc in aimsg.tool_calls: # tc is a ToolCall TypedDict
            storage_dict["tool_calls"].append({"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["args"]) if tc["args"] else "{}"}})
    if hasattr(aimsg, 'metadata') and aimsg.metadata: # Store metadata if it exists
        storage_dict["metadata"] = aimsg.metadata
    return storage_dict

def process_loaded_history_data(loaded_data, source_name="file"):
    """Processes loaded history data, updates session state, and shows messages."""
    if not (isinstance(loaded_data, list) and
            all(isinstance(h, dict) and
                "provider" in h and
                "model" in h and
                "messages" in h for h in loaded_data if isinstance(h, dict))):
        st.error(
            f"Invalid history file format from '{source_name}'. "
            "Expected a list of history objects, each with at least 'provider', 'model', and 'messages' keys."
        )
        return

    loaded_count = 0
    updated_count = 0
    for hist_to_load in loaded_data:
        provider = hist_to_load.get("provider")
        model = hist_to_load.get("model")
        messages = hist_to_load.get("messages")

        if not (provider and model and isinstance(messages, list)):
            st.warning(f"Skipping an invalid history entry from '{source_name}': provider={provider}, model={model}, messages_type={type(messages)}")
            continue

        existing_history_index = -1
        for idx, existing_hist in enumerate(st.session_state.histories):
            if existing_hist.get("provider") == provider and \
               existing_hist.get("model") == model:
                existing_history_index = idx
                break
        
        history_entry_data = {
            "provider": provider,
            "model": model,
            "base_url": hist_to_load.get("base_url", DEFAULT_PROVIDER_URLS.get(provider, "")),
            "messages": messages,
            "timestamp": hist_to_load.get("timestamp", datetime.datetime.now().isoformat()),
            "mcp_config_snapshot": hist_to_load.get("mcp_config_snapshot", {}), # Load MCP config
            "mcp_enabled_snapshot": hist_to_load.get("mcp_enabled_snapshot", False), # Load MCP toggle state
            "mcp_config_input": hist_to_load.get("mcp_config_input", "")  # Preserve raw JSON input
        }

        if existing_history_index != -1:
            st.session_state.histories[existing_history_index] = history_entry_data
            updated_count += 1
        else:
            st.session_state.histories.append(history_entry_data)
            loaded_count += 1
    
    if loaded_count > 0 or updated_count > 0:
        st.success(f"Successfully loaded {loaded_count} new and updated {updated_count} existing chat histories from '{source_name}'. They are now available in the 'Chat Histories' dropdown.")
        st.caption("Chat histories loaded. Select a chat from the 'Available Chats' dropdown in the sidebar to activate it.")
        st.rerun()
    else:
        st.info(f"No new or updatable chat histories found in '{source_name}'.")

def process_documents_for_rag(uploaded_files):
    """Loads, splits, embeds, and indexes uploaded documents for RAG."""
    if not uploaded_files:
        st.sidebar.warning("No files uploaded to process.")
        return

    if not st.session_state.openai_client:
        st.sidebar.error("LLM client not initialized. Please connect to a provider before processing documents for RAG.")
        return

    # Use API key and base URL from the initialized ChatOpenAI client
    # This assumes the configured endpoint is also compatible for embeddings.
    # For OpenAI, this is fine. For others, it might need specific embedding models/configs.
    try:
        embeddings_client = OpenAIEmbeddings(
            openai_api_key=st.session_state.openai_client.openai_api_key,
            openai_api_base=st.session_state.openai_client.openai_api_base,
            model="text-embedding-3-small" # Or another preferred OpenAI embedding model
        )
    except Exception as e:
        st.sidebar.error(f"Failed to initialize embeddings client: {e}")
        return

    all_docs_splits = []
    processed_file_names_this_run = []

    with st.spinner("Processing documents for RAG..."):
        for uploaded_file in uploaded_files:
            try:
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                docs = []
                if file_extension == ".pdf":
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                        tmpfile.write(uploaded_file.getvalue())
                        loader = PyPDFLoader(tmpfile.name)
                        docs = loader.load()
                    os.remove(tmpfile.name) # Clean up temp file
                elif file_extension == ".txt":
                    text_content = uploaded_file.getvalue().decode('utf-8')
                    docs = [Document(page_content=text_content, metadata={"source": uploaded_file.name})]
                else:
                    st.sidebar.warning(f"Unsupported file type: {uploaded_file.name}. Skipping.")
                    continue

                if docs:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    splits = text_splitter.split_documents(docs)
                    all_docs_splits.extend(splits)
                    processed_file_names_this_run.append(uploaded_file.name)
                    st.sidebar.write(f"Processed: {uploaded_file.name} ({len(splits)} chunks)")

            except Exception as e:
                st.sidebar.error(f"Error processing {uploaded_file.name}: {e}")

        if all_docs_splits:
            try:
                vector_store = FAISS.from_documents(all_docs_splits, embeddings_client)
                st.session_state.retriever = vector_store.as_retriever()
                st.session_state.processed_doc_names.extend(processed_file_names_this_run)
                st.session_state.processed_doc_names = list(set(st.session_state.processed_doc_names)) # Remove duplicates
                st.sidebar.success(f"RAG index created/updated with {len(all_docs_splits)} chunks from {len(processed_file_names_this_run)} file(s).")
            except Exception as e:
                st.sidebar.error(f"Error creating FAISS index: {e}")
        elif not processed_file_names_this_run: # No files were successfully processed to splits
            st.sidebar.info("No new document content was processed.")


# Sidebar for LLM Configuration
with st.sidebar:
    # Centralized logic to handle history activation before widgets are rendered
    if st.session_state.pending_history_activation_index is not None:
        new_active_idx = st.session_state.pending_history_activation_index
        if 0 <= new_active_idx < len(st.session_state.histories):
            active_history_obj = st.session_state.histories[new_active_idx]
            
            st.session_state.current_history = new_active_idx
            st.session_state.current_provider = active_history_obj["provider"]
            st.session_state.current_model = active_history_obj["model"]
            # Update base_url from the selected history's object, with fallback to default
            st.session_state.base_url = active_history_obj.get("base_url", DEFAULT_PROVIDER_URLS.get(st.session_state.current_provider, ""))
            
            st.session_state.openai_client = initialize_client(
                st.session_state.current_provider,
                st.session_state.current_model,
                st.session_state.api_key, # User's current API key from sidebar
                st.session_state.base_url
            )
            try_initialize_mcp_agent() # Attempt to init agent after LLM is ready

            # Restore MCP settings and raw input from activated history
            st.session_state.mcp_config = active_history_obj.get("mcp_config_snapshot", {})
            # Update the text area with the raw JSON input from history
            if "mcp_config_input" in active_history_obj:
                mcp_config_input_area = json.dumps(active_history_obj["mcp_config_input"], indent=2)
            st.session_state.mcp_enabled_by_user = active_history_obj.get("mcp_enabled_snapshot", False)
            if "mcp_config_snapshot" in active_history_obj or "mcp_enabled_snapshot" in active_history_obj:
                st.session_state.show_mcp_restore_info = True

            # Check for past RAG usage in the newly activated history
            st.session_state.show_past_rag_usage_toast = False # Reset before check
            st.session_state.past_rag_documents_to_restore = [] # Reset
            if isinstance(active_history_obj.get("messages"), list):
                for msg_dict_check in active_history_obj["messages"]:
                    if isinstance(msg_dict_check, dict) and \
                       isinstance(msg_dict_check.get("metadata"), dict):
                        rag_details = msg_dict_check["metadata"].get("rag_details")
                        if isinstance(rag_details, str): # Try to parse if it's a JSON string (old format)
                            try: rag_details = json.loads(rag_details)
                            except json.JSONDecodeError: rag_details = None
                        
                        if isinstance(rag_details, dict) and rag_details.get("toggle_status") == "enabled" and rag_details.get("indexed_documents"):
                            st.session_state.show_past_rag_usage_toast = True
                            st.session_state.past_rag_documents_to_restore = list(set(st.session_state.past_rag_documents_to_restore + rag_details["indexed_documents"])) # Accumulate unique doc names
                            # No break, accumulate all mentioned docs from the history
        else:
            st.error(f"Attempted to activate invalid history index: {new_active_idx}")

        st.session_state.pending_history_activation_index = None # Reset the trigger

    with st.sidebar.expander("Model Configuration", expanded=False):
        selected_provider = st.selectbox(
            "Select Provider:",
            options=list(DEFAULT_PROVIDER_MODELS.keys()),
            index=list(DEFAULT_PROVIDER_MODELS.keys()).index(st.session_state.current_provider)
        )

        # Update model and base_url defaults when provider changes
        if selected_provider != st.session_state.current_provider:
            st.session_state.current_provider = selected_provider
            st.session_state.current_model = DEFAULT_PROVIDER_MODELS.get(selected_provider, "")
            st.session_state.base_url = DEFAULT_PROVIDER_URLS.get(selected_provider, "")
            # Clear client to force re-initialization
            st.session_state.openai_client = None 

        st.session_state.current_model = st.text_input("Model Name:", value=st.session_state.current_model)
        
        # The API key input is bound to st.session_state.api_key.
        # It will start empty as st.session_state.api_key is initialized to "".
        # If left empty by the user, initialize_client will try to use the ENV var.
        st.text_input("API Key (optional, uses ENV if blank):", type="password", key="api_key")
        
        st.text_input("Base URL (optional, uses default if blank):", key="base_url")

        if st.button("Apply Configuration & Connect"):
            if not st.session_state.current_model:
                st.sidebar.error("Model name cannot be empty.")
            else:
                st.session_state.openai_client = initialize_client(
                    st.session_state.current_provider,
                    st.session_state.current_model,
                    st.session_state.api_key,
                    st.session_state.base_url
                )
                try_initialize_mcp_agent() # Attempt to init agent after LLM is ready
                if st.session_state.openai_client:
                    # Check if a history with the same provider and model already exists
                    existing_history_index = -1
                    for i, hist in enumerate(st.session_state.histories):
                        if hist["provider"] == st.session_state.current_provider and \
                        hist["model"] == st.session_state.current_model:
                            existing_history_index = i
                            # Check this existing_history for past RAG usage and set toast flag
                            current_active_history_obj = st.session_state.histories[existing_history_index]
                            st.session_state.show_past_rag_usage_toast = False # Reset before check
                            # Restore MCP settings from this existing history
                            st.session_state.mcp_config = current_active_history_obj.get("mcp_config_snapshot", {})
                            st.session_state.mcp_enabled_by_user = current_active_history_obj.get("mcp_enabled_snapshot", False)
                            if "mcp_config_snapshot" in current_active_history_obj or "mcp_enabled_snapshot" in current_active_history_obj:                             st.session_state.show_mcp_restore_info = True
                            st.session_state.past_rag_documents_to_restore = [] # Reset
                            if isinstance(current_active_history_obj.get("messages"), list):
                                for msg_dict_check in current_active_history_obj["messages"]:
                                    if isinstance(msg_dict_check, dict) and \
                                    isinstance(msg_dict_check.get("metadata"), dict):
                                        rag_details = msg_dict_check["metadata"].get("rag_details")
                                        if isinstance(rag_details, str): # Try to parse if it's a JSON string
                                            try: rag_details = json.loads(rag_details)
                                            except json.JSONDecodeError: rag_details = None
                                        
                                        if isinstance(rag_details, dict) and rag_details.get("toggle_status") == "enabled" and rag_details.get("indexed_documents"):
                                            st.session_state.show_past_rag_usage_toast = True
                                            st.session_state.past_rag_documents_to_restore = list(set(st.session_state.past_rag_documents_to_restore + rag_details["indexed_documents"]))
                            break
                    
                    if existing_history_index != -1:
                        st.session_state.current_history = existing_history_index
                        st.sidebar.info(f"Switched to existing chat history for {st.session_state.current_provider} - {st.session_state.current_model}.")
                    else:
                        # Create new history entry
                        new_history = {
                            "provider": st.session_state.current_provider,
                            "model": st.session_state.current_model,
                            "base_url": st.session_state.base_url, # Store the current base_url
                            "messages": [],
                            "timestamp": datetime.datetime.now().isoformat(),
                            "mcp_config_snapshot": json.loads(st.session_state.mcp_config_input),  # Store parsed config
                            "mcp_config_input": st.session_state.mcp_config_input,  # Store raw textarea content
                            "mcp_enabled_snapshot": st.session_state.mcp_enabled_by_user # Snapshot current MCP toggle
                        }
                        st.session_state.histories.append(new_history)
                        st.session_state.current_history = len(st.session_state.histories) - 1
                        st.sidebar.success("New chat history created. Configuration applied successfully. Ready to chat!")
                        st.session_state.past_rag_documents_to_restore = [] # New history won't have past RAG
                        st.session_state.show_past_rag_usage_toast = False # New history won't have past RAG
                        st.session_state.show_mcp_restore_info = False # New history reflects current, no "restore" needed yet
                    st.rerun() # Rerun to update the main page title and reflect connection status
                else:
                    st.sidebar.error("LLM Client connection failed. Please check settings in sidebar and console.")

    with st.sidebar.expander("Chat History Management", expanded=False):
        # --- Load Histories Section ---
        st.text_input(
            "History JSON File Path:",
            key="history_file_path", # Binds to st.session_state.history_file_path
            help="Path to the chat histories JSON file (e.g., chat_histories.json)"
        )

        if st.button("Load Histories from Path"):
            file_path = st.session_state.history_file_path
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                process_loaded_history_data(data, source_name=file_path) # Directly processes and reruns if successful
            except FileNotFoundError:
                st.error(f"History file '{file_path}' not found.")
            except json.JSONDecodeError:
                st.error(f"Error decoding JSON from '{file_path}'. Please ensure it's a valid JSON file.")
            except Exception as e:
                st.error(f"Error loading histories from '{file_path}': {e}")

        # Save All History
        if st.session_state.histories:
            history_json = json.dumps(st.session_state.histories, indent=2, ensure_ascii=False)
            st.download_button(
                label="Save All History",
                data=history_json,
                file_name="chat_histories.json",
                mime="application/json",
            )
    with st.sidebar.expander("Available Chats", expanded=False):
        # Chat selection dropdown
        if st.session_state.histories:
            history_names = [ 
                f"{hist['provider']} - {hist['model']} ({len(hist['messages'])} messages)" 
                for hist in st.session_state.histories
            ]
            selected_hist = st.selectbox( 
                "Choose a Chat History to Activate:", 
                options=range(len(st.session_state.histories)),
                format_func=lambda x: history_names[x],
                index=st.session_state.current_history or 0 
            )
            
            if selected_hist != st.session_state.current_history:
                st.session_state.pending_history_activation_index = selected_hist
                st.rerun()
        else:
            st.caption("No chats available. Start a new chat by applying a model configuration.")
        
        # Button to clear the currently selected chat history (moved here)
        if st.session_state.current_history is not None and st.session_state.histories: # Only show if a history is active
            if st.button("Clear Selected Chat History", key="clear_selected_chat_history_sidebar_button"):
                if 0 <= st.session_state.current_history < len(st.session_state.histories):
                    st.session_state.histories[st.session_state.current_history]["messages"] = []
                st.rerun()

    with st.sidebar.expander("RAG Configuration", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT) for RAG",
            accept_multiple_files=True,
            type=['pdf', 'txt'],
            key="rag_file_uploader"
        )
        st.toggle("Enable RAG", key="rag_enabled_by_user", help="If enabled and documents are indexed, RAG context will be used with a direct LLM call.")

        if st.session_state.rag_enabled_by_user and st.session_state.retriever:
            st.info("RAG is ENABLED and documents are indexed.", icon="📄")
        elif st.session_state.rag_enabled_by_user and not st.session_state.retriever:
            st.warning("RAG is enabled by toggle, but no documents are indexed yet. Process documents to use RAG.", icon="⚠️")

        # Add warning if both RAG and MCP are enabled by user
        if st.session_state.rag_enabled_by_user and st.session_state.mcp_enabled_by_user:
            st.warning("Both RAG and MCP are enabled. RAG will take precedence. To use MCPAgent, disable RAG.", icon="ℹ️")

        if st.button("Process Documents for RAG"):
            process_documents_for_rag(uploaded_files)

        if st.session_state.processed_doc_names:
            st.write("Currently indexed documents for RAG:")
            for i, name in enumerate(st.session_state.processed_doc_names):
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.caption(f"- {name}")
                with col2:
                    if st.button("❌", key=f"remove_rag_doc_{i}", help=f"Remove '{name}' from RAG list and clear RAG index."):
                        st.session_state.processed_doc_names.pop(i)
                        st.session_state.retriever = None # Clear the RAG index
                        st.sidebar.info(f"Removed '{name}' from RAG list. The RAG index has been cleared. Please re-process documents if needed.")
                        st.rerun()
            if st.button("Clear All RAG Data (Index & Processed Documents)"):
                st.session_state.retriever = None
                st.session_state.processed_doc_names = []
                st.rerun()
        # Display RAG restore prompt if applicable for the currently active chat
        if st.session_state.current_history is not None and st.session_state.past_rag_documents_to_restore:
            st.info(
                f"Active chat previously used RAG with: **{', '.join(st.session_state.past_rag_documents_to_restore)}**. "
                "To restore this RAG context, upload these documents above and click 'Process Uploaded Documents for RAG', then ensure 'Enable RAG' is on."
            )
    with st.sidebar.expander("MCP Configuration", expanded=False):
        # MCP config JSON input - mcp_config now stores the full {"mcpServers": ...} structure
        mcp_config_input_val = json.dumps(st.session_state.mcp_config, indent=2) if st.session_state.mcp_config else \
            json.dumps({
                "mcpServers": {
                    "playwright": {
                        "command": "npx",
                        "args": ["@playwright/mcp@latest"],
                        "env": {"DISPLAY": ":1"}
                    }
                }
            }, indent=2)

        mcp_config_input_area = st.text_area(
            "MCP Server Config JSON:",
            value=mcp_config_input_val,
            height=200,
            help="Enter MCP server configuration in JSON format, e.g., {\"mcpServers\": {\"server_name\": {\"command\": ...}}}. This will be used to initialize MCPClient from mcp-use.",
            key="mcp_config_input"
        )
        if st.session_state.show_mcp_restore_info:
            st.info("MCP settings from the active chat history have been loaded into the controls above. Click 'Apply MCP Configuration' if you wish to use them.", icon="💡")
            st.session_state.show_mcp_restore_info = False # Clear the flag

        st.toggle("Enable MCP Servers/Agent", key="mcp_enabled_by_user", help="If enabled and an MCP Agent is configured, it will be used for processing requests (unless RAG is active).")

        if st.session_state.mcp_enabled_by_user and st.session_state.mcp_agent:
            st.info("MCP is ENABLED and Agent is active.", icon="🤖")
        elif st.session_state.mcp_enabled_by_user and not st.session_state.mcp_agent:
            st.warning("MCP is enabled by toggle, but MCPAgent is not active (ensure LLM and MCPClient are configured).", icon="⚠️")


        if st.button("Apply MCP Configuration"):
            try:
                loaded_mcp_json = json.loads(mcp_config_input_area)
                if not isinstance(loaded_mcp_json, dict) or "mcpServers" not in loaded_mcp_json:
                    st.error("MCP configuration must be a JSON object with a top-level 'mcpServers' key.")
                else:
                    st.session_state.mcp_config = loaded_mcp_json # Store the full config

                    # Attempt to close/stop the old client if it exists and has a close method
                    if st.session_state.mcp_client and hasattr(st.session_state.mcp_client, 'close'):
                        try:
                            if asyncio.iscoroutinefunction(st.session_state.mcp_client.close):
                                asyncio.run(st.session_state.mcp_client.close())
                            else:
                                st.session_state.mcp_client.close()
                            st.info("Previous MCPClient resources released.")
                        except Exception as e:
                            st.warning(f"Error closing previous MCPClient: {e}")
                    
                    st.session_state.mcp_client = MCPClient.from_dict(st.session_state.mcp_config)
                    num_servers = len(st.session_state.mcp_config.get("mcpServers", {}))
                    st.success(f"MCPClient initialized with {num_servers} server definition(s).")
                    try_initialize_mcp_agent() # Attempt to init agent after MCPClient is ready
                    st.rerun()

            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON for MCP config: {e}")
            except Exception as e: # Catch errors from MCPClient.from_dict or get_tools
                st.error(f"Error during MCPClient initialization: {e}")

        if not st.session_state.openai_client:
            st.error("LLM client is not initialized. Please configure and apply settings in the sidebar.")
        
        if st.session_state.mcp_client:
            if st.session_state.mcp_config and "mcpServers" in st.session_state.mcp_config:
                st.write("Configured MCP Servers (status managed by MCPClient):")
                for server_name in st.session_state.mcp_config["mcpServers"].keys():
                    st.write(f"  - {server_name}")
        elif not st.session_state.mcp_enabled_by_user:
            st.info("MCP is currently DISABLED by toggle.")
        else:
            st.info("MCPClient/Agent is not active. Configure and apply above.")

        if st.session_state.mcp_client or st.session_state.mcp_agent:
            if st.button("Stop MCP Client & Agent"):
                if st.session_state.mcp_client and hasattr(st.session_state.mcp_client, 'close'):
                    try:
                        if asyncio.iscoroutinefunction(st.session_state.mcp_client.close):
                            asyncio.run(st.session_state.mcp_client.close())
                        else:
                            st.session_state.mcp_client.close()
                        st.info("MCPClient resources released.")
                    except Exception as e:
                        st.warning(f"Error closing MCPClient: {e}")
                
                st.session_state.mcp_client = None
                st.session_state.mcp_agent = None
                # Optionally clear mcp_config: st.session_state.mcp_config = {}
                st.success("MCP Client and Agent have been stopped.")
                st.rerun()

# --- Main Chat Display Logic ---
# Display chat messages from current history
if st.session_state.current_history is not None and st.session_state.histories:
    # Ensure current_history index is valid before accessing
    if 0 <= st.session_state.current_history < len(st.session_state.histories):
        current_history_entry = st.session_state.histories[st.session_state.current_history]

        # Check if the history entry itself is valid and has 'messages'
        if not isinstance(current_history_entry, dict) or \
           "messages" not in current_history_entry or \
           not isinstance(current_history_entry.get("messages"), list):
            st.error(f"Chat history at index {st.session_state.current_history} is malformed or messages are not a list.")
        else:
            current_messages = current_history_entry["messages"]
            for i, message_data in enumerate(current_messages): # Renamed for clarity
                if not isinstance(message_data, dict):
                    st.warning(f"Skipping malformed message (expected dict, got {type(message_data)}) at index {i} in history {st.session_state.current_history}.")
                    continue

                unique_message_key_prefix = f"msg_{st.session_state.current_history}_{i}"
                with st.container(border=True):
                    role = message_data.get("role")
                    if role is None:
                        st.warning(f"Message at index {i} in history {st.session_state.current_history} is missing 'role'. Skipping display of this message part.")
                        continue

                    avatar = None
                    if role == "tool":
                        avatar = "🛠️" 
                    message_metadata = message_data.get("metadata", {})
                    
                    with st.chat_message(role if role != "tool" else "assistant", avatar=avatar):
                        content = message_data.get("content")
                        tool_calls = message_data.get("tool_calls")

                        if role == "tool":
                            st.markdown(f"**Tool Response ({message_data.get('name', 'N/A')})**:")
                            # Display tool metadata if available
                            if message_metadata.get("tool_executor"):
                                st.caption(f"Executed by: {message_metadata['tool_executor']}")
                            st.code(content if content is not None else "No content provided by tool.", language="json" if isinstance(content, str) and content.startswith(("{","[")) else "text")
                        
                        elif tool_calls and isinstance(tool_calls, list): # Ensure tool_calls is a list
                            if content: 
                                st.markdown(content)
                            for tc_idx, tc in enumerate(tool_calls):
                                if isinstance(tc, dict) and isinstance(tc.get("function"), dict):
                                    func_name = tc.get("function", {}).get("name", "N/A")
                                    func_args = tc.get("function", {}).get("arguments", "N/A")
                                    st.markdown(f"Assistant wants to use a tool (`{func_name}`):")
                                    # Display tool call metadata if available
                                    if message_metadata.get("tool_caller_type"):
                                        st.caption(f"Caller Type: {message_metadata['tool_caller_type']}")
                                    st.code(f"Arguments: {func_args}", language="json")

                                else:
                                    st.warning(f"Malformed tool_call entry in message {i}, tool_call index {tc_idx}.")
                        elif content is not None:
                            st.markdown(content)
                        elif role == "assistant" and message_data.get("refusal"):
                            st.markdown(f"*Assistant refused to answer: {message_data.get('refusal')}*")
                        elif role == "assistant" and content is None and not tool_calls:
                            st.markdown("*Assistant provided no textual response or tool calls.*")

                        # Display common metadata for user and assistant messages
                        if role in ["user", "assistant"]:
                            meta_parts = []
                            if message_metadata.get("source"):
                                meta_parts.append(f"Source: {message_metadata['source']}")
                            if message_metadata.get("rag_details"):
                                rag_details_dict = message_metadata['rag_details']
                                # Attempt to parse if it's a string (for backward compatibility or if saved as string)
                                if isinstance(rag_details_dict, str):
                                    try: rag_details_dict = json.loads(rag_details_dict)
                                    except json.JSONDecodeError: rag_details_dict = {"raw_string": rag_details_dict} # Fallback
                                if isinstance(rag_details_dict, dict): # Proceed if it's a dict
                                    rag_status_parts = []
                                    toggle_stat = rag_details_dict.get('toggle_status', 'Unknown')
                                    rag_status_parts.append(f"RAG Toggle: {toggle_stat.capitalize()}")
                                    if toggle_stat == "enabled":
                                        context_stat = rag_details_dict.get('context_status', 'N/A')
                                        rag_status_parts.append(f"Context: {context_stat.replace('_', ' ').title()}")
                                        if rag_details_dict.get("indexed_documents"):
                                            docs_str = ", ".join(rag_details_dict["indexed_documents"])
                                            rag_status_parts.append(f"Indexed Docs: [{docs_str if docs_str else 'None'}]")
                                        if "chunks_retrieved" in rag_details_dict:
                                            rag_status_parts.append(f"Chunks: {rag_details_dict['chunks_retrieved']}")
                                    meta_parts.append("RAG (" + " | ".join(rag_status_parts) + ")")
                            if message_metadata.get("processing_conflict_note"):
                                meta_parts.append(f"Note: {message_metadata['processing_conflict_note'].replace('_', ' ').title()}")
                            if meta_parts:
                                st.caption(" | ".join(meta_parts))

                    # Action buttons
                    cols = st.columns((1, 1, 12)) # Delete, Redo, Spacer - increased spacer ratio for tighter button grouping
                    with cols[1]:
                        if st.button("🗑️", key=f"{unique_message_key_prefix}_delete", help="Delete this message"):
                            del st.session_state.histories[st.session_state.current_history]["messages"][i]
                            st.rerun()
                    with cols[2]:
                        # Redo button is only shown for user messages.
                        if message_data.get("role") == "user":
                            if st.button("🔄", key=f"{unique_message_key_prefix}_redo", help="Redo from this point"):
                                user_msg_idx_for_redo = -1
                                # Find the most recent user message at or before current message 'i'
                                # Since this button is now only on user messages, i will be the user_msg_idx_for_redo.
                                for j_redo in range(i, -1, -1):
                                    # Ensure message being checked is a dict and has a role
                                    msg_to_check_for_redo = current_messages[j_redo]
                                    if isinstance(msg_to_check_for_redo, dict) and msg_to_check_for_redo.get("role") == "user":
                                        if msg_to_check_for_redo.get("content") is not None: # Ensure there's content to redo
                                            user_msg_idx_for_redo = j_redo
                                            break
                                
                                if user_msg_idx_for_redo != -1:
                                    # Content is already verified to exist above
                                    content_to_redo = current_messages[user_msg_idx_for_redo].get("content")
                                    # Truncate history up to (but not including) the user message we are redoing
                                    st.session_state.histories[st.session_state.current_history]["messages"] = \
                                        current_messages[:user_msg_idx_for_redo] # Use st.session_state.current_history
                                    
                                    st.session_state.force_process_prompt = content_to_redo
                                    st.rerun()
                                else:
                                    # This case should ideally not be reached if button is only on user messages with content.
                                    st.toast("Could not find a user message to redo.", icon="⚠️")

# --- Tool Definitions ---
def call_rest_api(method: str, url: str, headers: dict = None, params: dict = None, json_payload: dict = None) -> str:
    """
    Makes an HTTP request to a given REST API endpoint and returns the response text or JSON.

    Args:
        method (str): HTTP method (GET, POST, PUT, DELETE, etc.).
        url (str): The URL of the API endpoint.
        headers (dict, optional): Dictionary of HTTP headers.
        params (dict, optional): Dictionary of URL parameters for GET requests.
        json_payload (dict, optional): JSON payload for POST/PUT requests.

    Returns:
        str: The response text from the API, or an error message.
    """
    try:
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            params=params,
            json=json_payload,
            timeout=15  # Set a reasonable timeout
        )
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        # Try to return JSON if possible, otherwise text
        try:
            return json.dumps(response.json(), indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            return response.text
    except Exception as e:
        return f"An unexpected error occurred while calling API: {e}"

# Schema for the REST API tool (if used directly, or as a reference for mcp-use tool structure)
rest_api_tool_definition = { # This is a Pydantic model schema, not a BaseTool instance
    "type": "function",
    "function": {
        "name": "call_rest_api",
        "description": "Makes HTTP requests (e.g., GET, POST, PUT, DELETE) to a specified REST API endpoint. Use this to fetch live data, interact with external services, or trigger actions. If the API requires authentication, such as an API key or a Bearer token, include it in the 'headers' parameter. For example, to get data, use GET. To send data, use POST or PUT.",
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "HTTP method (GET, POST, PUT, DELETE, etc.).",
                    "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
                },
                "url": {
                    "type": "string",
                    "description": "The URL of the API endpoint."
                },
                "headers": {
                    "type": "object",
                    "description": "Headers to include in the API request.",
                    "additionalProperties": {
                        "type": "string"
                    }
                },
                "params": {
                    "type": "object",
                    "description": "Query parameters for the API request.",
                    "additionalProperties": {
                        "type": "string"
                    }
                },
                "json_payload": {
                    "type": "object",
                    "description": "JSON payload for POST/PUT requests.",
                    "additionalProperties": {
                        "type": "string"
                    }
                }
            },
            "required": ["method", "url"]
        }
    }
}

# Mapping of tool names to their functions
available_tools_mapping = {
    "call_rest_api": call_rest_api, # Temporarily commented out
} # This mapping is for non-MCP tools
# Accept user input
prompt_from_redo = st.session_state.pop("force_process_prompt", None)
prompt_from_chat_input = st.chat_input("What is your question?")

final_prompt_to_process = None
if prompt_from_redo:
    final_prompt_to_process = prompt_from_redo
elif prompt_from_chat_input:
    final_prompt_to_process = prompt_from_chat_input

if final_prompt_to_process:
    if not st.session_state.openai_client:
        st.error("LLM client is not initialized. Please configure and apply settings in the sidebar.")
    else:
        # Add user message to current history
        active_history_idx = st.session_state.current_history
        current_hist_valid_for_prompt = active_history_idx is not None and \
                                     0 <= active_history_idx < len(st.session_state.histories) # type: ignore

        if current_hist_valid_for_prompt:
            st.session_state.histories[active_history_idx]["messages"].append(
                {
                    "role": "user", 
                    "content": final_prompt_to_process,
                    "metadata": {"source": "user_input"}
                }
            )
            # Display of this user message will be handled by the main message display loop on st.rerun or at end of script

        langchain_conversation_messages: list[BaseMessage] = []
        
        def _augment_prompt_with_rag_if_possible(query_string: str) -> tuple[str, dict]:
            """
            Attempts to augment the given query_string with RAG context.
            Returns the (potentially augmented) query string and a boolean indicating if RAG was active.
            """
            if st.session_state.rag_enabled_by_user and st.session_state.retriever: # Check toggle and retriever
                with st.spinner("Retrieving relevant documents for RAG..."):
                    retrieved_docs = st.session_state.retriever.invoke(query_string)
                rag_details_dict = {"toggle_status": "enabled", "context_status": "retriever_active"}
                if retrieved_docs:
                    context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    augmented_content = (
                        f"Please answer the following question based on the provided context.\n\n"
                        f"Context:\n{context_str}\n\n"
                        f"Question: {query_string}"
                    )
                    # Use the list of all currently processed document names for rag_details
                    rag_doc_names_for_metadata = list(st.session_state.processed_doc_names) # Make a copy
                    rag_toast_message = f"ℹ️ RAG: Used {len(retrieved_docs)} chunks from indexed documents."
                    st.toast(rag_toast_message, icon="📄")
                    rag_details_dict["context_status"] = "found"
                    rag_details_dict["indexed_documents"] = rag_doc_names_for_metadata
                    rag_details_dict["chunks_retrieved"] = len(retrieved_docs)
                    return augmented_content, rag_details_dict
                else: # Retriever exists, but no docs found for this query
                    st.toast("ℹ️ No relevant documents found for RAG. Using original prompt.", icon="📄")
                    rag_details_dict["context_status"] = "not_found"
                    rag_details_dict["indexed_documents"] = list(st.session_state.processed_doc_names) # Still show what was indexed
                    rag_details_dict["chunks_retrieved"] = 0
                    return query_string, rag_details_dict
            elif st.session_state.rag_enabled_by_user and not st.session_state.retriever:
                return query_string, {"toggle_status": "enabled", "context_status": "no_retriever"}
            return query_string, {"toggle_status": "disabled"} # RAG toggle is off

        rag_active_for_this_prompt = False 
        # Add conversation history from the current active chat
        if current_hist_valid_for_prompt:
            history_messages_to_convert = st.session_state.histories[active_history_idx]["messages"]
            for i, msg_dict in enumerate(history_messages_to_convert):
                is_last_message = (i == len(history_messages_to_convert) - 1)
                current_message_content = msg_dict.get("content", "")
                if msg_dict.get("role") == "user" and is_last_message:
                    # This is the current user prompt, try to augment it for RAG
                    augmented_content, rag_info_dict = _augment_prompt_with_rag_if_possible(current_message_content)
                    human_message_metadata = {"source": "user_input_processed"}
                    human_message_metadata["rag_details"] = rag_info_dict # Always add rag_info_dict
                    langchain_conversation_messages.append(
                        HumanMessage(content=augmented_content, metadata=human_message_metadata)
                    )
                    # RAG is active for the prompt if the toggle was on and context was actually found
                    if rag_info_dict.get("toggle_status") == "enabled" and rag_info_dict.get("context_status") == "found":
                        rag_active_for_this_prompt = True
                else:
                    langchain_conversation_messages.append(convert_dict_to_langchain_message(msg_dict))
        else:
            # If no valid/active history, send only the current prompt.
            # Attempt RAG augmentation for the standalone prompt
            augmented_content, rag_info_dict = _augment_prompt_with_rag_if_possible(final_prompt_to_process)
            human_message_metadata = {"source": "user_input_standalone"}
            human_message_metadata["rag_details"] = rag_info_dict # Always add rag_info_dict
            langchain_conversation_messages.append(
                HumanMessage(content=augmented_content, metadata=human_message_metadata)
            )
            if rag_info_dict.get("toggle_status") == "enabled" and rag_info_dict.get("context_status") == "found":
                rag_active_for_this_prompt = True

        def _handle_direct_llm_call(
            current_lc_messages: list[BaseMessage],
            hist_valid: bool,
            current_active_hist_idx: int | None,
            is_rag_call: bool = False,
            source_description: str = "llm_direct"
        ) -> None:
            """Handles direct LLM invocation, including tool binding and tool call loop."""
            bound_llm = st.session_state.openai_client # type: ignore
            # Conditionally add non-MCP tools.
            if st.session_state.current_provider not in ["sambanova", "ollama"]:
                if bound_llm:
                    tools_to_bind = []
                    if "call_rest_api" in available_tools_mapping: # Only bind non-MCP tools
                        tools_to_bind.append(rest_api_tool_definition)
                    
                    if tools_to_bind: 
                        bound_llm = bound_llm.bind_tools(tools_to_bind)

            spinner_text = "Thinking with RAG context..." if is_rag_call else "Thinking..."
            with st.spinner(spinner_text):
                if not bound_llm:
                    st.error("LLM client is not properly initialized.")
                    st.stop()
                # Type: ignore - response will always be AIMessage when using ChatOpenAI
                response_aimessage = bound_llm.invoke(current_lc_messages)
            
            if not response_aimessage or (not response_aimessage.content and not response_aimessage.tool_calls):
                error_message_ui = "LLM response was empty or did not contain content or tool calls..."
                st.error(error_message_ui)
                if hist_valid and current_active_hist_idx is not None:
                    error_metadata = {"source": source_description, "error": "empty_response"}
                    # Try to get rag_details from the input HumanMessage if RAG was intended
                    # The last message in current_lc_messages is the one that would have been augmented
                    relevant_human_message_for_rag_details = current_lc_messages[-1] if current_lc_messages and isinstance(current_lc_messages[-1], HumanMessage) else None
                    if is_rag_call and relevant_human_message_for_rag_details and \
                       hasattr(relevant_human_message_for_rag_details, 'metadata') and relevant_human_message_for_rag_details.metadata.get('rag_details'):
                        error_metadata["rag_details"] = current_lc_messages[0].metadata['rag_details']
                    st.session_state.histories[current_active_hist_idx]["messages"].append(
                        {
                            "role": "assistant", 
                            "content": "[Error: LLM returned no valid response content or tool calls]",
                            "metadata": error_metadata
                        }
                    )
                st.rerun(); st.stop()

            parsed_tool_calls = response_aimessage.tool_calls

            if parsed_tool_calls:
                if hist_valid and current_active_hist_idx is not None:
                    ai_message_metadata = {"source": source_description, "tool_caller_type": "llm_direct"}
                    relevant_human_message_for_rag_details = current_lc_messages[-2] if len(current_lc_messages) > 1 and isinstance(current_lc_messages[-2], HumanMessage) else None # -2 because AIMessage was just appended
                    if is_rag_call and relevant_human_message_for_rag_details and \
                       hasattr(relevant_human_message_for_rag_details, 'metadata') and relevant_human_message_for_rag_details.metadata.get('rag_details'):
                         ai_message_metadata["rag_details"] = relevant_human_message_for_rag_details.metadata['rag_details']
                    response_aimessage.metadata = ai_message_metadata # Add metadata to AIMessage object
                    st.session_state.histories[current_active_hist_idx]["messages"].append(
                        convert_aimessage_to_storage_dict(response_aimessage) # This will now store metadata
                    )
                current_lc_messages.append(response_aimessage)

                for tool_call_data in parsed_tool_calls:
                    function_name = tool_call_data["name"]
                    function_to_call = available_tools_mapping.get(function_name)
                    if function_to_call:
                        try:
                            if function_name == "call_rest_api": st.toast(f"Attempting to use tool: {function_name}", icon="🚀")
                            function_args = tool_call_data["args"]; tool_output = function_to_call(**function_args)
                        except Exception as e: tool_output = f"Error parsing arguments or calling tool {function_name}: {e}"
                    else: tool_output = f"Error: Tool '{function_name}' not found in available_tools_mapping."
                    tool_message_for_storage = {
                        "role": "tool", 
                        "tool_call_id": tool_call_data["id"], 
                        "name": function_name, 
                        "content": tool_output,
                        "metadata": {"tool_executor": "chitchat_manual_tool"}
                    }
                    if hist_valid and current_active_hist_idx is not None: st.session_state.histories[current_active_hist_idx]["messages"].append(tool_message_for_storage)
                    current_lc_messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call_data["id"]))
                with st.spinner("Processing tool results..."):
                    if not bound_llm: st.error("LLM client became uninitialized."); st.stop()
                    second_response_aimessage = bound_llm.invoke(current_lc_messages)
                if hist_valid and current_active_hist_idx is not None:
                    second_ai_message_metadata = {"source": source_description, "after_tool_call": True}
                    # Find the original HumanMessage that might have RAG details
                    original_human_message_for_rag = next((m for m in reversed(current_lc_messages) if isinstance(m, HumanMessage) and hasattr(m, 'metadata') and m.metadata.get('rag_details')), None)
                    if is_rag_call and original_human_message_for_rag:
                         second_ai_message_metadata["rag_details"] = original_human_message_for_rag.metadata['rag_details']
                    second_response_aimessage.metadata = second_ai_message_metadata
                    st.session_state.histories[current_active_hist_idx]["messages"].append(convert_aimessage_to_storage_dict(second_response_aimessage))
            else: # No tool calls
                if response_aimessage.content and hist_valid and current_active_hist_idx is not None:
                    ai_message_metadata = {"source": source_description}
                    relevant_human_message_for_rag_details = current_lc_messages[-1] if current_lc_messages and isinstance(current_lc_messages[-1], HumanMessage) else None
                    if is_rag_call and relevant_human_message_for_rag_details and \
                       hasattr(relevant_human_message_for_rag_details, 'metadata') and relevant_human_message_for_rag_details.metadata.get('rag_details'):
                        ai_message_metadata["rag_details"] = relevant_human_message_for_rag_details.metadata['rag_details']
                    if is_rag_call and st.session_state.mcp_enabled_by_user: # Both RAG and MCP were enabled
                        ai_message_metadata["processing_conflict_note"] = "rag_mcp_conflict_rag_precedence"
                    response_aimessage.metadata = ai_message_metadata
                    st.session_state.histories[current_active_hist_idx]["messages"].append(convert_aimessage_to_storage_dict(response_aimessage))

        try:
            if not st.session_state.current_model:
                    st.error("Please set a valid model name in the sidebar.")
                    st.stop()
            elif st.session_state.rag_enabled_by_user and rag_active_for_this_prompt: # RAG toggle is ON and RAG context was found
                st.toast("Using LLM directly with RAG context.", icon="📄")
                _handle_direct_llm_call(langchain_conversation_messages, current_hist_valid_for_prompt, active_history_idx, is_rag_call=True, source_description="llm_direct_with_rag")
            elif st.session_state.mcp_enabled_by_user and st.session_state.mcp_agent : # MCP Toggle ON and MCPAgent is active (and RAG was not used for this prompt)
                with st.spinner("Agent is working..."):
                    try:
                        st.toast("Using Agent to process your request.", icon="🤖")
                        # Create fresh client and agent instance for each request
                        fresh_client = MCPClient.from_dict(st.session_state.mcp_config)
                        fresh_agent = MCPAgent(
                            llm=st.session_state.openai_client,
                            client=fresh_client,
                            memory_enabled=False,
                            max_steps=10
                        )
                        agent_response_text = asyncio.run(fresh_agent.run(final_prompt_to_process))
                        # Explicitly close the client after use
                        if hasattr(fresh_client, 'close'):
                            if asyncio.iscoroutinefunction(fresh_client.close):
                                asyncio.run(fresh_client.close())
                            else:
                                fresh_client.close()
                        
                        if current_hist_valid_for_prompt:
                            st.session_state.histories[active_history_idx]["messages"].append(
                                {
                                    "role": "assistant", 
                                    "content": agent_response_text,
                                    "metadata": {"source": "mcp_agent"}
                                }
                            )
                    except Exception as e: # Catch more general exceptions from MCPAgent
                        detailed_error_msg = f"MCPAgent Error: {type(e).__name__} - {e}"
                        st.error(detailed_error_msg)
                        print("--- MCPAgent Execution Error Traceback ---")
                        traceback.print_exc() # Print full traceback to console
                        print("-----------------------------------------")
                        if current_hist_valid_for_prompt:
                             st.session_state.histories[active_history_idx]["messages"].append(
                                {
                                    "role": "assistant", 
                                    "content": f"[Error during MCPAgent execution: {type(e).__name__} - {e}]",
                                    "metadata": {"source": "mcp_agent", "error": str(e)}
                                }
                            )
            else: # Fallback: RAG not used for this prompt AND (MCP toggle OFF OR MCPAgent not active) -> Direct LLM
                st.toast("Using LLM directly to process your request.", icon="🧠") # Adjusted icon for clarity
                _handle_direct_llm_call(langchain_conversation_messages, current_hist_valid_for_prompt, active_history_idx, is_rag_call=False, source_description="llm_direct_no_rag_no_mcp")
            
            # Rerun to display all new messages, including tool interactions
            st.rerun()
        except Exception as e: # General fallback for non-OpenAI errors or unexpected issues
            st.error(f"An unexpected error occurred: {e}")
