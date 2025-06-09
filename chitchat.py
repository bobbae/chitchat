#!/usr/bin/env python3
import os
import requests # Add this import at the top
import json
import datetime
import openai # Import the openai library for its exception types
import tempfile # For handling uploaded PDF files
import subprocess # For running MCP servers
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage, ToolCall
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from io import BytesIO

DEFAULT_PROVIDER_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
    "sambanova": "https://api.sambanova.ai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/"
}

PROVIDER_API_KEY_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_API_KEY", # For a compatible endpoint
    "sambanova": "SAMBANOVA_API_KEY"
}

DEFAULT_PROVIDER_MODELS = {
    "openai": "gpt-4o", 
    "ollama": "qwen3:8b", 
    "openrouter": "meta-llama/llama-4-maverick:free",
    "sambanova": "DeepSeek-R1",
    "gemini": "gemini-1.5-flash" #"gemini-2.5-pro-preview-05-06" 
}

st.set_page_config(page_title="Chit-Chat with Models", layout="wide")


def initialize_client(provider, model_name, api_key_input, base_url_input):
    """Initializes the OpenAI client based on selected provider and settings."""
    current_api_key = api_key_input
    provider_env_var_name = PROVIDER_API_KEY_ENV_VARS.get(provider)

    if not current_api_key and provider_env_var_name:
        current_api_key = os.environ.get(provider_env_var_name)

    if provider == "ollama":
        if not current_api_key:
            current_api_key = "ollama" # Placeholder for Ollama
            st.sidebar.info("Using placeholder API key for Ollama.")
    elif not current_api_key:
        st.sidebar.error(f"API key for {provider} is missing. Please set it in the sidebar or as an environment variable ({provider_env_var_name}).")
        return None

    current_base_url = base_url_input
    if not current_base_url:
        current_base_url = DEFAULT_PROVIDER_URLS.get(provider)

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
        st.sidebar.success(f"Client initialized for {provider} with model {model_name} at {current_base_url}")
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
if "mcp_config" not in st.session_state: # For MCP servers
    st.session_state.mcp_config = {}
if "mcp_servers" not in st.session_state: # Active MCP server processes
    st.session_state.mcp_servers = {}

# Early History State Update Block:
# This runs before the title is rendered to ensure provider/model reflect the selected history
# if a history activation is pending. The full client (re)initialization and reset of
# pending_history_activation_index will still happen in the sidebar's main activation logic.
_pending_idx = st.session_state.get("pending_history_activation_index")
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
            # Note: We do NOT reset pending_history_activation_index or call initialize_client here.
            # Those are handled by the main activation logic in the sidebar.

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
        return AIMessage(content=content, tool_calls=lc_tool_calls if lc_tool_calls else [])
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
            "timestamp": hist_to_load.get("timestamp", datetime.datetime.now().isoformat())
        }

        if existing_history_index != -1:
            st.session_state.histories[existing_history_index] = history_entry_data
            updated_count += 1
        else:
            st.session_state.histories.append(history_entry_data)
            loaded_count += 1
    
    if loaded_count > 0 or updated_count > 0:
        st.success(f"Successfully loaded {loaded_count} new and updated {updated_count} existing chat histories from '{source_name}'. They are now available in the 'Chat Histories' dropdown.")
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

        else:
            st.error(f"Attempted to activate invalid history index: {new_active_idx}")

        st.session_state.pending_history_activation_index = None # Reset the trigger

    st.header("Model Configuration")

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
            if st.session_state.openai_client:
                # Check if a history with the same provider and model already exists
                existing_history_index = -1
                for i, hist in enumerate(st.session_state.histories):
                    if hist["provider"] == st.session_state.current_provider and \
                       hist["model"] == st.session_state.current_model:
                        existing_history_index = i
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
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    st.session_state.histories.append(new_history)
                    st.session_state.current_history = len(st.session_state.histories) - 1
                    st.sidebar.success("New chat history created. Configuration applied successfully. Ready to chat!")
                st.rerun() # Rerun to update the main page title and reflect connection status
            else:
                 st.sidebar.error("LLM Client connection failed. Please check settings in sidebar and console.")


    st.markdown("---")
    st.subheader("Chat History Management")
    
    # History selection
    if st.session_state.histories:
        history_names = [
            f"{hist['provider']} - {hist['model']} ({len(hist['messages'])} messages)" 
            for hist in st.session_state.histories
        ]
        selected_hist = st.selectbox(
            "Chat Histories",
            options=range(len(st.session_state.histories)),
            format_func=lambda x: history_names[x],
            index=st.session_state.current_history or 0
        )
        
        if selected_hist != st.session_state.current_history:
            # Set pending activation and rerun
            st.session_state.pending_history_activation_index = selected_hist
            # Client initialization and state updates will happen at the top of the sidebar in the next run
            st.rerun()

        # Button to clear the currently selected chat history
        if st.button("Clear Selected Chat History", key="clear_selected_chat_history_sidebar_button"):
            if st.session_state.current_history is not None and 0 <= st.session_state.current_history < len(st.session_state.histories):
                st.session_state.histories[st.session_state.current_history]["messages"] = []
            st.rerun()

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
        history_json = json.dumps(st.session_state.histories, indent=2)
        st.download_button(
            label="Save All History",
            data=history_json,
            file_name="chat_histories.json",
            mime="application/json",
        )

    st.markdown("---")
    st.subheader("RAG - Document Q&A")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT) for RAG",
        accept_multiple_files=True,
        type=['pdf', 'txt'],
        key="rag_file_uploader"
    )
    if st.button("Process Uploaded Documents for RAG"):
        process_documents_for_rag(uploaded_files)

    if st.session_state.processed_doc_names:
        st.write("Currently indexed documents for RAG:")
        for name in st.session_state.processed_doc_names:
            st.caption(f"- {name}")
        if st.button("Clear RAG Index & Processed Documents"):
            st.session_state.retriever = None
            st.session_state.processed_doc_names = []
            st.rerun()

    st.markdown("---")
    st.subheader("MCP Server Configuration")
    
    # MCP config JSON input
    mcp_config_input = st.text_area(
        "MCP Server Config (JSON):",
        value=json.dumps(st.session_state.mcp_config, indent=2) if st.session_state.mcp_config else '{\n  "example-server": {\n    "command": "npx",\n    "args": ["-y", "@modelcontextprotocol/server-example"]\n  }\n}',
        height=150,
        help="Enter MCP server configuration in JSON format. Each server should have a 'command' and optional 'args' array."
    )
    
    if st.button("Apply MCP Configuration"):
        try:
            mcp_config = json.loads(mcp_config_input)
            if not isinstance(mcp_config, dict):
                st.error("MCP configuration must be a JSON object")
            else:
                st.session_state.mcp_config = mcp_config
                st.success(f"MCP configuration updated with {len(mcp_config)} server(s)")
                
                # Stop any existing MCP servers
                for server_name, process in st.session_state.mcp_servers.items():
                    if process and process.poll() is None:
                        process.terminate()
                        st.info(f"Stopped MCP server: {server_name}")
                st.session_state.mcp_servers = {}
                
                # Start configured MCP servers
                for server_name, server_config in mcp_config.items():
                    if isinstance(server_config, dict) and "command" in server_config:
                        try:
                            command = [server_config["command"]]
                            if "args" in server_config and isinstance(server_config["args"], list):
                                command.extend(server_config["args"])
                            
                            # Start the MCP server process
                            process = subprocess.Popen(
                                command,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True
                            )
                            st.session_state.mcp_servers[server_name] = process
                            st.success(f"Started MCP server: {server_name}")
                        except Exception as e:
                            st.error(f"Failed to start MCP server '{server_name}': {e}")
                    else:
                        st.warning(f"Invalid configuration for server '{server_name}'")
                        
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
    
    # Display active MCP servers
    if st.session_state.mcp_servers:
        st.write("Active MCP Servers:")
        for server_name, process in st.session_state.mcp_servers.items():
            if process and process.poll() is None:
                st.caption(f"✅ {server_name} (PID: {process.pid})")
            else:
                st.caption(f"❌ {server_name} (stopped)")
    
    if st.button("Stop All MCP Servers"):
        for server_name, process in st.session_state.mcp_servers.items():
            if process and process.poll() is None:
                process.terminate()
                st.info(f"Stopped MCP server: {server_name}")
        st.session_state.mcp_servers = {}
        st.rerun()

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
                    
                    with st.chat_message(role if role != "tool" else "assistant", avatar=avatar):
                        content = message_data.get("content")
                        tool_calls = message_data.get("tool_calls")

                        if role == "tool":
                            st.markdown(f"**Tool Response ({message_data.get('name', 'N/A')})**:")
                            st.code(content if content is not None else "No content provided by tool.", language="json")
                        elif tool_calls and isinstance(tool_calls, list): # Ensure tool_calls is a list
                            if content: 
                                st.markdown(content)
                            for tc_idx, tc in enumerate(tool_calls):
                                if isinstance(tc, dict) and isinstance(tc.get("function"), dict):
                                    func_name = tc.get("function", {}).get("name", "N/A")
                                    func_args = tc.get("function", {}).get("arguments", "N/A")
                                    st.markdown(f"Assistant wants to use a tool (`{func_name}`):")
                                    st.code(f"Arguments: {func_args}", language="json")
                                else:
                                    st.warning(f"Malformed tool_call entry in message {i}, tool_call index {tc_idx}.")
                        elif content is not None:
                            st.markdown(content)
                        elif role == "assistant" and message_data.get("refusal"):
                            st.markdown(f"*Assistant refused to answer: {message_data.get('refusal')}*")
                        elif role == "assistant" and content is None and not tool_calls:
                            st.markdown("*Assistant provided no textual response or tool calls.*")

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
def call_mcp_server(server_name: str, method: str, params: dict = None) -> str:
    """
    Calls a configured MCP server with a specific method and parameters.
    
    Args:
        server_name (str): Name of the MCP server as configured
        method (str): The method/tool to call on the MCP server
        params (dict, optional): Parameters to pass to the method
        
    Returns:
        str: The response from the MCP server or an error message
    """
    try:
        if server_name not in st.session_state.mcp_servers:
            return f"Error: MCP server '{server_name}' is not configured or not running"
        
        process = st.session_state.mcp_servers[server_name]
        if process.poll() is not None:
            return f"Error: MCP server '{server_name}' is not running"
        
        # Create MCP request
        mcp_request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1
        }
        
        # Send request to MCP server (this is a simplified example)
        # In a real implementation, you would need proper IPC with the MCP server
        # For now, we'll return a placeholder response
        return f"MCP call to '{server_name}.{method}' with params {params} (Note: Full MCP protocol implementation needed)"
        
    except Exception as e:
        return f"Error calling MCP server: {e}"

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
            return json.dumps(response.json(), indent=2)
        except json.JSONDecodeError:
            return response.text
    except requests.exceptions.HTTPError as http_err:
        error_message = f"HTTP error occurred: {http_err}"
        if http_err.response is not None:
            error_message += f" - Status Code: {http_err.response.status_code} - Response: {http_err.response.text}"
        return error_message
    except requests.exceptions.ConnectionError as conn_err:
        return f"Connection error occurred: {conn_err}"
    except requests.exceptions.Timeout as timeout_err:
        return f"Timeout error occurred: {timeout_err}"
    except requests.exceptions.RequestException as req_err:
        return f"An error occurred during the API request: {req_err}"
    except Exception as e:
        return f"An unexpected error occurred while calling API: {e}"

# Schema for MCP server tool
mcp_tool_definition = {
    "type": "function",
    "function": {
        "name": "call_mcp_server",
        "description": "Calls a configured MCP (Model Context Protocol) server to execute tools or retrieve context. Use this to interact with MCP-enabled services that provide specialized capabilities.",
        "parameters": {
            "type": "object",
            "properties": {
                "server_name": {
                    "type": "string",
                    "description": "The name of the MCP server as configured in the sidebar"
                },
                "method": {
                    "type": "string",
                    "description": "The method/tool name to call on the MCP server"
                },
                "params": {
                    "type": "object",
                    "description": "Optional parameters to pass to the MCP server method",
                    "additionalProperties": True
                }
            },
            "required": ["server_name", "method"]
        }
    }
}

# Schema for the LLM to understand the tool
rest_api_tool_definition = {
    "type": "function",
    "function": {
        "name": "call_rest_api",
        "description": "Makes HTTP requests (e.g., GET, POST, PUT, DELETE) to a specified REST API endpoint. Use this to fetch live data, interact with external services, or trigger actions. If the API requires authentication, such as an API key or a Bearer token, include it in the 'headers' parameter. For example, to get data, use GET. To send data, use POST or PUT.",
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "The HTTP method (e.g., 'GET', 'POST', 'PUT', 'DELETE').",
                    "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]
                },
                "url": {
                    "type": "string",
                    "description": "The full URL of the API endpoint to call."
                },
                "headers": {
                    "type": "object",
                    "description": "Optional. A dictionary of HTTP headers. Example: {\"Authorization\": \"Bearer YOUR_TOKEN\", \"Content-Type\": \"application/json\"}.",
                    "additionalProperties": {"type": "string"}
                },
                "params": {
                    "type": "object",
                    "description": "Optional. For GET requests, a dictionary of URL parameters. Example: {\"search_query\": \"entries\", \"limit\": \"10\"}.",
                    "additionalProperties": {"type": "string"}
                },
                "json_payload": {
                    "type": "object",
                    "description": "Optional. For POST, PUT, PATCH requests, a JSON serializable dictionary for the request body. Example: {\"name\": \"new_entity\", \"status\": \"active\"}.",
                }
            },
            "required": ["method", "url"]
        }
    }
}
# Mapping of tool names to their functions
available_tools_mapping = {
    "call_rest_api": call_rest_api,
    "call_mcp_server": call_mcp_server,
}
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
                {"role": "user", "content": final_prompt_to_process}
            )
            # Display of this user message will be handled by the main message display loop on st.rerun or at end of script

        langchain_conversation_messages: list[BaseMessage] = []

        # Add conversation history from the current active chat
        if current_hist_valid_for_prompt:
            history_messages_to_convert = st.session_state.histories[active_history_idx]["messages"]
            for i, msg_dict in enumerate(history_messages_to_convert):
                is_last_message = (i == len(history_messages_to_convert) - 1)
                current_message_content = msg_dict.get("content", "")

                if msg_dict.get("role") == "user" and is_last_message and st.session_state.retriever:
                    # This is the current user prompt, try to augment it for RAG
                    with st.spinner("Retrieving relevant documents for RAG..."):
                        retrieved_docs = st.session_state.retriever.invoke(current_message_content)
                    
                    if retrieved_docs:
                        context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
                        augmented_content = (
                            f"Please answer the following question based on the provided context.\n\n"
                            f"Context:\n{context_str}\n\n"
                            f"Question: {current_message_content}"
                        )
                        langchain_conversation_messages.append(HumanMessage(content=augmented_content))
                        st.toast(f"ℹ️ Augmented prompt with context from {len(retrieved_docs)} document chunks.", icon="📄")
                    else:
                        langchain_conversation_messages.append(HumanMessage(content=current_message_content))
                        st.toast("ℹ️ No relevant documents found for RAG. Using original prompt.", icon="📄")
                else:
                    langchain_conversation_messages.append(convert_dict_to_langchain_message(msg_dict))
        
        else:
            # If no valid/active history, send only the current prompt.
            if st.session_state.retriever and final_prompt_to_process:
                with st.spinner("Retrieving relevant documents for RAG..."):
                    retrieved_docs = st.session_state.retriever.invoke(final_prompt_to_process)
                if retrieved_docs:
                    context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    augmented_content = (
                        f"Please answer the following question based on the provided context.\n\n"
                        f"Context:\n{context_str}\n\n"
                        f"Question: {final_prompt_to_process}"
                    )
                    langchain_conversation_messages.append(HumanMessage(content=augmented_content))
                    st.toast(f"ℹ️ Augmented prompt with context from {len(retrieved_docs)} document chunks.", icon="📄")
                else:
                    langchain_conversation_messages.append(HumanMessage(content=final_prompt_to_process))
                    st.toast("ℹ️ No relevant documents found for RAG. Using original prompt.", icon="📄")
            else:
                langchain_conversation_messages.append(HumanMessage(content=final_prompt_to_process))
        try:
            if not st.session_state.current_model:
                st.error("Model name is currently empty. Please set a valid model name in the sidebar.")
                st.stop() # Stop execution for this run if model name is missing

            bound_llm = st.session_state.openai_client
            # Conditionally add tools and tool_choice.
            if st.session_state.current_provider not in ["sambanova", "ollama"]:
                if bound_llm: # Ensure client is not None
                    tools_to_bind = [rest_api_tool_definition]
                    # Add MCP tool if MCP servers are configured
                    if st.session_state.mcp_servers:
                        tools_to_bind.append(mcp_tool_definition)
                    bound_llm = bound_llm.bind_tools(tools_to_bind)

            with st.spinner("Thinking..."):
                if not bound_llm:
                    st.error("LLM client is not properly initialized.")
                    st.stop()
                response_aimessage = bound_llm.invoke(langchain_conversation_messages)
            
            if not response_aimessage or (not response_aimessage.content and not response_aimessage.tool_calls):
                error_message_ui = "LLM response was empty or did not contain content or tool calls. This might be an issue with the API provider or the selected model."
                print(f"Error: {error_message_ui}") # Log to console
                if response_aimessage:
                    try:
                        print(f"Raw LLM AIMessage object: {response_aimessage.model_dump_json(indent=2)}")
                    except Exception as e:
                        print(f"Could not serialize response_aimessage, printing as string: {str(response_aimessage)}. Error: {e}")
                else:
                    print("LLM: response_aimessage object was None.")

                st.error(error_message_ui + " Check console logs for more details if a response object was received.")
                if current_hist_valid_for_prompt:
                    st.session_state.histories[active_history_idx]["messages"].append(
                        {"role": "assistant", "content": "[Error: LLM returned no valid response content or tool calls]"}
                    )
                st.rerun()
                st.stop() # Halt further processing in this block for this run

            parsed_tool_calls = response_aimessage.tool_calls

            if parsed_tool_calls:
                # Assistant wants to call a tool
                if current_hist_valid_for_prompt:
                    # Store assistant's intent to call tool(s)
                    st.session_state.histories[active_history_idx]["messages"].append(
                        convert_aimessage_to_storage_dict(response_aimessage)
                    )
                
                # Add assistant's AIMessage to the ongoing LangChain conversation
                langchain_conversation_messages.append(response_aimessage)

                for tool_call_data in parsed_tool_calls: # tool_call_data is a ToolCall dict
                    function_name = tool_call_data["name"]
                    function_to_call = available_tools_mapping.get(function_name)
                    
                    if function_to_call:
                        try:
                            function_args = tool_call_data["args"] # Already a dict
                            tool_output = function_to_call(**function_args)
                        except Exception as e:
                            tool_output = f"Error parsing arguments or calling tool {function_name}: {e}"
                    else:
                        tool_output = f"Error: Tool '{function_name}' not found."

                    # Store tool message in history (dict format)
                    tool_message_for_storage = {
                        "role": "tool", 
                        "tool_call_id": tool_call_data["id"], 
                        "name": function_name, 
                        "content": tool_output
                    }
                    if current_hist_valid_for_prompt:
                        st.session_state.histories[active_history_idx]["messages"].append(tool_message_for_storage)
                    
                    # Add ToolMessage to LangChain conversation for the next LLM call
                    langchain_conversation_messages.append(
                        ToolMessage(content=tool_output, tool_call_id=tool_call_data["id"])
                    )
                
                # Second call to LLM with tool responses
                with st.spinner("Processing tool results..."):
                    if not bound_llm: # Should not happen if first call succeeded
                        st.error("LLM client became uninitialized before second call.")
                        st.stop()
                    second_response_aimessage = bound_llm.invoke(langchain_conversation_messages)
                
                if current_hist_valid_for_prompt:
                    st.session_state.histories[active_history_idx]["messages"].append(
                        convert_aimessage_to_storage_dict(second_response_aimessage)
                    )
            else:
                # No tool calls, direct response
                if response_aimessage.content and current_hist_valid_for_prompt:
                    st.session_state.histories[active_history_idx]["messages"].append(
                        convert_aimessage_to_storage_dict(response_aimessage)
                    )
            
            # Rerun to display all new messages, including tool interactions
            st.rerun()

        except openai.APIConnectionError as e:
            st.error(f"API Connection Error: {e}. Please check your network and the Base URL.")
            print(f"API Connection Error: {type(e).__name__}: {e}")
        except openai.RateLimitError as e:
            st.error(f"Rate Limit Exceeded: {e}")
        except openai.AuthenticationError as e:
            st.error(f"Authentication Error: {e}. Check your API key.")
        except openai.APIStatusError as e:
            error_detail = "No additional detail in response."
            response_text_for_logging = "" # For console logging
            if e.response is not None:
                response_text_for_logging = e.response.text
                try:
                    # Try to parse as JSON for prettier display if it's structured error
                    parsed_json_error = json.loads(e.response.text)
                    error_detail = json.dumps(parsed_json_error, indent=2)
                except json.JSONDecodeError:
                    error_detail = e.response.text # Fallback to raw text
            
            print(f"API Status Error (code {e.status_code}). Raw response text: {response_text_for_logging}") # Log to console

            st.error(f"API Status Error (code {e.status_code}):\nThe API provider returned an error. This can be due to various reasons such as malformed request data, an issue with the selected model, or API-specific limitations. See details below from the API response:\n```\n{error_detail}\n```")
            
            # Specific hint for Ollama 404, can be expanded for other common provider/status code issues
            if st.session_state.current_provider == "ollama" and e.status_code == 404:
                st.error(
                    "For Ollama, a 404 error (Not Found) often means:\n"
                    "1. The Ollama server is not running or not accessible at the configured Base URL (default: http://localhost:11434/v1).\n"
                    "2. The specific model (e.g., '" + st.session_state.current_model + "') is not available/pulled in Ollama. Try: `ollama pull " + st.session_state.current_model + "` in your terminal.\n"
                    "3. If using Docker for this Streamlit app and Ollama is on the host, ensure the app can reach the Ollama server (network config, e.g., use 'host.docker.internal' instead of 'localhost' in Base URL if applicable).\n"
                    "4. The Ollama version might be outdated or its OpenAI-compatible endpoint `/v1/chat/completions` is not enabled/available."
                )
        except openai.APIError as e: # Catch other OpenAI API errors
            st.error(f"OpenAI API Error: {e}")
            print(f"OpenAI API Error: {type(e).__name__}: {e}")
        except Exception as e: # General fallback for non-OpenAI errors or unexpected issues
            st.error(f"An unexpected error occurred: {e}")
            print(f"An unexpected error occurred: {type(e).__name__}: {e}")
