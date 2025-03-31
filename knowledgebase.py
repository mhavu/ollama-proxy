#!/usr/bin/env python3
"""Command-line tool for building LLM RAG knowledge bases"""
from pathlib import Path
import argparse
from datetime import datetime
import chromadb
import ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.base.llms.types import ChatMessage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from flask import Flask, request, jsonify
from flask.json.provider import DefaultJSONProvider
from waitress import serve


class CustomJSONProvider(DefaultJSONProvider):
    """JSONProvider that encodes datetimes in ISO 8601 format"""
    def default(self, obj):
        """Add encoding datetimes in ISO 8601 format"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(self, obj)

def get_knowledge_base(name, embedding, database_path,
                       input_dir=None, input_files=None, exclude=None,
                       recursive=False, chunk_size=512, chunk_overlap=20):
    """Build an offline retrieval-augmented generation (RAG) system"""

    # Initialize database
    database = chromadb.PersistentClient(path=database_path)
    chroma_collection = database.get_or_create_collection(name)

    # Assign Chroma as the vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create an index
    Settings.embed_model = OllamaEmbedding(model_name=embedding)
    if input_dir is None and input_files is None:
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )
    else:
        documents = SimpleDirectoryReader(input_dir=input_dir,
                                          input_files=input_files,
                                          exclude=exclude,
                                          recursive=recursive).load_data()
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[SentenceSplitter(chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap)],
            upsert=True
        )
    return index

def create_query_engine(model, index, timeout=120.0, similarity_top_k=10,
                       chat=True):
    """Initialize the LLM with Ollama and create a query engine"""
    Settings.llm = Ollama(model=model, request_timeout=timeout)
    if chat:
        engine = index.as_chat_engine(chat_mode='context',
                                      similarity_top_k=similarity_top_k)
    else:
        engine = index.as_query_engine(response_mode='refine',
                                       similarity_top_k=similarity_top_k)
    return engine

def create_app(name, model, engine):
    """Make an Ollama proxy for chat UIs"""

    app = Flask(__name__)
    app.json = CustomJSONProvider(app)

    @app.route('/api/generate', methods=['POST'])
    def generate():
        try:
            data = request.get_json()
            target = data.get('model')
            prompt = data.get('prompt')
            suffix = data.get('suffix')
            images = data.get('images')

            if not prompt:
                return jsonify({'error': 'Prompt is required'}), 400

            if target == name:
                # Query the knowledge base
                if not hasattr(engine, 'query'):
                    typename = type(engine).__name__
                    raise AttributeError(f"{name} is a {typename}")
                response = engine.query(prompt)
            else:
                response = ollama.generate(model=target, prompt=prompt,
                                           suffix=suffix, images=images)
            return jsonify(response.response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/chat', methods=['POST'])
    def chat():
        try:
            data = request.get_json()
            target = data.get('model')
            messages = data.get('messages')

            if target == name:
                # Chat with the knowledge base
                if not hasattr(engine, 'chat'):
                    type_ = type(engine).__name__
                    raise AttributeError(f"{name} is a {type_}")
                history = [ChatMessage(m['content'], role=m['role'])
                           for m in messages[:-1]]
                response = engine.chat(messages[-1]['content'],
                                       chat_history=history).response
                response = {'model': name,
                            'done': True,
                            'message': {'role': 'assistant',
                                        'content': response}}
            else:
                response = ollama.chat(model=target,
                                       messages=messages).model_dump()
            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/tags', methods=['GET'])
    def list_models():
        try:
            model_list = ollama.list().model_dump()['models']
            for m in model_list:
                m['name'] = m['model']
                if m['model'] == model:
                    proxy_model = m.copy()
                    proxy_model['name'] = name
                    model_list = model_list + [proxy_model]

            return jsonify({'models': model_list})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/show', methods=['POST'])
    def show_information():
        try:
            data = request.get_json()
            target = data.get('model')

            if target == name:
                target = model
            response = ollama.show(model=target)

            return jsonify(response.model_dump())

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/ps', methods=['GET'])
    def list_running_models():
        try:
            response = ollama.ps().model_dump()

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/', methods=['HEAD'])
    def head():
        return ''

    return app


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="An offline retrieval-augmented generation (RAG) system "
        "accessible as a proxy Ollama server"
    )
    subparsers = parser.add_subparsers()
    # Build an offline retrieval-augmented generation (RAG) system
    list_parser = subparsers.add_parser('list', help="list available models")
    list_parser.add_argument('list', action='store_const', const=True,
                             help=argparse.SUPPRESS)
    generate_parser = subparsers.add_parser('generate',
                                            help="generate knowledge base")
    generate_parser.add_argument('generate', action='store_const', const=True,
                                 help=argparse.SUPPRESS)
    generate_parser.add_argument('-m', '--model', type=str,
                                 help="name of the model to use")
    generate_parser.add_argument('-e', '--embedding', type=str,
                                 help="name of the embedding model to use")
    generate_parser.add_argument('-d', '--dbpath', type=Path,
                                 default='./chroma_db',
                                 help="path to the database directory "
                                      "(default: \"./chroma_db\")")
    generate_parser.add_argument('-n', '--name', type=str,
                                 default='knowledge_base_db',
                                 help="name of the database and the proxy "
                                      "(default: \"knowledge_base_db\")")
    generate_parser.add_argument('-p', '--port', type=int, default=None,
                                 help="port number for the Ollama proxy "
                                      "(if omitted, the proxy is not started)")
    generate_parser.add_argument('--debug', action='store_true',
                                 help="print debug information")
    files = generate_parser.add_argument_group(
        'Files',
        "These arguments specify the documents to include in the knowledge "
        "base. If omitted, the proxy uses the existing database as is."
    )
    files.add_argument('path', type=Path, nargs='*',
                       help="document file(s) or a directory to add "
                       "to the database")
    files.add_argument('-R', '--recursive', action='store_true',
                        help="recurse into directories")
    files.add_argument('-x', '--exclude', type=str,
                        help="ignore files that match the exclude pattern")
    args = parser.parse_args()

    if hasattr(args, 'list'):
        for m in ollama.list().model_dump()['models']:
            print(m['model'])
    elif hasattr(args, 'generate'):
        models = [m['model'] for m in ollama.list().model_dump()['models']]
        if args.name in models:
            raise KeyError(f"Name {args.name} is already in use")
        if len(args.path) == 0:
            INPUT_DIR = None
            INPUT_FILES = None
        elif len(args.path) == 1:
            if args.path[0].is_dir():
                INPUT_DIR = args.path[0]
                INPUT_FILES = None
            else:
                INPUT_DIR = None
                INPUT_FILES = args.path
        else:
            INPUT_DIR = None
            INPUT_FILES = args.path
        knowledge_base_index = get_knowledge_base(
            args.name,
            args.embedding,
            str(args.dbpath.expanduser().resolve()),
            INPUT_DIR,
            INPUT_FILES,
            args.exclude,
            args.recursive,
            chunk_size=512,
            chunk_overlap=20,
        )
        if args.port is not None:
            query_engine = create_query_engine(
                args.model,
                knowledge_base_index,
                timeout=120.0,
                similarity_top_k=10
            )
            proxy_app = create_app(args.name, args.model, query_engine)
            if args.debug:
                proxy_app.run(port=args.port, debug=True)
            else:
                serve(proxy_app, host='127.0.0.1', port=args.port)
    else:
        parser.print_help()
