import chromadb
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
import os
import uuid

def create_rag_from_jsonl(jsonl_path, db_path):
    print("Iniciando creación del RAG...")
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=db_path)
    
    # Create or get collection
    collection_name = "drugs_documents"
    try:
        collection = client.get_collection(collection_name)
        print(f"Colección '{collection_name}' encontrada, eliminando para recrear...")
        client.delete_collection(collection_name)
    except Exception:
        print(f"Creando nueva colección '{collection_name}'...")
    
    collection = client.create_collection(collection_name)
    
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize semantic chunker
    splitter = SemanticChunker(embeddings=embeddings, min_chunk_size=1000)
    
    print(f"Procesando archivo: {jsonl_path}")
    
    # Check if file exists
    if not os.path.exists(jsonl_path):
        print(f"Error: El archivo {jsonl_path} no existe")
        return
    
    # Process JSONL file
    chunk_counter = 0
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            try:
                # Parse each line as a JSON object
                data = json.loads(line.strip())
                
                # Get text content - usando 'document_text' en lugar de 'text'
                text_content = data.get('document_text', '')
                if not text_content:
                    print(f"Línea {line_num}: No hay contenido de texto")
                    continue
                
                # Split text into chunks
                chunks = splitter.split_text(text_content)
                
                # Add each chunk to the database
                for i, chunk in enumerate(chunks):
                    chunk_id = str(uuid.uuid4())
                    metadata = {
                        "source": data.get('drug_name', 'unknown'),  # usando 'drug_name' en lugar de 'url'
                        "chunk_index": i,
                        "line_number": line_num,
                        "total_chunks": len(chunks)
                    }
                    
                    collection.add(
                        documents=[chunk],
                        metadatas=[metadata],
                        ids=[chunk_id]
                    )
                    chunk_counter += 1
                
                if line_num % 10 == 0:
                    print(f"Procesadas {line_num} líneas, {chunk_counter} chunks creados")
                    
            except json.JSONDecodeError:
                print(f"Error al parsear JSON en línea {line_num}")
                continue
            except Exception as e:
                print(f"Error procesando línea {line_num}: {e}")
                continue
    
    print(f"RAG creado exitosamente!")
    print(f"Total de chunks creados: {chunk_counter}")
    print(f"Base de datos guardada en: {db_path}")

def query_rag(query_text, db_path, n_results=5):
    """Función para consultar el RAG"""
    print(f"Consultando RAG: {query_text}")
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=db_path)
    
    # Get collection
    collection_name = "drugs_documents"
    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        print(f"Error: No se pudo encontrar la colección {collection_name}: {e}")
        return
    
    # Query the collection
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    print(f"\nResultados encontrados: {len(results['documents'][0])}")
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\n--- Resultado {i+1} ---")
        print(f"Fuente: {metadata['source']}")
        print(f"Chunk {metadata['chunk_index']+1}/{metadata['total_chunks']}")
        print(f"Texto: {doc[:500]}{'...' if len(doc) > 500 else ''}")

if __name__ == "__main__":
    # Path to the JSONL file
    jsonl_path = 'all_drugs_docs.jsonl'
    # Path to save the ChromaDB
    db_path = 'chromadb'
    
    # Create the RAG
    create_rag_from_jsonl(jsonl_path, db_path)
    
    # Example query
    print("\n" + "="*50)
    print("EJEMPLO DE CONSULTA")
    print("="*50)
    query_rag("¿Qué efectos secundarios tiene el ibuprofeno?", db_path)
