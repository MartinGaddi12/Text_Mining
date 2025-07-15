import chromadb
import sys

def query_rag(query_text, db_path='chromadb', n_results=5):
    """Función para consultar el RAG existente"""
    print(f"Consultando RAG: {query_text}")
    
    try:
        # Initialize ChromaDB client (compatible con versiones anteriores)
        try:
            # Intentar API nueva
            client = chromadb.PersistentClient(path=db_path)
        except AttributeError:
            # API antigua
            client = chromadb.Client()
        
        # Get collection
        collection_name = "drugs_documents"
        collection = client.get_collection(collection_name)
        
        # Query the collection
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        print(f"\nResultados encontrados: {len(results['documents'][0])}")
        print("="*80)
        
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            print(f"\n--- Resultado {i+1} ---")
            print(f"Medicamento: {metadata['source']}")
            print(f"Chunk {metadata['chunk_index']+1}/{metadata['total_chunks']}")
            print(f"Texto: {doc}")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error al consultar RAG: {e}")
        return None

def main():
    # Si se proporciona una consulta como argumento de línea de comandos
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # Consulta por defecto
        query = "¿Qué efectos secundarios tiene el ibuprofeno?"
    
    print("="*80)
    print("CONSULTA AL RAG DE MEDICAMENTOS")
    print("="*80)
    
    query_rag(query)
    
    print("\n" + "="*80)
    print("Para hacer más consultas, ejecuta:")
    print("python query_rag.py 'tu consulta aquí'")
    print("="*80)

if __name__ == "__main__":
    main()
