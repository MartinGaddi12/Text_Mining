import chromadb
import json
import os

def check_rag_progress(jsonl_path='all_drugs_docs.jsonl', db_path='chromadb'):
    """Verificar el progreso del RAG y contar documentos procesados"""
    
    print("="*60)
    print("VERIFICANDO PROGRESO DEL RAG")
    print("="*60)
    
    # Verificar archivo JSONL
    if not os.path.exists(jsonl_path):
        print(f"❌ Error: El archivo {jsonl_path} no existe")
        return
    
    # Contar líneas totales en el archivo JSONL
    total_lines = 0
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # Solo contar líneas no vacías
                total_lines += 1
    
    print(f"📄 Total de líneas en {jsonl_path}: {total_lines:,}")
    
    # Verificar ChromaDB
    try:
        # Intentar diferentes APIs de ChromaDB
        try:
            # API nueva
            client = chromadb.PersistentClient(path=db_path)
        except AttributeError:
            # API antigua
            client = chromadb.Client()
        
        collection_name = "drugs_documents"
        
        try:
            collection = client.get_collection(collection_name)
            total_chunks = collection.count()
            
            # Obtener metadatos para ver hasta qué línea llegamos
            if total_chunks > 0:
                # Obtener todos los metadatos (puede ser lento si hay muchos)
                all_metadata = collection.get(include=['metadatas'])
                
                # Encontrar la línea máxima procesada
                max_line = 0
                lines_processed = set()
                
                for metadata in all_metadata['metadatas']:
                    line_num = metadata.get('line_number', 0)
                    lines_processed.add(line_num)
                    max_line = max(max_line, line_num)
                
                unique_lines = len(lines_processed)
                
                print(f"🗄️  ChromaDB:")
                print(f"   - Total de chunks: {total_chunks:,}")
                print(f"   - Líneas únicas procesadas: {unique_lines:,}")
                print(f"   - Última línea procesada: {max_line:,}")
                print(f"   - Progreso: {(unique_lines/total_lines)*100:.1f}%")
                
                if unique_lines < total_lines:
                    remaining = total_lines - unique_lines
                    print(f"   - Líneas restantes: {remaining:,}")
                    print(f"✅ Puede continuar desde la línea {max_line + 1}")
                else:
                    print(f"✅ ¡Procesamiento completo!")
                    
            else:
                print(f"❌ ChromaDB vacío - puede empezar desde el principio")
                
        except Exception as e:
            print(f"❌ No se encontró la colección: {e}")
            print(f"💡 Puede empezar desde el principio")
            
    except Exception as e:
        print(f"❌ Error accediendo a ChromaDB: {e}")
    
    print("="*60)

if __name__ == "__main__":
    check_rag_progress()
