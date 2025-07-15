import os
import json

def simple_check():
    """Verificación simple del progreso"""
    
    print("="*60)
    print("VERIFICACIÓN SIMPLE DEL RAG")
    print("="*60)
    
    jsonl_path = 'all_drugs_docs.jsonl'
    db_path = 'chromadb'
    
    # Verificar archivo JSONL
    if os.path.exists(jsonl_path):
        total_lines = 0
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    total_lines += 1
        print(f"📄 Total de líneas en JSONL: {total_lines:,}")
    else:
        print(f"❌ No se encontró {jsonl_path}")
        return
    
    # Verificar directorio ChromaDB
    if os.path.exists(db_path):
        print(f"🗄️  Directorio ChromaDB encontrado: {db_path}")
        
        # Listar contenido del directorio
        try:
            db_files = os.listdir(db_path)
            print(f"   - Archivos en ChromaDB: {len(db_files)}")
            for file in db_files[:5]:  # Mostrar solo los primeros 5
                file_path = os.path.join(db_path, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"     • {file}: {size:,} bytes")
                else:
                    print(f"     • {file}/ (directorio)")
            
            if len(db_files) > 5:
                print(f"     ... y {len(db_files) - 5} archivos más")
                
        except Exception as e:
            print(f"   ❌ Error leyendo directorio: {e}")
    else:
        print(f"❌ No se encontró directorio ChromaDB")
    
    print("\n" + "="*60)
    print("RECOMENDACIONES:")
    
    if os.path.exists(db_path):
        print("✅ El RAG parece estar construido")
        print("💡 Puedes intentar hacer consultas con:")
        print("   python query_rag.py 'tu consulta'")
    else:
        print("❌ Necesitas construir el RAG")
        print("💡 Ejecuta: python rag_script.py")
    
    print("="*60)

if __name__ == "__main__":
    simple_check()
