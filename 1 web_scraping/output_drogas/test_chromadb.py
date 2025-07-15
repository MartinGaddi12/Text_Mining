import chromadb

# Vamos a probar diferentes formas de acceder a ChromaDB
print("Probando APIs de ChromaDB...")

try:
    # Opción 1: Client simple
    print("1. Probando chromadb.Client()...")
    client = chromadb.Client()
    print("   ✅ chromadb.Client() funciona")
except Exception as e:
    print(f"   ❌ chromadb.Client() falló: {e}")

try:
    # Opción 2: PersistentClient
    print("2. Probando chromadb.PersistentClient()...")
    client = chromadb.PersistentClient(path="chromadb")
    print("   ✅ chromadb.PersistentClient() funciona")
except Exception as e:
    print(f"   ❌ chromadb.PersistentClient() falló: {e}")

try:
    # Opción 3: HttpClient  
    print("3. Probando chromadb.HttpClient()...")
    client = chromadb.HttpClient()
    print("   ✅ chromadb.HttpClient() funciona")
except Exception as e:
    print(f"   ❌ chromadb.HttpClient() falló: {e}")

try:
    # Opción 4: API directa
    print("4. Probando acceso directo...")
    import chromadb.api
    print("   ✅ chromadb.api disponible")
except Exception as e:
    print(f"   ❌ chromadb.api falló: {e}")

# Mostrar atributos disponibles en chromadb
print("\n5. Atributos disponibles en chromadb:")
attrs = [attr for attr in dir(chromadb) if not attr.startswith('_')]
for attr in attrs:
    print(f"   - {attr}")

# Probar la versión de ChromaDB
try:
    print(f"\n6. Versión de ChromaDB: {chromadb.__version__}")
except:
    print("\n6. No se pudo obtener la versión")

# Intentar usar la API antigua
print("\n7. Probando API antigua...")
try:
    # Versión muy antigua usaba get_client
    if hasattr(chromadb, 'get_client'):
        client = chromadb.get_client()
        print("   ✅ chromadb.get_client() funciona")
    elif hasattr(chromadb, 'Client'):
        client = chromadb.Client()
        print("   ✅ chromadb.Client() funciona")
    else:
        print("   ❌ No se encontró forma de crear cliente")
except Exception as e:
    print(f"   ❌ API antigua falló: {e}")
