from pathlib import Path
import os
from dotenv import load_dotenv

from openai import OpenAI
from fastembed import TextEmbedding
from chromadb import PersistentClient

load_dotenv()
API_KEY = os.getenv('AI_KEY')
embedd_model = TextEmbedding()
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=API_KEY,
)
chroma_client = PersistentClient()
collection = chroma_client.get_or_create_collection(name='main')

def load_and_index_document(documets_path: str):
    doc_dir = Path(documets_path)
    if not doc_dir.exists():
        doc_dir.mkdir(parents=True)
        return 
    indexes_count = 0

    for file_path in doc_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix in ['.txt', '.md', '.py', '.pdf']:
        
            content = file_path.read_text(encoding='utf-8')
            
            if len(content) < 50:
                continue
            chunks = chunking(content)
            embeddings = embedd_model.embed(chunks)
            ids = [f'{file_path.name}_{i}' for i in range(len(chunks))]
            metadatas = [{'source': str(file_path), 'chunk_index': i} for i in range(len(chunks))]

            collection.add(
                ids=ids,
                embeddings=[*embeddings],  # ty:ignore[invalid-argument-type]
                documents=chunks,
                metadatas=metadatas  # ty:ignore[invalid-argument-type]
            )

            indexes_count += len(chunks)
            print(f'Indexes: {file_path.name} ({len(chunks)} chunks)')
        

    print('Total chunks processed', indexes_count)

def chunking(text: str, size: int = 500, overlap: int = 50) -> list[str]:

    chunks = []
    start = 0
    while start < len(text):
        end = start + size 
        chunk = text[start:end]

        if end < len(text):
            last_break = max(
                chunk.rfind('. '),
                chunk.rfind('? '),
                chunk.rfind('\n')
            )
            if last_break > size // 2:
                end = start + last_break + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if c]

def search_knowledge_base(query: str, num_results=3) -> list[dict]:
    
    result = collection.query(
        query_texts=query,
        n_results=num_results,
        include=['documents', 'metadatas', 'distances']
    )
    formatted = []
    print(result)
    if not result['documents']:
        raise Exception 
    for i in range(len(result['documents'][0])):
        print(i)
        formatted.append({
            'content': result['documents'][0][i],
            'source': result['metadatas'][0][i]['source'],  # ty:ignore[not-subscriptable]
            'relevance_score': 1 - result['distances'][0][i]  # ty:ignore[not-subscriptable]
        })
        
    return formatted

TOOLS = [
    {
        'name': 'search_knowledge_base',
        'description': """Search the indexed documents for relevant information. 
    Use this tool when you need find information to answer a question.
    The search uses semantic similarity - phrase your query naturaly.

    CONSTRAINS:
        - Search BEFORE answering question about documents
        - You can search multiple times with different queries 
        - if results arent relevant, try rephrasing your search
        """,
        'input_schema': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'The search query. Be specific and natural'
                },
                'num_results': {
                    'type': 'integer',
                    'description': 'Number of results to return.',
                    'default': 5
                }
            },
            'required': ['query']
        }
    },
    {
        'name': 'list_documents',
        'description': 'List all documents that have been indexed.',
        'input_schema': {'type': 'object', 'properties': {}, 'required': []}
    }
]

def main():
    print("Hello from agency!")
    # load_and_index_document('data')
    print(search_knowledge_base('Agents?'))


if __name__ == "__main__":
    main()
