from pathlib import Path
import os
from dotenv import load_dotenv

from openai import OpenAI
from fastembed import TextEmbedding
from chromadb import Client

load_dotenv()
API_KEY = os.getenv('AI_KEY')
embedd_model = TextEmbedding()
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=API_KEY,
)
chroma_client = Client()
collenction = chroma_client.get_or_create_collection(name='main')

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

            collenction.add(
                ids=ids,
                embeddings=[*embeddings],
                documents=chunks,
                metadatas=metadatas
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

def main():
    print("Hello from agency!")
    load_and_index_document('data')


if __name__ == "__main__":
    main()
