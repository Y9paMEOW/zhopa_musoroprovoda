Отличные уточняющие вопросы! Разберём каждый пункт подробно с практическими рекомендациями и кодом.

1. Умное чанкование

Стратегии чанкование:
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter, 
    MarkdownHeaderTextSplitter,
    SentenceTransformersTokenTextSplitter
)

# 1. Рекурсивное чанкование (универсальное)
def recursive_chunking(text, chunk_size=512, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # Приоритет разделителей
    )
    return splitter.split_text(text)

# 2. Чанкование по семантическим границам (для Markdown/HTML)
def semantic_chunking_markdown(text):
    headers_to_split_on = [
        ("#", "Заголовок 1"),
        ("##", "Заголовок 2"), 
        ("###", "Заголовок 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    return markdown_splitter.split_text(text)

# 3. Чанкование с учетом токенов для моделей эмбеддингов
def token_aware_chunking(text, chunk_size=256):
    splitter = SentenceTransformersTokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=20,
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    return splitter.split_text(text)

# 4. Чанкование с сохранением контекста
def contextual_chunking(documents):
    chunks = []
    for doc in documents:
        # Разбиваем на абзацы
        paragraphs = doc.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) < 50:  # Пропускаем очень короткие абзацы
                continue
                
            # Добавляем контекст из предыдущего/следующего абзаца
            context = ""
            if i > 0 and len(paragraphs[i-1]) > 30:
                context += paragraphs[i-1][-100:] + " "
            if i < len(paragraphs)-1 and len(paragraphs[i+1]) > 30:
                context += paragraphs[i+1][:100]
                
            chunk = f"{paragraph} {context}".strip()
            chunks.append(chunk)
    
    return chunks

2. Очистка датасета
import re
import html
from bs4 import BeautifulSoup

def clean_dataset(texts):
    cleaned_texts = []
    
    for text in texts:
        # 1. Удаление HTML/XML тегов
        clean_text = re.sub(r'<[^>]+>', '', text)
        
        # 2. Декодирование HTML entities
        clean_text = html.unescape(clean_text)
        
        # 3. Удаление специальных символов (сохраняем кириллицу, латиницу, пунктуацию)
        clean_text = re.sub(r'[^\w\sа-яА-ЯёЁ.,!?;:()\-–—]', ' ', clean_text)
        
        # 4. Нормализация пробелов
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        # 5. Удаление слишком коротких текстов
        if len(clean_text.strip()) > 25:
            cleaned_texts.append(clean_text.strip())
    
    return cleaned_texts

# Расширенная очистка с анализом качества
def advanced_cleaning(texts, min_length=50, max_repetition=3):
    cleaned = []
    
    for text in texts:
        # Проверка на повторяющиеся фразы (признак плохого парсинга)
        words = text.split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            max_repeats = max(word_counts.values())
            if max_repeats > max_repetition * (len(words) / 10):
                continue  # Пропускаем текст с повторениями
        
        # Проверка на осмысленность (соотношение уникальных слов)
        if len(set(words)) / len(words) < 0.3:
            continue
            
        if len(text) >= min_length:
            cleaned.append(text)
    
    return cleaned

3. Комбинированный поиск (Гибридный поиск)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from typing import List, Tuple

class HybridSearch:
    def __init__(self, embedding_model, chunks: List[str]):
        self.chunks = chunks

self.embedding_model = embedding_model
        
        # Семантические эмбеддинги
        self.embeddings = embedding_model.encode(chunks)
        
        # BM25 для лексческого поиска
        tokenized_chunks = [chunk.split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
    
    def search(self, query: str, top_k: int = 10, alpha: float = 0.5) -> List[Tuple[str, float]]:
        # Семантический поиск
        query_embedding = self.embedding_model.encode([query])
        semantic_scores = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Лексческий поиск (BM25)
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Нормализация scores
        semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        
        # Комбинирование
        combined_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores
        
        # Топ-K результатов
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        return [(self.chunks[i], combined_scores[i]) for i in top_indices]

# Использование
# hybrid_searcher = HybridSearch(embedding_model, chunks)
# results = hybrid_searcher.search("ваш запрос", top_k=10, alpha=0.6)

4. Переранжирование (Re-ranking)
from sentence_transformers import CrossEncoder
import numpy as np

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        # Создаем пары (query, candidate)
        pairs = [[query, candidate] for candidate in candidates]
        
        # Получаем скоринги от cross-encoder
        scores = self.model.predict(pairs)
        
        # Сортируем по убыванию score
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(candidates[i], scores[i]) for i in ranked_indices]

# Полный пайплайн с переранжированием
def full_retrieval_pipeline(query, chunks, embedding_model, top_k=10, rerank_top_k=5):
    # 1. Гибридный поиск
    hybrid_searcher = HybridSearch(embedding_model, chunks)
    initial_results = hybrid_searcher.search(query, top_k=top_k)
    
    # 2. Извлекаем только тексты для переранжирования
    candidate_texts = [result[0] for result in initial_results]
    
    # 3. Переранжирование
    reranker = Reranker()
    final_results = reranker.rerank(query, candidate_texts, top_k=rerank_top_k)
    
    return final_results

5. Расширение запроса (Query Expansion)
from openai import OpenAI
import random

class QueryExpander:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def generate_alternative_queries(self, original_query: str, num_variants: int = 3) -> List[str]:
        prompt = f"""
        Исходный запрос: "{original_query}"
        
        Сгенерируй {num_variants} альтернативных формулировок этого запроса, которые могут быть использованы для поиска информации. Формулировки должны быть разнообразными: синонимы, перефразирование, более общие и более конкретные версии.
        
        Верни только список вариантов, каждый с новой строки.
        """
        
        try:
            response = self.llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            alternatives = response.choices[0].message.content.strip().split('\n')
            # Очищаем от нумерации
            cleaned_alternatives = [alt.split('. ', 1)[-1] if '. ' in alt else alt for alt in alternatives]
            return [original_query] + cleaned_alternatives[:num_variants]
            
        except Exception as e:
            print(f"Ошибка генерации запросов: {e}")
            return [original_query]

def hyde_expansion(self, query: str) -> str:
        """Hypothetical Document Embeddings"""
        prompt = f"""
        На основе следующего запроса пользователя, сгенерируй идеальный ответ, который мог бы содержаться в документе.
        
        Запрос: {query}
        
        Сгенерируй краткий, информативный ответ (2-3 предложения), который полностью отвечает на запрос.
        """
        
        try:
            response = self.llm.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            hypothetical_doc = response.choices[0].message.content.strip()
            return hypothetical_doc
            
        except Exception as e:
            print(f"Ошибка HyDE: {e}")
            return query

# Расширенный поиск с query expansion
class AdvancedSearch:
    def __init__(self, embedding_model, chunks, llm_client):
        self.embedding_model = embedding_model
        self.chunks = chunks
        self.query_expander = QueryExpander(llm_client)
        self.hybrid_searcher = HybridSearch(embedding_model, chunks)
        self.reranker = Reranker()
    
    def search_with_expansion(self, query: str, use_hyde: bool = True, use_multi_query: bool = True):
        # 1. Расширение запроса
        if use_hyde:
            hyde_query = self.query_expander.hyde_expansion(query)
            search_queries = [query, hyde_query]
        elif use_multi_query:
            search_queries = self.query_expander.generate_alternative_queries(query)
        else:
            search_queries = [query]
        
        # 2. Поиск по всем вариантам запросов
        all_results = []
        for search_query in search_queries:
            results = self.hybrid_searcher.search(search_query, top_k=15)
            all_results.extend(results)
        
        # 3. Убираем дубликаты и берем уникальные чанки
        unique_chunks = {}
        for chunk, score in all_results:
            if chunk not in unique_chunks or score > unique_chunks[chunk]:
                unique_chunks[chunk] = score
        
        # 4. Переранжирование
        candidate_chunks = list(unique_chunks.keys())[:20]  # Берем топ-20 для реранкинга
        final_results = self.reranker.rerank(query, candidate_chunks, top_k=7)
        
        return final_results

Полный пример использования:
# Инициализация компонентов
embedding_model = ... # ваша модель эмбеддингов
llm_client = ... # ваш LLM клиент

# Загрузка и очистка данных
with open("data.txt", "r", encoding="utf-8") as f:
    raw_texts = f.readlines()

cleaned_texts = clean_dataset(raw_texts)
cleaned_texts = advanced_cleaning(cleaned_texts)

# Чанкование
chunks = []
for text in cleaned_texts:
    text_chunks = recursive_chunking(text, chunk_size=512, chunk_overlap=50)
    chunks.extend(text_chunks)

# Создание поисковой системы
search_system = AdvancedSearch(embedding_model, chunks, llm_client)

# Поиск
results = search_system.search_with_expansion(
    "Как настроить модель для обработки текстов?", 
    use_hyde=True, 
    use_multi_query=True
)

for chunk, score in results:
    print(f"Score: {score:.3f}")
    print(f"Text: {chunk[:200]}...")
    print("---")

Эти техники значительно улучшат качество поиска релевантных чанков, что напрямую повлияет на качество финальных ответов вашей RAG-системы.
