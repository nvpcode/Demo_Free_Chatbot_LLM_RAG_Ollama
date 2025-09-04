import json
import os
import threading
import time
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import pickle
from datetime import datetime, timedelta

class CacheEntry:
    """Lớp đại diện cho một entry trong cache"""
    def __init__(self, question: str, answer: str, timestamp: datetime = None):
        self.question = question
        self.answer = answer
        self.timestamp = timestamp or datetime.now()
        self.access_count = 0
        self.last_accessed = timestamp or datetime.now()
    
    def update_access(self):
        """Cập nhật thông tin truy cập"""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def to_dict(self):
        """Chuyển đổi thành dictionary để lưu trữ"""
        return {
            'question': self.question,
            'answer': self.answer,
            'timestamp': self.timestamp.isoformat(),
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Tạo instance từ dictionary"""
        entry = cls(
            question=data['question'],
            answer=data['answer'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )
        entry.access_count = data['access_count']
        entry.last_accessed = datetime.fromisoformat(data['last_accessed'])
        return entry

class TextSimilarityThread:
    """Luồng 1: So sánh từ ngữ thông thường"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        self.question_vectors = None
        self.questions = []
    
    def update_questions(self, questions: List[str]):
        """Cập nhật danh sách câu hỏi và tính toán vector"""
        if questions:
            self.questions = questions
            self.question_vectors = self.vectorizer.fit_transform(questions)
    
    def find_similar_question(self, query: str) -> Tuple[Optional[str], float]:
        """Tìm câu hỏi tương tự sử dụng TF-IDF và cosine similarity"""
        if not self.questions or self.question_vectors is None:
            return None, 0.0
        
        # Tính vector cho câu hỏi mới
        query_vector = self.vectorizer.transform([query])
        
        # Tính cosine similarity
        similarities = cosine_similarity(query_vector, self.question_vectors).flatten()
        
        # Tìm câu hỏi có độ tương tự cao nhất
        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[max_similarity_idx]
        
        if max_similarity >= self.similarity_threshold:
            return self.questions[max_similarity_idx], max_similarity
        
        return None, max_similarity
    
    def find_similar_question_fuzzy(self, query: str) -> Tuple[Optional[str], float]:
        """Tìm câu hỏi tương tự sử dụng fuzzy matching"""
        if not self.questions:
            return None, 0.0
        
        best_match = None
        best_ratio = 0.0
        
        for question in self.questions:
            # Sử dụng SequenceMatcher để so sánh
            ratio = SequenceMatcher(None, query.lower(), question.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = question
        
        if best_ratio >= self.similarity_threshold:
            return best_match, best_ratio
        
        return None, best_ratio

class VectorSimilarityThread:
    """Luồng 2: So sánh vector biểu diễn"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.question_embeddings = {}
        self.embedding_model = None
    
    def set_embedding_model(self, embedding_model):
        """Thiết lập model embedding"""
        self.embedding_model = embedding_model
    
    def update_question_embeddings(self, questions: List[str]):
        """Cập nhật embedding cho danh sách câu hỏi"""
        if not self.embedding_model:
            return
        
        self.question_embeddings.clear()
        for question in questions:
            try:
                embedding = self.embedding_model.embed_query(question)
                self.question_embeddings[question] = embedding
            except Exception as e:
                print(f"Lỗi khi tạo embedding cho câu hỏi: {e}")
    
    def find_similar_question(self, query: str) -> Tuple[Optional[str], float]:
        """Tìm câu hỏi tương tự sử dụng vector similarity"""
        if not self.question_embeddings or not self.embedding_model:
            return None, 0.0
        
        try:
            # Tạo embedding cho câu hỏi mới
            query_embedding = self.embedding_model.embed_query(query)
            
            best_match = None
            best_similarity = 0.0
            
            # So sánh với tất cả câu hỏi trong cache
            for question, embedding in self.question_embeddings.items():
                # Tính cosine similarity
                similarity = self._cosine_similarity(query_embedding, embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = question
            
            if best_similarity >= self.similarity_threshold:
                return best_match, best_similarity
            
            return None, best_similarity
            
        except Exception as e:
            print(f"Lỗi khi tìm kiếm vector similarity: {e}")
            return None, 0.0
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Tính cosine similarity giữa 2 vector"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class CacheManager:
    """Quản lý cache với 2 luồng tìm kiếm"""
    
    def __init__(self, 
                 cache_file: str = "cache_data.pkl",
                 max_cache_size: int = 1000,
                 cache_expiry_days: int = 30):
        self.cache_file = cache_file
        self.max_cache_size = max_cache_size
        self.cache_expiry_days = cache_expiry_days
        
        # Khởi tạo 2 luồng
        self.text_thread = TextSimilarityThread()
        self.vector_thread = VectorSimilarityThread()
        
        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_lock = threading.Lock()
        
        # Load cache từ file
        self.load_cache()
        
        # Thread pool cho việc tìm kiếm song song
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def set_embedding_model(self, embedding_model):
        """Thiết lập model embedding cho vector thread"""
        self.vector_thread.set_embedding_model(embedding_model)
        self._update_thread_questions()
    
    def _update_thread_questions(self):
        """Cập nhật danh sách câu hỏi cho cả 2 thread"""
        questions = list(self.cache.keys())
        self.text_thread.update_questions(questions)
        self.vector_thread.update_question_embeddings(questions)
    
    def add_to_cache(self, question: str, answer: str):
        """Thêm câu hỏi và câu trả lời vào cache"""
        with self.cache_lock:
            # Tạo hash key cho câu hỏi
            question_hash = hashlib.md5(question.lower().encode()).hexdigest()
            
            # Thêm vào cache
            self.cache[question_hash] = CacheEntry(question, answer)
            
            # Kiểm tra kích thước cache
            if len(self.cache) > self.max_cache_size:
                self._cleanup_cache()
            
            # Cập nhật questions cho các thread
            self._update_thread_questions()
            
            # Lưu cache
            self.save_cache()
    
    def get_from_cache(self, question: str) -> Optional[str]:
        """Lấy câu trả lời từ cache nếu có câu hỏi tương tự"""
        # Tạo hash cho câu hỏi hiện tại
        current_hash = hashlib.md5(question.lower().encode()).hexdigest()
        
        # Kiểm tra cache trực tiếp trước
        if current_hash in self.cache:
            entry = self.cache[current_hash]
            entry.update_access()
            return entry.answer
        
        # Sử dụng 2 luồng để tìm kiếm song song
        futures = []
        
        # Luồng 1: Text similarity
        futures.append(
            self.executor.submit(self.text_thread.find_similar_question, question)
        )
        
        # Luồng 2: Vector similarity
        futures.append(
            self.executor.submit(self.vector_thread.find_similar_question, question)
        )
        
        # Chờ kết quả từ cả 2 luồng
        best_match = None
        best_similarity = 0.0
        best_method = ""
        
        for future in as_completed(futures):
            try:
                similar_question, similarity = future.result()
                if similar_question and similarity > best_similarity:
                    best_match = similar_question
                    best_similarity = similarity
                    best_method = "text" if future == futures[0] else "vector"
            except Exception as e:
                print(f"Lỗi khi tìm kiếm cache: {e}")
        
        # Nếu tìm thấy câu hỏi tương tự
        if best_match and best_similarity > 0.5:
            # Tìm hash của câu hỏi tương tự
            for q_hash, entry in self.cache.items():
                if entry.question == best_match:
                    entry.update_access()
                    print(f"� Cache hit! Sử dụng {best_method} similarity (độ tương tự: {best_similarity:.3f})")
                    return entry.answer
        
        print(f"❌ Cache miss! Không tìm thấy câu hỏi tương tự")
        return None
    
    def _cleanup_cache(self):
        """Dọn dẹp cache: xóa các entry cũ và ít sử dụng"""
        current_time = datetime.now()
        expiry_time = current_time - timedelta(days=self.cache_expiry_days)
        
        # Lọc các entry hết hạn
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.timestamp < expiry_time
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        # Nếu vẫn còn quá nhiều, xóa các entry ít sử dụng
        if len(self.cache) > self.max_cache_size:
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: (x[1].access_count, x[1].last_accessed)
            )
            
            # Xóa 20% entry ít sử dụng nhất
            delete_count = int(len(self.cache) * 0.2)
            for i in range(delete_count):
                if i < len(sorted_entries):
                    del self.cache[sorted_entries[i][0]]
    
    def save_cache(self):
        """Lưu cache vào file"""
        try:
            cache_data = {
                key: entry.to_dict() 
                for key, entry in self.cache.items()
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            print(f"Lỗi khi lưu cache: {e}")
    
    def load_cache(self):
        """Load cache từ file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.cache = {
                    key: CacheEntry.from_dict(data)
                    for key, data in cache_data.items()
                }
                
                print(f"\n✅ Đã load {len(self.cache)} entries từ cache")
                
        except Exception as e:
            print(f"Lỗi khi load cache: {e}")
            self.cache = {}
    
    def get_cache_stats(self) -> Dict:
        """Lấy thống kê về cache"""
        with self.cache_lock:
            total_entries = len(self.cache)
            total_access = sum(entry.access_count for entry in self.cache.values())
            
            if total_entries > 0:
                avg_access = total_access / total_entries
            else:
                avg_access = 0
            
            return {
                'total_entries': total_entries,
                'total_access': total_access,
                'average_access': avg_access,
                'cache_size_mb': os.path.getsize(self.cache_file) / (1024 * 1024) if os.path.exists(self.cache_file) else 0
            }
    
    def clear_cache(self):
        """Xóa toàn bộ cache"""
        with self.cache_lock:
            self.cache.clear()
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
            print("🗑️ Đã xóa toàn bộ cache")


# Ví dụ sử dụng
if __name__ == "__main__":
    # Khởi tạo cache manager
    cache_manager = CacheManager()
    
    # Test thêm một số câu hỏi
    test_qa = [
        ("Bạn có thể giải thích về machine learning không?", "Machine learning là một nhánh của AI..."),
        ("ML là gì?", "ML viết tắt của Machine Learning..."),
        ("Làm thế nào để học deep learning?", "Để học deep learning, bạn cần...")
    ]
    
    for question, answer in test_qa:
        cache_manager.add_to_cache(question, answer)
    
    # Test tìm kiếm
    test_question = "Machine learning là gì?"
    result = cache_manager.get_from_cache(test_question)
    
    if result:
        print(f"Tìm thấy câu trả lời: {result}")
    else:
        print("Không tìm thấy câu trả lời tương tự")
    
    # In thống kê
    stats = cache_manager.get_cache_stats()
    print(f"Thống kê cache: {stats}")