import os
import re
import torch
import numpy as np
import pickle
import math
from typing import List, Dict, Tuple, Optional, Set
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import PyPDF2
from docx import Document
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from collections import defaultdict, Counter

@dataclass
class ConfidenceMetrics:
    """置信度指标"""
    max_similarity: float      # 最大相似度
    avg_similarity: float      # 平均相似度
    coverage_score: float      # 覆盖度评分
    consistency_score: float   # 一致性评分
    medical_relevance: float   # 医学相关性
    domain_alignment: float    # 领域对齐度
    final_confidence: float    # 最终置信度

@dataclass
class DomainContext:
    """医学领域上下文"""
    primary_domain: str           # 主要领域
    domain_confidence: float      # 领域置信度
    entities: List[str]           # 识别的医学实体
    expanded_terms: List[str]     # 扩展术语
    domain_weights: Dict[str, float]  # 领域权重

class MedicalDomainClassifier:
    """医学领域分类器"""
    def __init__(self):
        # 详细的医学专业领域分类
        self.domain_keywords = {
            'radiology': {
                'keywords': ['CT', 'MRI', '核磁', 'X光', 'X线', 'B超', '超声', '影像', '放射',
                           'OCT', 'PET', 'SPECT', '造影', 'DSA', '血管造影', '介入', '影像学',
                           '扫描', '断层', '透视', '胸片', '腹部平片'],
                'weight': 1.0,
                'synonyms': ['影像科', '放射科', '超声科']
            },
            'internal_medicine': {
                'keywords': ['内科', '高血压', '糖尿病', '心脏病', '肺病', '肝病', '肾病',
                           '内分泌', '消化', '呼吸', '循环', '血液', '风湿', '免疫',
                           '代谢', '心血管', '胃肠', '肝胆', '肾脏'],
                'weight': 1.0,
                'synonyms': ['内科学', '大内科']
            },
            'surgery': {
                'keywords': ['手术', '外科', '切除', '缝合', '麻醉', '术前', '术后', '手术室',
                           '开刀', '微创', '腹腔镜', '介入手术', '移植', '重建', '修复'],
                'weight': 1.0,
                'synonyms': ['外科学', '手术科']
            },
            'oncology': {
                'keywords': ['肿瘤', '癌症', '癌', '恶性', '良性', '转移', '化疗', '放疗',
                           '靶向治疗', '免疫治疗', '肿瘤标志物', '病理', '活检', '穿刺',
                           '肝癌', '肺癌', '胃癌', '乳腺癌', '结肠癌', '胰腺癌'],
                'weight': 1.2,
                'synonyms': ['肿瘤科', '癌症科']
            },
            'gastroenterology': {
                'keywords': ['消化', '胃', '肠', '肝', '胆', '胰腺', '脾', '食管', '十二指肠',
                           '胃镜', '肠镜', 'ERCP', 'MRCP', '胆管', '肝硬化', '肝炎',
                           '胆囊', '胆石', '消化道', '肠胃', '肝胆胰'],
                'weight': 1.1,
                'synonyms': ['消化科', '肝胆科', '胃肠科']
            },
            'cardiology': {
                'keywords': ['心脏', '心血管', '心电图', '心脏病', '冠心病', '心律',
                           '心肌', '心绞痛', '心梗', '心衰', '高血压', '动脉硬化',
                           '心脏彩超', '冠脉造影', '心导管', 'ECG', 'EKG'],
                'weight': 1.0,
                'synonyms': ['心内科', '心血管科']
            },
            'emergency': {
                'keywords': ['急诊', '急救', '抢救', '危重', '重症', 'ICU', 'CCU',
                           '休克', '昏迷', '外伤', '中毒', '窒息', '心跳骤停'],
                'weight': 1.3,
                'synonyms': ['急诊科', '急救科', 'ICU']
            },
            'laboratory': {
                'keywords': ['化验', '检验', '血常规', '尿常规', '生化', '免疫', '微生物',
                           '血糖', '肝功', '肾功', '电解质', '凝血', '感染指标',
                           '肿瘤标志物', '激素', '血脂', '血气分析'],
                'weight': 1.0,
                'synonyms': ['检验科', '化验科']
            }
        }

        # 医学实体识别模式
        self.entity_patterns = {
            'disease': [r'(\w*病\w*)', r'(\w*症\w*)', r'(\w*炎\w*)', r'(\w*癌\w*)', r'(\w*瘤\w*)'],
            'symptom': [r'(疼痛|头痛|胸痛|腹痛|恶心|呕吐|发热|咳嗽|气短|乏力)'],
            'examination': [r'(CT|MRI|X光|B超|胃镜|肠镜|心电图|血常规|尿常规|生化检查)'],
            'treatment': [r'(手术|化疗|放疗|药物治疗|保守治疗|介入治疗)'],
            'anatomy': [r'(心脏|肺部|肝脏|肾脏|胃|肠|胆囊|胰腺|脾脏|甲状腺)']
        }

    def classify_domain(self, query: str) -> DomainContext:
        """分类医学领域并提取上下文"""
        query_lower = query.lower()
        domain_scores = {}
        identified_entities = []

        # 计算每个领域的匹配分数
        for domain, info in self.domain_keywords.items():
            score = 0
            matched_terms = []

            for keyword in info['keywords']:
                if keyword.lower() in query_lower:
                    score += info['weight']
                    matched_terms.append(keyword)

            # 同义词匹配
            for synonym in info.get('synonyms', []):
                if synonym.lower() in query_lower:
                    score += info['weight'] * 0.8
                    matched_terms.append(synonym)

            if score > 0:
                domain_scores[domain] = score

        # 提取医学实体
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                identified_entities.extend(matches)

        # 确定主要领域
        if domain_scores:
            primary_domain = max(domain_scores, key=domain_scores.get)
            domain_confidence = min(1.0, domain_scores[primary_domain] / 5.0)
        else:
            primary_domain = 'general'
            domain_confidence = 0.1

        # 生成扩展术语
        expanded_terms = self._generate_expanded_terms(query, primary_domain, identified_entities)

        # 计算领域权重
        total_score = sum(domain_scores.values())
        domain_weights = {domain: score/total_score for domain, score in domain_scores.items()} if total_score > 0 else {}

        return DomainContext(
            primary_domain=primary_domain,
            domain_confidence=domain_confidence,
            entities=identified_entities,
            expanded_terms=expanded_terms,
            domain_weights=domain_weights
        )

    def _generate_expanded_terms(self, query: str, domain: str, entities: List[str]) -> List[str]:
        """生成扩展术语"""
        expanded = []

        # 基于领域的术语扩展
        if domain in self.domain_keywords:
            domain_info = self.domain_keywords[domain]
            expanded.extend(domain_info['keywords'][:5])
            expanded.extend(domain_info.get('synonyms', []))

        # 基于实体的扩展
        for entity in entities[:3]:
            if '病' in entity:
                expanded.append(entity.replace('病', '疾病'))
                expanded.append(entity + '症状')
                expanded.append(entity + '治疗')
            elif '癌' in entity or '瘤' in entity:
                expanded.append(entity + '诊断')
                expanded.append(entity + '分期')
                expanded.append(entity + '预后')

        return list(set(expanded))  # 去重

class EnhancedSemanticChunker:
    """增强版语义分块器 - 医学领域感知"""
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.domain_classifier = MedicalDomainClassifier()

    def clean_text(self, text: str) -> str:
        """清理文本"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef.,;:!?()（）。，；：！？]', '', text)
        return text.strip()

    def split_by_medical_sections(self, text: str) -> List[str]:
        """基于医学文档结构分割"""
        section_patterns = [
            r'(?:病史|现病史|既往史)[:：]',
            r'(?:体格检查|查体)[:：]',
            r'(?:辅助检查|实验室检查)[:：]',
            r'(?:影像学检查|影像学表现)[:：]',
            r'(?:诊断|临床诊断|最终诊断)[:：]',
            r'(?:治疗|处理|治疗方案)[:：]',
            r'(?:预后|随访)[:：]',
        ]

        sections = []
        current_section = ""

        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            is_new_section = any(re.search(pattern, line, re.IGNORECASE) for pattern in section_patterns)

            if is_new_section and current_section:
                sections.append(current_section.strip())
                current_section = line
            else:
                current_section += " " + line if current_section else line

        if current_section.strip():
            sections.append(current_section.strip())

        return sections if sections else [text]

    def chunk_text_with_domain_awareness(self, text: str, document_domain: str = None) -> List[Dict]:
        """领域感知的文本分块"""
        text = self.clean_text(text)
        sections = self.split_by_medical_sections(text)

        chunks_with_metadata = []

        for section_idx, section in enumerate(sections):
            section_domain_context = self.domain_classifier.classify_domain(section)

            if len(section) <= self.chunk_size:
                chunks_with_metadata.append({
                    'text': section,
                    'domain': section_domain_context.primary_domain,
                    'domain_confidence': section_domain_context.domain_confidence,
                    'entities': section_domain_context.entities,
                    'section_type': self._identify_section_type(section),
                    'medical_density': self._calculate_medical_density(section)
                })
            else:
                sub_chunks = self._split_long_section(section)
                for chunk in sub_chunks:
                    chunk_domain_context = self.domain_classifier.classify_domain(chunk)
                    chunks_with_metadata.append({
                        'text': chunk,
                        'domain': chunk_domain_context.primary_domain,
                        'domain_confidence': chunk_domain_context.domain_confidence,
                        'entities': chunk_domain_context.entities,
                        'section_type': self._identify_section_type(chunk),
                        'medical_density': self._calculate_medical_density(chunk)
                    })

        return chunks_with_metadata

    def _split_long_section(self, section: str) -> List[str]:
        """分割长章节"""
        sentences = re.split(r'[。！？.!?]+', section)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    words = current_chunk.split()
                    if len(words) > self.overlap:
                        overlap_text = ' '.join(words[-self.overlap:])
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _identify_section_type(self, text: str) -> str:
        """识别章节类型"""
        text_lower = text.lower()

        if any(keyword in text_lower for keyword in ['病史', '现病史', '既往史', '家族史']):
            return 'history'
        elif any(keyword in text_lower for keyword in ['体检', '查体', '体格检查']):
            return 'physical_exam'
        elif any(keyword in text_lower for keyword in ['检查', '化验', '影像', 'ct', 'mri']):
            return 'examination'
        elif any(keyword in text_lower for keyword in ['诊断', '疾病', '疑诊']):
            return 'diagnosis'
        elif any(keyword in text_lower for keyword in ['治疗', '手术', '药物', '处理']):
            return 'treatment'
        else:
            return 'general'

    def _calculate_medical_density(self, text: str) -> float:
        """计算医学术语密度"""
        words = re.findall(r'\w+', text.lower())
        if not words:
            return 0.0

        medical_count = 0
        medical_terms = set()

        for domain_info in self.domain_classifier.domain_keywords.values():
            medical_terms.update(keyword.lower() for keyword in domain_info['keywords'])

        for word in words:
            if word in medical_terms or any(med_term in word for med_term in medical_terms if len(med_term) >= 3):
                medical_count += 1

        return medical_count / len(words)

class DomainAwareVectorStore:
    """领域感知向量存储"""
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []
        self.metadata = []
        self.domain_indices = defaultdict(list)  # 按领域索引

    def add_documents(self, chunk_data: List[Dict], embeddings: np.ndarray, doc_metadata: List[Dict]):
        """添加文档到向量存储"""
        faiss.normalize_L2(embeddings)

        start_idx = len(self.chunks)
        self.index.add(embeddings.astype('float32'))

        for i, (chunk_info, doc_meta) in enumerate(zip(chunk_data, doc_metadata)):
            chunk_idx = start_idx + i

            combined_metadata = {
                **doc_meta,
                'domain': chunk_info['domain'],
                'domain_confidence': chunk_info['domain_confidence'],
                'entities': chunk_info['entities'],
                'section_type': chunk_info['section_type'],
                'medical_density': chunk_info['medical_density']
            }

            self.chunks.append(chunk_info['text'])
            self.metadata.append(combined_metadata)
            self.domain_indices[chunk_info['domain']].append(chunk_idx)

    def domain_aware_search(self, query_embedding: np.ndarray, domain_context: DomainContext,
                            k: int = 5, domain_boost: float = 1.5) -> List[Tuple[str, float, Dict]]:
        """领域感知搜索"""
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding.astype('float32'), k * 3)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                candidates.append((
                    self.chunks[idx],
                    float(score),
                    self.metadata[idx],
                    idx
                ))

        reranked_results = self._rerank_by_domain(candidates, domain_context, domain_boost)
        return reranked_results[:k]

    def _rerank_by_domain(self, candidates: List[Tuple[str, float, Dict, int]],
                          domain_context: DomainContext, domain_boost: float) -> List[Tuple[str, float, Dict]]:
        """基于领域上下文重排序"""
        enhanced_candidates = []

        for chunk, base_score, metadata, idx in candidates:
            enhanced_score = base_score

            chunk_domain = metadata.get('domain', 'general')
            if chunk_domain == domain_context.primary_domain:
                enhanced_score *= domain_boost
            elif chunk_domain in domain_context.domain_weights:
                enhanced_score *= (1 + domain_context.domain_weights[chunk_domain] * 0.5)

            medical_density = metadata.get('medical_density', 0)
            enhanced_score *= (1 + medical_density * 0.3)

            chunk_entities = set(metadata.get('entities', []))
            query_entities = set(domain_context.entities)
            entity_overlap = len(chunk_entities.intersection(query_entities))
            if entity_overlap > 0:
                enhanced_score *= (1 + entity_overlap * 0.2)

            section_type = metadata.get('section_type', 'general')
            if self._is_relevant_section(section_type, domain_context):
                enhanced_score *= 1.2

            enhanced_candidates.append((chunk, enhanced_score, metadata))

        enhanced_candidates.sort(key=lambda x: x[1], reverse=True)
        return enhanced_candidates

    def _is_relevant_section(self, section_type: str, domain_context: DomainContext) -> bool:
        """判断章节类型是否与查询领域相关"""
        relevance_map = {
            'radiology': ['examination', 'diagnosis'],
            'oncology': ['diagnosis', 'treatment', 'examination'],
            'surgery': ['treatment', 'physical_exam'],
            'laboratory': ['examination'],
            'emergency': ['history', 'physical_exam', 'diagnosis']
        }
        relevant_sections = relevance_map.get(domain_context.primary_domain, ['general'])
        return section_type in relevant_sections

    def save(self, path: str):
        """保存向量存储"""
        data = {
            'chunks': self.chunks,
            'metadata': self.metadata,
            'domain_indices': dict(self.domain_indices)
        }

        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """加载向量存储"""
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadata = data['metadata']
            self.domain_indices = defaultdict(list, data.get('domain_indices', {}))

class DocumentProcessor:
    """文档处理器，支持PDF和DOCX格式"""
    def __init__(self):
        pass

    def read_pdf(self, file_path: str) -> str:
        """读取PDF文件内容"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"读取PDF文件失败 {file_path}: {e}")
            return ""

    def read_docx(self, file_path: str) -> str:
        """读取DOCX文件内容"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"读取DOCX文件失败 {file_path}: {e}")
            return ""

    def load_documents(self, data_dir: str) -> Dict[str, str]:
        """加载指定目录下的所有文档"""
        documents = {}
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)

            if filename.lower().endswith('.pdf'):
                content = self.read_pdf(file_path)
            elif filename.lower().endswith('.docx'):
                content = self.read_docx(file_path)
            else:
                continue

            if content.strip():
                documents[filename] = content
                print(f"成功加载文档: {filename}")

        return documents

class EmbeddingModel:
    """嵌入模型"""
    def __init__(self, model_path: str = None):
        # 使用环境变量，避免硬编码个人路径
        model_path = model_path or os.getenv(
            "EMBED_MODEL_PATH",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        """平均池化"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """将文本编码为向量"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            all_embeddings.append(embeddings.numpy())
        return np.vstack(all_embeddings)

class LLMClient:
    """LLM客户端"""
    def __init__(self):
        # 改为读取环境变量，避免暴露个人信息
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")  # 可为空，走官方默认
        )
        # 可通过环境变量覆盖模型名
        self.model_name = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-R1")

    def generate_response(self, query: str, context: str = "", domain_context: DomainContext = None) -> str:
        """生成回答"""
        if context.strip():
            domain_info = ""
            if domain_context:
                ents_preview = ', '.join(domain_context.entities[:5]) if domain_context.entities else ''
                domain_info = f"\n检测到的医学领域: {domain_context.primary_domain}\n相关医学实体: {ents_preview}\n"

            prompt = f"""请结合以下检索到的相关文档信息和你自身的专业知识，全面回答用户的问题。{domain_info}

            检索到的相关信息：
            {context}
            
            问题：{query}
            
            请基于上述文档信息和你的专业知识提供准确、全面的回答："""
        else:
            prompt = f"""请基于你的专业知识回答以下问题：
            
            问题：{query}
            
            回答："""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': '你是一个专业的医学助手。请结合提供的文档信息（如果有）和你自身的医学知识，为用户提供准确、全面且有用的回答。'},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"生成回答时出错: {e}"

class EnhancedRAGSystem:
    """增强版RAG系统 - 领域感知检索"""
    def __init__(self, model_path: str = None):
        self.doc_processor = DocumentProcessor()
        self.chunker = EnhancedSemanticChunker(chunk_size=500, overlap=100)
        self.embedding_model = EmbeddingModel(model_path)
        self.vector_store = DomainAwareVectorStore()
        self.llm_client = LLMClient()
        self.domain_classifier = MedicalDomainClassifier()
        self.is_built = False

        # 调整后的阈值参数
        self.similarity_threshold = 0.4
        self.coverage_threshold = 0.3
        self.consistency_threshold = 0.4
        self.medical_threshold = 0.3
        self.domain_threshold = 0.25     # 新增：领域对齐阈值
        self.confidence_threshold = 0.5

    def build_index(self, data_dir: str, save_path: str = "enhanced_rag_index"):
        """构建增强索引"""
        print("正在加载文档...")
        documents = self.doc_processor.load_documents(data_dir)

        if not documents:
            print("未找到有效文档")
            return

        print("正在进行领域感知分割...")
        all_chunk_data = []
        all_metadata = []

        for doc_name, content in documents.items():
            print(f"处理文档: {doc_name}")
            chunk_data = self.chunker.chunk_text_with_domain_awareness(content)
            all_chunk_data.extend(chunk_data)
            for chunk_info in chunk_data:
                all_metadata.append({
                    'document': doc_name,
                    'chunk_length': len(chunk_info['text'])
                })

        print(f"总共分割出 {len(all_chunk_data)} 个领域感知文本块")

        domain_stats = Counter(chunk['domain'] for chunk in all_chunk_data)
        print("领域分布:")
        for domain, count in domain_stats.most_common():
            print(f"  {domain}: {count}")

        print("正在生成嵌入向量...")
        chunk_texts = [chunk['text'] for chunk in all_chunk_data]
        embeddings = self.embedding_model.encode(chunk_texts)

        print("正在构建领域感知向量索引...")
        self.vector_store.add_documents(all_chunk_data, embeddings, all_metadata)

        print("正在保存索引...")
        self.vector_store.save(save_path)

        self.is_built = True
        print("增强索引构建完成！")

    def load_index(self, save_path: str = "enhanced_rag_index"):
        """加载索引"""
        try:
            self.vector_store.load(save_path)
            self.is_built = True
            print("增强索引加载成功！")
        except Exception as e:
            print(f"加载索引失败: {e}")

    def _calculate_enhanced_confidence_metrics(self, query: str, search_results: List[Tuple[str, float, Dict]],
                                               domain_context: DomainContext) -> ConfidenceMetrics:
        """计算增强的置信度指标"""
        if not search_results:
            return ConfidenceMetrics(0, 0, 0, 0, 0, 0, 0)

        chunks = [result[0] for result in search_results]
        similarities = [result[1] for result in search_results]

        max_sim = max(similarities)
        avg_sim = sum(similarities) / len(similarities)

        # 覆盖度评分
        coverage = 0.0
        for i, sim in enumerate(similarities[:5]):
            position_weight = 1.0 / math.log2(i + 2)
            indicator = 1.0 if sim >= self.similarity_threshold else 0.0
            coverage += position_weight * indicator
        max_coverage = sum(1.0 / math.log2(i + 2) for i in range(min(len(similarities), 5)))
        coverage_score = coverage / max_coverage if max_coverage > 0 else 0.0

        # 一致性评分
        if len(similarities) <= 1:
            consistency = 1.0
        else:
            variance = sum((sim - avg_sim) ** 2 for sim in similarities) / len(similarities)
            consistency = 1.0 / (1.0 + variance)

        # 医学相关性（简化版本）
        medical_relevance = domain_context.domain_confidence

        # 新增：领域对齐度评分
        domain_alignment = self._calculate_domain_alignment(search_results, domain_context)

        # 权重调整：加入领域对齐度
        w1, w2, w3, w4, w5 = 0.35, 0.25, 0.15, 0.15, 0.1
        final_confidence = (w1 * max_sim + w2 * coverage_score + w3 * consistency +
                            w4 * medical_relevance + w5 * domain_alignment)

        return ConfidenceMetrics(
            max_similarity=max_sim,
            avg_similarity=avg_sim,
            coverage_score=coverage_score,
            consistency_score=consistency,
            medical_relevance=medical_relevance,
            domain_alignment=domain_alignment,
            final_confidence=final_confidence
        )

    def _calculate_domain_alignment(self, search_results: List[Tuple[str, float, Dict]],
                                    domain_context: DomainContext) -> float:
        """计算领域对齐度"""
        if not search_results:
            return 0.0

        alignment_score = 0.0
        for chunk, score, metadata in search_results:
            chunk_domain = metadata.get('domain', 'general')
            domain_confidence = metadata.get('domain_confidence', 0.0)

            if chunk_domain == domain_context.primary_domain:
                alignment_score += score * domain_confidence * 1.0
            elif chunk_domain in domain_context.domain_weights:
                weight = domain_context.domain_weights[chunk_domain]
                alignment_score += score * domain_confidence * weight * 0.7
            else:
                alignment_score += score * domain_confidence * 0.3

        return min(1.0, alignment_score / len(search_results))

    def _should_reject_enhanced(self, confidence_metrics: ConfidenceMetrics) -> Tuple[bool, str]:
        """增强的拒绝判断"""
        if confidence_metrics.final_confidence < self.confidence_threshold:
            return True, f"整体置信度过低 ({confidence_metrics.final_confidence:.3f} < {self.confidence_threshold})"

        if confidence_metrics.max_similarity < self.similarity_threshold:
            return True, f"文档相似度过低 ({confidence_metrics.max_similarity:.3f} < {self.similarity_threshold})"

        if confidence_metrics.domain_alignment < self.domain_threshold:
            return True, f"领域对齐度过低 ({confidence_metrics.domain_alignment:.3f} < {self.domain_threshold})"

        return False, "通过增强置信度检查"

    def query(self, question: str, top_k: int = 3) -> str:
        """增强查询系统"""
        if not self.is_built:
            return "请先构建或加载索引"

        print(f"查询: {question}")

        domain_context = self.domain_classifier.classify_domain(question)
        print(f"\n=== 领域分析 ===")
        print(f"主要领域: {domain_context.primary_domain}")
        print(f"领域置信度: {domain_context.domain_confidence:.3f}")
        print(f"识别实体: {domain_context.entities}")
        print(f"扩展术语: {domain_context.expanded_terms[:5]}")

        enhanced_query = question + " " + " ".join(domain_context.expanded_terms[:3])
        query_embedding = self.embedding_model.encode([enhanced_query])

        results = self.vector_store.domain_aware_search(
            query_embedding,
            domain_context,
            k=max(top_k, 5),
            domain_boost=1.5
        )

        confidence_metrics = self._calculate_enhanced_confidence_metrics(question, results, domain_context)

        print(f"\n=== 增强置信度分析 ===")
        print(f"最大相似度: {confidence_metrics.max_similarity:.3f}")
        print(f"平均相似度: {confidence_metrics.avg_similarity:.3f}")
        print(f"覆盖度评分: {confidence_metrics.coverage_score:.3f}")
        print(f"一致性评分: {confidence_metrics.consistency_score:.3f}")
        print(f"医学相关性: {confidence_metrics.medical_relevance:.3f}")
        print(f"领域对齐度: {confidence_metrics.domain_alignment:.3f}")
        print(f"最终置信度: {confidence_metrics.final_confidence:.3f}")

        should_reject, reason = self._should_reject_enhanced(confidence_metrics)

        if should_reject:
            print(f"\n❌ 拒绝回答: {reason}")
            return f"""抱歉，基于当前文档库在{domain_context.primary_domain}领域的内容，我无法为您的问题提供可靠的医学建议。

拒绝原因：{reason}

建议您咨询专业的{domain_context.primary_domain}科医生。"""

        else:
            print(f"\n✅ 通过增强检查: {reason}")

            relevant_results = [(chunk, score, metadata) for chunk, score, metadata in results[:top_k]
                                if score >= self.similarity_threshold * 0.8]

            if relevant_results:
                context_parts = []
                for i, (chunk, score, metadata) in enumerate(relevant_results):
                    domain_info = f"[{metadata.get('domain', 'general')}领域]"
                    context_parts.append(
                        f"文档片段{i+1}{domain_info}（来源：{metadata['document']}，相似度：{score:.3f}）：\n{chunk}\n"
                    )

                context = "\n".join(context_parts)

                print("检索到的相关文档片段：")
                for i, (chunk, score, metadata) in enumerate(relevant_results):
                    print(f"片段{i+1} (领域: {metadata.get('domain')}, 相似度: {score:.3f}):")
                    print(f"{chunk[:200]}...")
                    print("-" * 50)

                response = self.llm_client.generate_response(question, context, domain_context)
                confidence_note = f"\n\n[系统置信度: {confidence_metrics.final_confidence:.3f}/1.0 - 基于{domain_context.primary_domain}领域的可靠回答]"
                return response + confidence_note

            else:
                print("未检索到高相关性的文档片段...")
                response = self.llm_client.generate_response(question, "", domain_context)
                confidence_note = f"\n\n[系统置信度: {confidence_metrics.final_confidence:.3f}/1.0 - 基于模型知识的{domain_context.primary_domain}领域回答]"
                return response + confidence_note

def main():
    """主函数"""
    rag = EnhancedRAGSystem()

    data_dir = "./data"

    if os.path.exists("enhanced_rag_index.faiss") and os.path.exists("enhanced_rag_index.pkl"):
        print("发现已存在的增强索引，正在加载...")
        rag.load_index()
    else:
        print("未发现增强索引，开始构建新的领域感知索引...")
        rag.build_index(data_dir)

    print("\n" + "="*80)
    print("🚀 领域感知增强医学RAG系统已准备就绪！")
    print("📚 特性：")
    print("   • 医学领域自动识别和分类")
    print("   • 领域感知的智能检索")
    print("   • 多层次置信度评估")
    print("   • 专业领域自适应权重调整")
    print("输入 'quit' 或 'exit' 退出")
    print("="*80)

    while True:
        question = input("\n🔍 请输入您的医学问题: ").strip()

        if question.lower() in ['quit', 'exit', '退出']:
            print("👋 再见！")
            break

        if not question:
            continue

        print("\n🔬 正在进行领域感知分析和检索...")
        answer = rag.query(question)

        print(f"\n💡 回答：\n{answer}")
        print("\n" + "-"*80)

if __name__ == "__main__":
    main()
