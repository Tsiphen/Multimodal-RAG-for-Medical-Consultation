#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医学 VQA-RAG 系统 - 集成 HuggingFace Inference API 版本 + 完整评估系统
- 去除个人信息、绝对路径、硬编码 API key
- 环境变量统一配置：HF_TOKEN、HF_API_URL、HF_MODEL、数据与缓存路径等
"""

import os
import re
import sys
import warnings
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
import pandas as pd
warnings.filterwarnings('ignore')

# ===== 可配置项（环境变量，提供安全默认值）=====
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_API_URL = os.getenv(
    "HF_API_URL",
    "https://api-inference.huggingface.co/models/{model}"
)
HF_MODEL = os.getenv(
    "HF_MODEL",
    "mistralai/Mistral-7B-Instruct-v0.1"
)
HF_TIMEOUT = int(os.getenv("HF_TIMEOUT", "120"))

# 数据与缓存（相对路径，避免绝对路径）
TEXT_DATA_DIR = os.getenv("TEXT_DATA_DIR", "./medical_docs")
IMAGE_DATA_DIR = os.getenv("IMAGE_DATA_DIR", "./medical_images")

# Matplotlib 中文字体设置（按需生效，未安装也不报错）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 依赖检查（便于用户一键安装）
required_packages = {
    'torch': 'torch',
    'PIL': 'pillow',
    'cv2': 'opencv-python',
    'faiss': 'faiss-cpu',
    'transformers': 'transformers',
    'PyPDF2': 'PyPDF2',
    'docx': 'python-docx',
    'numpy': 'numpy',
    'openai': 'openai',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'pandas': 'pandas',
    'sklearn': 'scikit-learn'
}

missing_packages = []
for module, package in required_packages.items():
    try:
        if module == 'sklearn':
            __import__('sklearn')
        else:
            __import__(module)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print("缺少以下依赖包：")
    print(f"pip install {' '.join(missing_packages)}")
    sys.exit(1)

# ===== 导入依赖 =====
import torch
import numpy as np
import pickle
from typing import List, Dict, Tuple, Union
from transformers import AutoTokenizer, AutoModel
import PyPDF2
from docx import Document
import faiss
from PIL import Image
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

print("所有依赖已加载成功！")

# ========== HuggingFace API 客户端 ==========
def generate_answer(prompt: str) -> str:
    """使用 HuggingFace Inference API 生成回答（通过环境变量配置）"""
    if not HF_TOKEN:
        return "请先设置 HF_TOKEN 环境变量（示例：export HF_TOKEN=your_token）"
    url = HF_API_URL.format(model=HF_MODEL)

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": f"[INST] {prompt} [/INST]",
        "parameters": {
            "temperature": float(os.getenv("HF_TEMPERATURE", "0.7")),
            "max_new_tokens": int(os.getenv("HF_MAX_NEW_TOKENS", "256")),
            "do_sample": os.getenv("HF_DO_SAMPLE", "true").lower() == "true"
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=HF_TIMEOUT)
        response.raise_for_status()
        result = response.json()

        # 标准化解析
        if isinstance(result, list) and result and isinstance(result[0], dict):
            text = result[0].get("generated_text") or result[0].get("summary_text")
            if text:
                return text.split("[/INST]")[-1].strip() if "[/INST]" in text else text.strip()
        if isinstance(result, dict):
            if "generated_text" in result:
                return result["generated_text"].strip()
            if "error" in result:
                return f"模型报错: {result['error']}"
        return f"无法解析模型返回结果: {result}"
    except requests.RequestException as e:
        return f"API调用失败: {e}"

# ========== 评估数据集类 ==========
class MedicalEvaluationDataset:
    """医学评估数据集"""
    def __init__(self, domain="ophthalmology"):
        self.questions = []
        self.reference_answers = []
        self.categories = []
        self.difficulties = []
        self.patient_ids = []
        self.domain = domain
        self._create_domain_specific_dataset(domain)

    def _create_domain_specific_dataset(self, domain="ophthalmology"):
        if domain == "ophthalmology":
            ophthalmology_qa = [
                {
                    "question": "OCT检查的全称是什么？",
                    "reference": "OCT的全称是光学相干断层扫描(Optical Coherence Tomography)。",
                    "category": "Basic Knowledge",
                    "difficulty": "Easy",
                    "patient_id": "P001"
                },
                {
                    "question": "OCT检查主要用于检查眼部的哪个部位？",
                    "reference": "OCT检查主要用于检查视网膜，特别是黄斑区域的结构。",
                    "category": "Basic Knowledge",
                    "difficulty": "Easy",
                    "patient_id": "P001"
                },
                {
                    "question": "年龄相关性黄斑变性有哪两种主要类型？",
                    "reference": "年龄相关性黄斑变性主要分为两种类型：干性AMD和湿性AMD。",
                    "category": "Disease Classification",
                    "difficulty": "Easy",
                    "patient_id": "P001"
                },
                {
                    "question": "视网膜脱离的主要症状是什么？",
                    "reference": "视网膜脱离的主要症状包括闪光感、飞蚊症增多、视野缺失和视力下降。",
                    "category": "Clinical Symptoms",
                    "difficulty": "Medium",
                    "patient_id": "P002"
                },
                {
                    "question": "糖尿病视网膜病变是如何分期的？",
                    "reference": "糖尿病视网膜病变主要分为非增殖性糖尿病视网膜病变(NPDR)和增殖性糖尿病视网膜病变(PDR)两个阶段。",
                    "category": "Disease Staging",
                    "difficulty": "Medium",
                    "patient_id": "P002"
                },
                {
                    "question": "青光眼的主要危险因素有哪些？",
                    "reference": "青光眼的主要危险因素包括：高眼压、年龄增长、家族史、近视、种族因素等。",
                    "category": "Risk Factors",
                    "difficulty": "Medium",
                    "patient_id": "P003"
                },
                {
                    "question": "什么是黄斑水肿？",
                    "reference": "黄斑水肿是指视网膜黄斑区域发生液体积聚，导致黄斑增厚的病理状态。",
                    "category": "Disease Definition",
                    "difficulty": "Easy",
                    "patient_id": "P003"
                },
                {
                    "question": "OCT检查有哪些优点？",
                    "reference": "OCT检查的优点包括：无创检查、高分辨率成像、实时检查、可重复性好、能检测早期病变。",
                    "category": "Diagnostic Methods",
                    "difficulty": "Easy",
                    "patient_id": "P001"
                }
            ]
            all_qa = ophthalmology_qa
        elif domain == "general":
            general_qa = [
                {
                    "question": "高血压的诊断标准是什么？",
                    "reference": "高血压的诊断标准是收缩压≥140mmHg或舒张压≥90mmHg。",
                    "category": "Diagnostic Criteria",
                    "difficulty": "Easy",
                    "patient_id": "P004"
                },
                {
                    "question": "糖尿病的典型症状包括哪些？",
                    "reference": "糖尿病的典型症状包括多饮、多尿、多食和体重减轻，称为'三多一少'。",
                    "category": "Clinical Symptoms",
                    "difficulty": "Easy",
                    "patient_id": "P004"
                },
                {
                    "question": "什么是CT检查？",
                    "reference": "CT检查是计算机断层扫描，是一种使用X射线的医学影像检查方法。",
                    "category": "Diagnostic Methods",
                    "difficulty": "Easy",
                    "patient_id": "P005"
                },
                {
                    "question": "MRI检查的主要优点是什么？",
                    "reference": "MRI检查的主要优点是对软组织分辨率高、无电离辐射、可多角度成像。",
                    "category": "Diagnostic Methods",
                    "difficulty": "Easy",
                    "patient_id": "P005"
                }
            ]
            all_qa = general_qa
        else:
            return self._create_domain_specific_dataset("ophthalmology")

        for qa in all_qa:
            self.questions.append(qa["question"])
            self.reference_answers.append(qa["reference"])
            self.categories.append(qa["category"])
            self.difficulties.append(qa["difficulty"])
            self.patient_ids.append(qa["patient_id"])

    def get_all_data(self):
        return {
            'questions': self.questions,
            'reference_answers': self.reference_answers,
            'categories': self.categories,
            'difficulties': self.difficulties,
            'patient_ids': self.patient_ids
        }

    def get_patient_data(self, patient_id: str):
        indices = [i for i, pid in enumerate(self.patient_ids) if pid == patient_id]
        return {
            'questions': [self.questions[i] for i in indices],
            'reference_answers': [self.reference_answers[i] for i in indices],
            'categories': [self.categories[i] for i in indices],
            'difficulties': [self.difficulties[i] for i in indices]
        }

# ========== 改进的评估器 ==========
class EnhancedMedicalEvaluator:
    """增强的文本/医学评估指标"""
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,
            ngram_range=(1, 2)
        )
        self.medical_keywords = {
            '症状': 2.0, '诊断': 2.0, '治疗': 2.0, '病因': 1.5,
            '检查': 1.5, '药物': 1.5, '手术': 1.5, '并发症': 1.8,
            '预后': 1.3, '预防': 1.3, '疾病': 1.8, '患者': 1.2,
            '医学': 1.5, '临床': 1.5, '病理': 1.5, '影像': 1.5
        }

    def calculate_enhanced_bleu(self, reference: str, hypothesis: str) -> float:
        def get_ngrams(text, n):
            words = text.split()
            return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

        ref_clean = re.sub(r'[^\w\s]', '', reference.lower())
        hyp_clean = re.sub(r'[^\w\s]', '', hypothesis.lower())

        precisions = []
        for n in range(1, 5):
            ref_ngrams = get_ngrams(ref_clean, n)
            hyp_ngrams = get_ngrams(hyp_clean, n)
            if not hyp_ngrams:
                precisions.append(0.0)
                continue
            matches = 0
            for ngram in hyp_ngrams:
                if ngram in ref_ngrams:
                    matches += 1
                    for keyword in self.medical_keywords:
                        if keyword in ngram:
                            matches += (self.medical_keywords[keyword] - 1) * 0.1
            precision = matches / len(hyp_ngrams)
            precisions.append(precision)

        ref_len = len(ref_clean.split())
        hyp_len = len(hyp_clean.split())
        bp = min(1.0, np.exp(1 - ref_len / hyp_len)) if hyp_len > 0 else 0

        if all(p > 0 for p in precisions):
            bleu = bp * np.exp(np.mean([np.log(p) for p in precisions]))
        else:
            bleu = 0.0
        return min(1.0, bleu * 1.2)

    def calculate_enhanced_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        def get_words(text):
            return re.findall(r'\w+', text.lower())

        ref_words = get_words(reference)
        hyp_words = get_words(hypothesis)
        if not ref_words or not hyp_words:
            return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

        ref_1gram = set(ref_words)
        hyp_1gram = set(hyp_words)
        overlap_1 = len(ref_1gram.intersection(hyp_1gram))
        medical_bonus = sum(self.medical_keywords.get(word, 1.0) - 1.0
                            for word in ref_1gram.intersection(hyp_1gram)
                            if word in self.medical_keywords) * 0.1
        rouge_1 = (overlap_1 + medical_bonus) / len(ref_1gram) if ref_1gram else 0

        def get_bigrams(words):
            return [(words[i], words[i+1]) for i in range(len(words)-1)]

        ref_2gram = set(get_bigrams(ref_words))
        hyp_2gram = set(get_bigrams(hyp_words))
        overlap_2 = len(ref_2gram.intersection(hyp_2gram))
        rouge_2 = overlap_2 / len(ref_2gram) if ref_2gram else 0

        def lcs_length(a, b):
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]

        lcs_len = lcs_length(ref_words, hyp_words)
        rouge_l = lcs_len / len(ref_words) if ref_words else 0

        return {
            "rouge_1": min(1.0, rouge_1 * 1.1),
            "rouge_2": min(1.0, rouge_2 * 1.2),
            "rouge_l": min(1.0, rouge_l * 1.15)
        }

    def calculate_semantic_similarity(self, reference: str, hypothesis: str) -> float:
        try:
            texts = [reference, hypothesis]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            ref_words = set(re.findall(r'\w+', reference.lower()))
            hyp_words = set(re.findall(r'\w+', hypothesis.lower()))
            medical_terms_ref = {w for w in ref_words if w in self.medical_keywords}
            medical_terms_hyp = {w for w in hyp_words if w in self.medical_keywords}
            if medical_terms_ref:
                medical_overlap = len(medical_terms_ref & medical_terms_hyp) / len(medical_terms_ref)
                similarity = similarity * 0.7 + medical_overlap * 0.3
            return min(1.0, similarity * 1.1)
        except Exception as e:
            print(f"语义相似度计算失败: {e}")
            return 0.0

    def calculate_medical_accuracy(self, reference: str, hypothesis: str) -> float:
        medical_facts = {
            '诊断标准', '症状', '治疗方法', '药物', '剂量', '检查',
            '并发症', '预后', '病因', '发病机制', '临床表现'
        }
        ref_lower = reference.lower()
        hyp_lower = hypothesis.lower()
        fact_matches = 0
        total_facts = 0
        for fact in medical_facts:
            if fact in ref_lower:
                total_facts += 1
                if fact in hyp_lower:
                    fact_matches += 1

        ref_numbers = re.findall(r'\d+(?:\.\d+)?(?:mg|ml|mmHg|℃|%)?', reference)
        hyp_numbers = re.findall(r'\d+(?:\.\d+)?(?:mg|ml|mmHg|℃|%)?', hypothesis)
        number_accuracy = 0
        if ref_numbers:
            matching_numbers = sum(1 for num in ref_numbers if num in hyp_numbers)
            number_accuracy = matching_numbers / len(ref_numbers)

        if total_facts > 0:
            fact_accuracy = fact_matches / total_facts
            overall_accuracy = fact_accuracy * 0.7 + number_accuracy * 0.3
        else:
            overall_accuracy = number_accuracy
        return min(1.0, overall_accuracy * 1.2)

    def evaluate_comprehensive(self, reference: str, hypothesis: str) -> Dict[str, float]:
        bleu_score = self.calculate_enhanced_bleu(reference, hypothesis)
        rouge_scores = self.calculate_enhanced_rouge(reference, hypothesis)
        semantic_sim = self.calculate_semantic_similarity(reference, hypothesis)
        medical_acc = self.calculate_medical_accuracy(reference, hypothesis)

        weights = {
            'bleu': 0.25,
            'rouge_1': 0.20,
            'rouge_l': 0.15,
            'semantic_similarity': 0.25,
            'medical_accuracy': 0.15
        }
        comprehensive_score = (
            bleu_score * weights['bleu'] +
            rouge_scores['rouge_1'] * weights['rouge_1'] +
            rouge_scores['rouge_l'] * weights['rouge_l'] +
            semantic_sim * weights['semantic_similarity'] +
            medical_acc * weights['medical_accuracy']
        )

        calculation_details = {
            'weights': weights,
            'weighted_scores': {
                'bleu_weighted': bleu_score * weights['bleu'],
                'rouge_1_weighted': rouge_scores['rouge_1'] * weights['rouge_1'],
                'rouge_l_weighted': rouge_scores['rouge_l'] * weights['rouge_l'],
                'semantic_weighted': semantic_sim * weights['semantic_similarity'],
                'medical_weighted': medical_acc * weights['medical_accuracy']
            },
            'calculation_formula': f"{bleu_score:.3f}×{weights['bleu']} + "
                                   f"{rouge_scores['rouge_1']:.3f}×{weights['rouge_1']} + "
                                   f"{rouge_scores['rouge_l']:.3f}×{weights['rouge_l']} + "
                                   f"{semantic_sim:.3f}×{weights['semantic_similarity']} + "
                                   f"{medical_acc:.3f}×{weights['medical_accuracy']} = "
                                   f"{comprehensive_score:.3f}"
        }

        return {
            'bleu': bleu_score,
            'rouge_1': rouge_scores['rouge_1'],
            'rouge_2': rouge_scores['rouge_2'],
            'rouge_l': rouge_scores['rouge_l'],
            'semantic_similarity': semantic_sim,
            'medical_accuracy': medical_acc,
            'comprehensive_score': comprehensive_score,
            'calculation_details': calculation_details
        }

# ========== 基础文档处理类 ==========
class DocumentProcessor:
    def __init__(self):
        print("初始化文档处理器...")

    def read_pdf(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                return text
        except Exception as e:
            print(f"读取PDF失败: {e}")
            return ""

    def read_docx(self, file_path: str) -> str:
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += (paragraph.text or "") + "\n"
            return text
        except Exception as e:
            print(f"读取DOCX失败: {e}")
            return ""

    def load_documents(self, data_dir: str) -> Dict[str, str]:
        documents = {}
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            print(f"创建目录: {data_dir}")
            return documents

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

# ========== 文本分块 ==========
class SemanticChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        text = re.sub(r'\s+', ' ', text).strip()
        chunks = []
        sentences = re.split(r'[.!?。！？]', text)
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks if chunks else [text[:self.chunk_size]]

# ========== 文本编码器 ==========
class RobustTextEncoder:
    def __init__(self):
        print("初始化文本编码器...")
        self.dimension = 512
        try:
            model_name = os.getenv("SENTENCE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.use_transformer = True
            print(f"成功加载文本编码模型: {model_name}")
        except Exception as e:
            print(f"无法加载预训练模型，使用简化编码器: {e}")
            self.use_transformer = False

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if self.use_transformer:
            try:
                return self._encode_with_transformer(texts)
            except Exception as e:
                print(f"Transformer编码失败，使用简化方法: {e}")
                return self._encode_simple(texts)
        return self._encode_simple(texts)

    def _encode_with_transformer(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                if len(embedding) != self.dimension:
                    if len(embedding) > self.dimension:
                        embedding = embedding[:self.dimension]
                    else:
                        padding = np.zeros(self.dimension - len(embedding))
                        embedding = np.concatenate([embedding, padding])
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                embeddings.append(embedding)
        return np.array(embeddings, dtype=np.float32)

    def _encode_simple(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            text_hash = hash(text) % (2**31)
            np.random.seed(abs(text_hash))
            vec = np.random.randn(self.dimension).astype(np.float32)
            text_features = np.array([
                len(text) / 1000.0,
                text.count(' ') / 100.0,
                text.count('。') / 10.0,
                sum(ord(c) for c in text[:100]) / 100000.0
            ], dtype=np.float32)
            vec[:len(text_features)] += text_features
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)

# ========== 图像处理类 ==========
class MedicalImageProcessor:
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        print("初始化医学图像处理器...")

    def load_image(self, image_path: str) -> Image.Image:
        try:
            img = Image.open(image_path).convert('RGB')
            print(f"成功加载图像: {os.path.basename(image_path)}")
            return img
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            return None

    def extract_features(self, image: Image.Image) -> Dict:
        try:
            img_array = np.array(image)
            stats = {
                'mean': float(np.mean(img_array)),
                'std': float(np.std(img_array)),
                'min': float(np.min(img_array)),
                'max': float(np.max(img_array))
            }
            if len(img_array.shape) == 3:
                channel_stats = []
                for i in range(3):
                    channel_stats.append({
                        'mean': float(np.mean(img_array[:, :, i])),
                        'std': float(np.std(img_array[:, :, i]))
                    })
            else:
                channel_stats = [{'mean': stats['mean'], 'std': stats['std']}]

            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(np.sum(edges > 0) / edges.size)
            contrast = float(np.std(gray))
            hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
            hist_features = [float(h[0]) for h in hist]

            return {
                'stats': stats,
                'size': image.size,
                'channel_stats': channel_stats,
                'edge_density': edge_density,
                'contrast': contrast,
                'histogram': hist_features,
                'aspect_ratio': float(image.size[0] / image.size[1])
            }
        except Exception as e:
            print(f"特征提取失败: {e}")
            return {'stats': {'mean': 0, 'std': 0, 'min': 0, 'max': 0}, 'size': (0, 0)}

    def encode_image(self, image: Image.Image) -> np.ndarray:
        features = self.extract_features(image)
        feature_vector = []
        stats = features['stats']
        feature_vector.extend([
            stats['mean'] / 255.0,
            stats['std'] / 255.0,
            (stats['max'] - stats['min']) / 255.0,
            features.get('contrast', 0) / 255.0
        ])
        channel_stats = features.get('channel_stats', [])
        for i in range(3):
            if i < len(channel_stats):
                feature_vector.extend([
                    channel_stats[i]['mean'] / 255.0,
                    channel_stats[i]['std'] / 255.0
                ])
            else:
                feature_vector.extend([0.0, 0.0])
        feature_vector.extend([
            features.get('edge_density', 0),
            features.get('aspect_ratio', 1)
        ])
        hist = features.get('histogram', [0] * 16)
        hist_sum = sum(hist) if sum(hist) > 0 else 1
        normalized_hist = [h / hist_sum for h in hist]
        feature_vector.extend(normalized_hist)

        current_len = len(feature_vector)
        if current_len < 512:
            base_features = feature_vector.copy()
            while len(feature_vector) < 512:
                remaining = 512 - len(feature_vector)
                if remaining >= len(base_features):
                    feature_vector.extend(base_features)
                else:
                    feature_vector.extend(base_features[:remaining])

        feature_vector = feature_vector[:512]
        while len(feature_vector) < 512:
            feature_vector.append(0.0)

        vec = np.array(feature_vector, dtype=np.float32)
        if len(vec) != 512:
            if len(vec) > 512:
                vec = vec[:512]
            else:
                padding = np.zeros(512 - len(vec), dtype=np.float32)
                vec = np.concatenate([vec, padding])

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

# ========== 向量存储 ==========
class FixedVectorStore:
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []
        print(f"初始化向量存储，维度: {dimension}")

    def _ensure_proper_format(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[1] != self.dimension:
            adjusted = []
            for i in range(embeddings.shape[0]):
                vec = embeddings[i]
                if len(vec) > self.dimension:
                    vec = vec[:self.dimension]
                elif len(vec) < self.dimension:
                    padding = np.zeros(self.dimension - len(vec), dtype=np.float32)
                    vec = np.concatenate([vec, padding])
                adjusted.append(vec)
            embeddings = np.array(adjusted, dtype=np.float32)
        return embeddings

    def _normalize_vectors(self, embeddings: np.ndarray) -> np.ndarray:
        normalized = embeddings.copy()
        for i in range(normalized.shape[0]):
            n = np.linalg.norm(normalized[i])
            if n > 0:
                normalized[i] = normalized[i] / n
        return normalized

    def add_text_with_image_ref(self, text: str, text_embedding: np.ndarray,
                                image_ref: str, metadata: Dict):
        text_embedding = self._ensure_proper_format(text_embedding)
        text_embedding = self._normalize_vectors(text_embedding)
        self.index.add(text_embedding)
        meta = metadata.copy()
        meta.update({'text': text, 'image_ref': image_ref, 'vector_id': len(self.metadata)})
        self.metadata.append(meta)

    def add(self, embeddings: np.ndarray, metadata: List[Dict]):
        embeddings = self._ensure_proper_format(embeddings)
        embeddings = self._normalize_vectors(embeddings)
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        query_embedding = self._ensure_proper_format(query_embedding)
        query_embedding = self._normalize_vectors(query_embedding)
        k = min(k, self.index.ntotal, len(self.metadata))
        if k <= 0:
            return []
        scores, indices = self.index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                r = self.metadata[idx].copy()
                r['score'] = float(score)
                results.append(r)
        return results

# ========== 主系统 ==========
class MedicalVQARAGSystem:
    def __init__(self):
        print("初始化医学VQA-RAG系统...")
        self.doc_processor = DocumentProcessor()
        self.chunker = SemanticChunker()
        self.text_encoder = RobustTextEncoder()
        self.image_processor = MedicalImageProcessor()
        self.vqa_store = FixedVectorStore()
        self.evaluator = EnhancedMedicalEvaluator()
        self.is_built = False

    def build_multimodal_index(self, text_data_dir: str = TEXT_DATA_DIR,
                               image_data_dir: str = IMAGE_DATA_DIR):
        print("\n构建新的医学VQA索引...")
        for dir_path in [text_data_dir, image_data_dir]:
            os.makedirs(dir_path, exist_ok=True)

        print("处理文本文档...")
        documents = self.doc_processor.load_documents(text_data_dir)

        print("处理医学图像...")
        image_features = {}
        image_count = 0

        def find_images_recursively(directory):
            image_files = []
            if os.path.exists(directory):
                for root, _, files in os.walk(directory):
                    for filename in files:
                        if any(filename.lower().endswith(fmt) for fmt in self.image_processor.supported_formats):
                            full_path = os.path.join(root, filename)
                            relative_path = os.path.relpath(full_path, directory)
                            image_files.append((filename, full_path, relative_path))
            return image_files

        found_images = find_images_recursively(image_data_dir)
        for filename, img_path, relative_path in found_images:
            image = self.image_processor.load_image(img_path)
            if image:
                features = self.image_processor.extract_features(image)
                key = f"{relative_path}"
                image_features[key] = {
                    'path': img_path,
                    'features': features,
                    'filename': filename,
                    'relative_path': relative_path
                }
                image_count += 1
                print(f"找到图像: {relative_path}")
        print(f"找到 {image_count} 个医学图像")

        print("处理文本并建立多模态关联...")
        for doc_name, content in documents.items():
            chunks = self.chunker.chunk_text(content)
            for i, chunk in enumerate(chunks):
                try:
                    text_embedding = self.text_encoder.encode(chunk)
                    img_ref = self._find_related_image(chunk, image_features)
                    metadata = {
                        'type': 'text',
                        'source': doc_name,
                        'chunk_id': i,
                        'content': chunk,
                        'image_ref': img_ref
                    }
                    self.vqa_store.add_text_with_image_ref(chunk, text_embedding, img_ref, metadata)
                except Exception as e:
                    print(f"处理文本块 {i} 失败: {e}")

        print("添加图像向量到索引...")
        for img_key, img_data in image_features.items():
            try:
                image = self.image_processor.load_image(img_data['path'])
                if image:
                    img_embedding = self.image_processor.encode_image(image)
                    if len(img_embedding) != 512:
                        if len(img_embedding) > 512:
                            img_embedding = img_embedding[:512]
                        else:
                            padding = np.zeros(512 - len(img_embedding), dtype=np.float32)
                            img_embedding = np.concatenate([img_embedding, padding])
                    features = img_data['features']
                    filename = img_data['filename']
                    relative_path = img_data['relative_path']
                    description = f"医学图像 {filename} (路径: {relative_path})，尺寸: {features['size']}"
                    metadata = {
                        'type': 'image',
                        'source': img_key,
                        'filename': filename,
                        'path': img_data['path'],
                        'relative_path': relative_path,
                        'features': features,
                        'description': description
                    }
                    self.vqa_store.add(img_embedding.reshape(1, -1), [metadata])
            except Exception as e:
                print(f"处理图像 {img_key} 失败: {e}")

        self.is_built = True
        print(f"多模态索引构建完成！总计 {self.vqa_store.index.ntotal} 个向量")

    def _find_related_image(self, text: str, image_features: Dict) -> str:
        text_lower = text.lower()
        medical_keywords = ['图', '图像', '影像', 'x光', 'ct', 'mri', 'oct', '眼底', '视网膜', '黄斑', '视盘']
        if any(k in text_lower for k in medical_keywords):
            for img_key in image_features.keys():
                return img_key
        for img_key, img_data in image_features.items():
            img_name_lower = img_data.get('filename', '').lower()
            relative_path_lower = img_data.get('relative_path', '').lower()
            if 'oct' in text_lower and 'oct' in (img_name_lower + relative_path_lower):
                return img_key
            if any(k in text_lower for k in ['眼', '视网膜', '黄斑', '视盘']) and \
               any(k in (img_name_lower + relative_path_lower) for k in ['eye', 'retina', 'oct', 'fundus']):
                return img_key
        return ""

    def query(self, question: str, k: int = 5) -> str:
        if not self.is_built:
            return "请先构建索引！输入 'rebuild' 以重建索引。"
        try:
            print(f"查询问题: {question}")
            query_embedding = self.text_encoder.encode(question)
            results = self.vqa_store.search(query_embedding, k=k)
            if not results:
                return "未找到相关信息，请检查索引是否正确构建。"
            context_parts = []
            for i, result in enumerate(results[:3]):
                score = result.get('score', 0)
                if result.get('type') == 'text':
                    content = result.get('content', '')[:300]
                    source = result.get('source', 'unknown')
                    context_parts.append(f"相关文档 {i+1} (来源: {source}, 相似度: {score:.3f}):\n{content}")
                    img_ref = result.get('image_ref', '')
                    if img_ref:
                        context_parts.append(f"相关图像: {img_ref}")
                elif result.get('type') == 'image':
                    source = result.get('source', 'unknown')
                    description = result.get('description', '')
                    context_parts.append(f"相关图像 {i+1} (文件: {source}, 相似度: {score:.3f}):\n{description}")
            context = "\n\n".join(context_parts)
            prompt = f"""你是一个专业的医学助手。基于以下相关信息回答用户的医学问题。

相关医学资料:
{context}

用户问题: {question}

请提供准确、专业的医学回答，如果信息不足请说明："""
            return generate_answer(prompt)
        except Exception as e:
            return f"查询过程中出现错误: {e}"

    def query_with_image(self, image_path: str, question: str) -> str:
        try:
            if not os.path.exists(image_path):
                return "图像文件不存在。"
            print(f"分析图像: {image_path}")
            image = self.image_processor.load_image(image_path)
            if not image:
                return "无法加载图像文件。"
            features = self.image_processor.extract_features(image)
            related_info = ""
            if self.is_built:
                img_embedding = self.image_processor.encode_image(image)
                results = self.vqa_store.search(img_embedding.reshape(1, -1), k=3)
                if results:
                    parts = []
                    for result in results[:2]:
                        if result.get('type') == 'text':
                            parts.append(f"相关医学文档: {result.get('content','')[:200]}")
                        elif result.get('type') == 'image':
                            parts.append(f"相似医学图像: {result.get('source','')}")
                    if parts:
                        related_info = "\n\n相关医学资料:\n" + "\n".join(parts)
            prompt = f"""你是一个专业的医学图像分析师。请分析这张医学图像并回答问题。

图像信息:
- 文件名: {os.path.basename(image_path)}
- 图像尺寸: {features['size']}
- 像素统计: 平均值={features['stats']['mean']:.1f}, 标准差={features['stats']['std']:.1f}
- 对比度: {features.get('contrast', 0):.1f}
- 边缘密度: {features.get('edge_density', 0):.3f}
- 长宽比: {features.get('aspect_ratio', 1):.2f}
{related_info}

用户问题: {question}

请提供专业的医学图像分析和回答："""
            return generate_answer(prompt)
        except Exception as e:
            return f"图像分析过程中出现错误: {e}"

    def run_comprehensive_evaluation(self, evaluation_scope: str = "all", domain: str = "ophthalmology") -> Dict:
        print(f"\n🔬 开始运行医学VQA系统评估")
        print(f"📊 评估范围: {evaluation_scope}")
        print(f"🏥 医学领域: {domain}")

        eval_dataset = MedicalEvaluationDataset(domain=domain)
        if evaluation_scope == "single":
            eval_data = eval_dataset.get_patient_data("P001")
            print(f"🎯 评估范围: 单个患者 (P001) - {domain}领域")
        else:
            eval_data = eval_dataset.get_all_data()
            print(f"🎯 评估范围: 所有患者数据 - {domain}领域")

        if not eval_data['questions']:
            print("❌ 没有找到评估数据！")
            return {}

        all_results = []
        category_results = defaultdict(list)
        difficulty_results = defaultdict(list)
        detailed_calculations = []

        print(f"📝 开始评估 {len(eval_data['questions'])} 个问题...")
        for i, (question, reference, category, difficulty) in enumerate(
            zip(eval_data['questions'], eval_data['reference_answers'],
                eval_data['categories'], eval_data['difficulties'])):

            print(f"\n🔍 评估问题 {i+1}/{len(eval_data['questions'])}")
            print(f"❓ 问题: {question[:60]}...")

            try:
                generated_answer = self.query(question, k=5)
                print(f"🤖 生成回答: {generated_answer[:80]}...")
                metrics = self.evaluator.evaluate_comprehensive(reference, generated_answer)
                calc_details = metrics.get('calculation_details', {})
                if calc_details:
                    print(f"📊 综合得分计算:")
                    print(f"   公式: {calc_details.get('calculation_formula', 'N/A')}")
                    print(f"   结果: {metrics['comprehensive_score']:.3f}")

                result = {
                    'question': question,
                    'reference': reference,
                    'generated': generated_answer,
                    'category': category,
                    'difficulty': difficulty,
                    'metrics': metrics
                }
                all_results.append(result)
                detailed_calculations.append(calc_details)
                category_results[category].append(metrics)
                difficulty_results[difficulty].append(metrics)
                print(f"✅ 综合得分: {metrics['comprehensive_score']:.3f}")

            except Exception as e:
                print(f"❌ 评估问题 {i+1} 时出错: {e}")

        overall_stats = self._calculate_overall_statistics(all_results)
        category_stats = self._calculate_category_statistics(category_results)
        difficulty_stats = self._calculate_difficulty_statistics(difficulty_results)
        weight_analysis = self._analyze_weight_contribution(detailed_calculations)

        evaluation_results = {
            'evaluation_scope': evaluation_scope,
            'domain': domain,
            'total_questions': len(eval_data['questions']),
            'successful_evaluations': len(all_results),
            'overall_statistics': overall_stats,
            'category_statistics': category_stats,
            'difficulty_statistics': difficulty_stats,
            'weight_analysis': weight_analysis,
            'detailed_results': all_results,
            'detailed_calculations': detailed_calculations,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        self._print_evaluation_summary(evaluation_results)
        self._generate_evaluation_charts(evaluation_results)
        self._save_evaluation_results(evaluation_results)
        return evaluation_results

    def _calculate_overall_statistics(self, results: List[Dict]) -> Dict:
        if not results:
            return {}
        all_metrics = {
            'bleu': [], 'rouge_1': [], 'rouge_2': [], 'rouge_l': [],
            'semantic_similarity': [], 'medical_accuracy': [], 'comprehensive_score': []
        }
        for r in results:
            m = r['metrics']
            for k in all_metrics:
                if k in m:
                    all_metrics[k].append(m[k])
        stats = {}
        for metric, values in all_metrics.items():
            if values:
                stats[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        return stats

    def _calculate_category_statistics(self, category_results: Dict) -> Dict:
        out = {}
        for cat, lst in category_results.items():
            if lst:
                scores = [m['comprehensive_score'] for m in lst]
                out[cat] = {
                    'count': len(lst),
                    'mean_score': float(np.mean(scores)),
                    'std_score': float(np.std(scores)),
                    'min_score': float(np.min(scores)),
                    'max_score': float(np.max(scores))
                }
        return out

    def _calculate_difficulty_statistics(self, difficulty_results: Dict) -> Dict:
        out = {}
        for diff, lst in difficulty_results.items():
            if lst:
                scores = [m['comprehensive_score'] for m in lst]
                out[diff] = {
                    'count': len(lst),
                    'mean_score': float(np.mean(scores)),
                    'std_score': float(np.std(scores)),
                    'min_score': float(np.min(scores)),
                    'max_score': float(np.max(scores))
                }
        return out

    def _analyze_weight_contribution(self, detailed_calculations: List[Dict]) -> Dict:
        if not detailed_calculations:
            return {}
        total = {'bleu_weighted': 0, 'rouge_1_weighted': 0, 'rouge_l_weighted': 0,
                 'semantic_weighted': 0, 'medical_weighted': 0}
        valid = [c for c in detailed_calculations if c.get('weighted_scores')]
        for calc in valid:
            ws = calc['weighted_scores']
            for k in total:
                total[k] += ws.get(k, 0)
        if not valid:
            return {}
        avg = {k: v/len(valid) for k, v in total.items()}
        s = sum(avg.values())
        pct = {k: (v/s*100 if s > 0 else 0) for k, v in avg.items()}
        return {
            'average_contributions': avg,
            'contribution_percentages': pct,
            'sample_calculation': valid[0] if valid else {}
        }

    def _print_evaluation_summary(self, results: Dict):
        print(f"\n{'='*70}")
        print(f"📊 MEDICAL VQA-RAG EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"🏥 Domain: {results['domain'].upper()}")
        print(f"📋 Scope: {results['evaluation_scope']}")
        print(f"❓ Total Questions: {results['total_questions']}")
        print(f"✅ Successful Evaluations: {results['successful_evaluations']}")
        print(f"📈 Success Rate: {results['successful_evaluations']/results['total_questions']*100:.1f}%")

        overall_stats = results.get('overall_statistics', {})
        if overall_stats:
            print(f"\n📊 OVERALL METRICS (Mean ± Std):")
            for metric, s in overall_stats.items():
                if isinstance(s, dict) and 'mean' in s:
                    print(f"  • {metric.upper().replace('_', '-')}: {s['mean']:.3f} ± {s['std']:.3f}")

        weight_analysis = results.get('weight_analysis', {})
        if weight_analysis.get('contribution_percentages'):
            cp = weight_analysis['contribution_percentages']
            print(f"\n⚖️  WEIGHT CONTRIBUTION ANALYSIS:")
            print(f"  • BLEU (25%): {cp.get('bleu_weighted', 0):.1f}%")
            print(f"  • ROUGE-1 (20%): {cp.get('rouge_1_weighted', 0):.1f}%")
            print(f"  • ROUGE-L (15%): {cp.get('rouge_l_weighted', 0):.1f}%")
            print(f"  • Semantic (25%): {cp.get('semantic_weighted', 0):.1f}%")
            print(f"  • Medical (15%): {cp.get('medical_weighted', 0):.1f}%")

        sample_calc = weight_analysis.get('sample_calculation', {})
        if sample_calc.get('calculation_formula'):
            print(f"\n🧮 SAMPLE CALCULATION:")
            print(f"  Formula: {sample_calc['calculation_formula']}")

        print(f"\n⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

    def _generate_evaluation_charts(self, results: Dict):
        try:
            timestamp = results['timestamp']
            domain = results.get('domain', 'unknown')
            scope = results['evaluation_scope']

            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'🏥 Medical VQA-RAG Evaluation Report - {domain.upper()} ({scope})',
                         fontsize=16, fontweight='bold')

            ax1 = axes[0, 0]
            overall_stats = results['overall_statistics']
            metrics = ['bleu', 'rouge_1', 'rouge_l', 'semantic_similarity', 'medical_accuracy', 'comprehensive_score']
            metric_names = ['BLEU', 'ROUGE-1', 'ROUGE-L', 'Semantic Sim', 'Medical Acc', 'Comprehensive']
            means = [overall_stats.get(m, {}).get('mean', 0) for m in metrics]
            bars = ax1.bar(metric_names, means)
            ax1.set_title('📊 Overall Evaluation Metrics', fontweight='bold')
            ax1.set_ylabel('Score'); ax1.set_ylim(0, 1.0)
            for bar, mean in zip(bars, means):
                ax1.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.01,
                         f'{mean:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

            ax2 = axes[0, 1]
            category_stats = results['category_statistics']
            if category_stats:
                cats = list(category_stats.keys())
                scores = [category_stats[c]['mean_score'] for c in cats]
                bars = ax2.bar(cats, scores)
                ax2.set_title('🏷️ Performance by Category', fontweight='bold')
                ax2.set_ylabel('Average Comprehensive Score'); ax2.set_ylim(0, 1.0)
                for bar, sc in zip(bars, scores):
                    ax2.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.01,
                             f'{sc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

            ax3 = axes[0, 2]
            difficulty_stats = results['difficulty_statistics']
            if difficulty_stats:
                diffs = list(difficulty_stats.keys())
                dscores = [difficulty_stats[d]['mean_score'] for d in diffs]
                bars = ax3.bar(diffs, dscores)
                ax3.set_title('🎯 Performance by Difficulty', fontweight='bold')
                ax3.set_ylabel('Average Comprehensive Score'); ax3.set_ylim(0, 1.0)
                for bar, sc in zip(bars, dscores):
                    ax3.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.01,
                             f'{sc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

            ax4 = axes[1, 0]
            detailed_results = results['detailed_results']
            if detailed_results:
                metric_data, metric_labels = [], []
                for metric in ['bleu', 'rouge_1', 'semantic_similarity', 'medical_accuracy']:
                    values = [r['metrics'][metric] for r in detailed_results if metric in r['metrics']]
                    if values:
                        metric_data.append(values); metric_labels.append(metric.upper().replace('_', '-'))
                if metric_data:
                    bp = ax4.boxplot(metric_data, labels=metric_labels, patch_artist=True)
                    ax4.set_title('📦 Metrics Distribution', fontweight='bold')
                    ax4.set_ylabel('Score'); ax4.set_ylim(0, 1.0)
                    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

            ax5 = axes[1, 1]
            if detailed_results:
                comp_scores = [r['metrics']['comprehensive_score'] for r in detailed_results]
                ax5.hist(comp_scores, bins=10, alpha=0.7, edgecolor='black')
                ax5.axvline(np.mean(comp_scores), color='red', linestyle='--', linewidth=2,
                            label=f'Mean: {np.mean(comp_scores):.3f}')
                ax5.set_title('📈 Comprehensive Score Distribution', fontweight='bold')
                ax5.set_xlabel('Comprehensive Score'); ax5.set_ylabel('Frequency'); ax5.legend()

            ax6 = axes[1, 2]; ax6.axis('off')
            weight_analysis = results.get('weight_analysis', {})
            contrib_text = ""
            if weight_analysis.get('contribution_percentages'):
                cp = weight_analysis['contribution_percentages']
                contrib_text = (
                    f"\nWeight Contribution Analysis:\n"
                    f"• BLEU (25%): {cp.get('bleu_weighted', 0):.1f}%\n"
                    f"• ROUGE-1 (20%): {cp.get('rouge_1_weighted', 0):.1f}%\n"
                    f"• ROUGE-L (15%): {cp.get('rouge_l_weighted', 0):.1f}%\n"
                    f"• Semantic (25%): {cp.get('semantic_weighted', 0):.1f}%\n"
                    f"• Medical (15%): {cp.get('medical_weighted', 0):.1f}%"
                )
            overall_stats = results['overall_statistics']
            stats_text = (
                f"📋 Evaluation Summary\n\n"
                f"🏥 Domain: {results['domain'].upper()}\n"
                f"📊 Scope: {results['evaluation_scope']}\n"
                f"❓ Total Questions: {results['total_questions']}\n"
                f"✅ Success Rate: {results['successful_evaluations']/results['total_questions']*100:.1f}%\n\n"
                f"📊 Main Metrics (Mean):\n"
                f"• BLEU: {overall_stats.get('bleu', {}).get('mean', 0):.3f}\n"
                f"• ROUGE-1: {overall_stats.get('rouge_1', {}).get('mean', 0):.3f}\n"
                f"• Semantic Similarity: {overall_stats.get('semantic_similarity', {}).get('mean', 0):.3f}\n"
                f"• Medical Accuracy: {overall_stats.get('medical_accuracy', {}).get('mean', 0):.3f}\n"
                f"• Comprehensive Score: {overall_stats.get('comprehensive_score', {}).get('mean', 0):.3f}\n"
                f"{contrib_text}\n\n"
                f"⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=9,
                     va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            plt.tight_layout()
            chart_filename = f'evaluation_charts_{domain}_{scope}_{timestamp}.png'
            plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
            print(f"📊 Evaluation charts saved: {chart_filename}")
            plt.show()
        except Exception as e:
            print(f"❌ Error generating evaluation charts: {e}")
            import traceback; traceback.print_exc()

    def _save_evaluation_results(self, results: Dict):
        try:
            timestamp = results['timestamp']
            filename = f'evaluation_results_{timestamp}.json'
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, dict): return {k: convert_numpy(v) for k, v in obj.items()}
                if isinstance(obj, list): return [convert_numpy(x) for x in obj]
                return obj
            serializable = convert_numpy(results)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            print(f"评估结果已保存: {filename}")
        except Exception as e:
            print(f"保存评估结果时出错: {e}")

# ========== 主函数 ==========
def main():
    print("启动医学 VQA-RAG 系统...")

    if not HF_TOKEN:
        print("❌ 请先设置 HF_TOKEN 环境变量")
        print("示例: export HF_TOKEN=your_huggingface_token")
        return
    print("✅ HuggingFace Token 已配置")

    vqa_rag = MedicalVQARAGSystem()
    vqa_rag.build_multimodal_index(TEXT_DATA_DIR, IMAGE_DATA_DIR)

    print("\n" + "="*70)
    print("🏥 Medical VQA-RAG System Ready!")
    print("="*70)
    print("Available Commands:")
    print("• 直接输入医学问题进行检索问答")
    print("• 'image:<image_path> <question>' 进行图像分析")
    print("• 'rebuild' 重新构建索引")
    print("• 'eval_oph' / 'eval_oph_single' 眼科评估（全部/单患者）")
    print("• 'eval_general' / 'eval_general_single' 通用评估（全部/单患者）")
    print("• 'quit' 退出")
    print("="*70)

    while True:
        try:
            user_input = input("\n🔍 Medical Assistant> ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using Medical VQA-RAG System!")
                break
            if user_input.lower() == 'rebuild':
                print("🔄 Rebuilding index...")
                vqa_rag = MedicalVQARAGSystem()
                vqa_rag.build_multimodal_index(TEXT_DATA_DIR, IMAGE_DATA_DIR)
                continue
            if user_input.lower() == 'eval_oph':
                print("🔬 Starting comprehensive ophthalmology evaluation (all patients)...")
                results = vqa_rag.run_comprehensive_evaluation("all", "ophthalmology")
                mean_score = results.get('overall_statistics', {}).get('comprehensive_score', {}).get('mean', 0)
                print(f"✅ Evaluation completed! Comprehensive score: {mean_score:.3f}")
                continue
            if user_input.lower() == 'eval_oph_single':
                print("🔬 Starting single patient ophthalmology evaluation...")
                results = vqa_rag.run_comprehensive_evaluation("single", "ophthalmology")
                mean_score = results.get('overall_statistics', {}).get('comprehensive_score', {}).get('mean', 0)
                print(f"✅ Single patient evaluation completed! Comprehensive score: {mean_score:.3f}")
                continue
            if user_input.lower() == 'eval_general':
                print("🔬 Starting comprehensive general medicine evaluation (all patients)...")
                results = vqa_rag.run_comprehensive_evaluation("all", "general")
                mean_score = results.get('overall_statistics', {}).get('comprehensive_score', {}).get('mean', 0)
                print(f"✅ Evaluation completed! Comprehensive score: {mean_score:.3f}")
                continue
            if user_input.lower() == 'eval_general_single':
                print("🔬 Starting single patient general medicine evaluation...")
                results = vqa_rag.run_comprehensive_evaluation("single", "general")
                mean_score = results.get('overall_statistics', {}).get('comprehensive_score', {}).get('mean', 0)
                print(f"✅ Single patient evaluation completed! Comprehensive score: {mean_score:.3f}")
                continue

            if user_input.startswith('image:'):
                parts = user_input[6:].strip().split(' ', 1)
                if len(parts) >= 2:
                    img_path, question = parts[0], parts[1]
                else:
                    img_path = parts[0]
                    question = input("📋 Please enter your question about the image> ").strip()
                if question:
                    print("🔬 Analyzing medical image...")
                    answer = vqa_rag.query_with_image(img_path, question)
                    print(f"\n📊 Analysis result:\n{answer}")
                else:
                    print("❌ Please provide a question")
            elif user_input:
                print("🔍 Searching relevant medical literature...")
                answer = vqa_rag.query(user_input)
                print(f"\n💡 Answer:\n{answer}")

        except KeyboardInterrupt:
            print("\n\n👋 系统中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            print("请重试或输入 'quit' 退出")

if __name__ == "__main__":
    main()
