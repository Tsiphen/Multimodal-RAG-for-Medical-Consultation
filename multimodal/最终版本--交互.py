import os

# =========================
# 环境与缓存（可用环境变量覆盖）
# =========================
os.environ['HF_ENDPOINT'] = os.getenv('HF_ENDPOINT', 'https://huggingface.co')
os.environ['TRANSFORMERS_CACHE'] = os.getenv('TRANSFORMERS_CACHE', './models_cache')
os.environ['HF_HOME'] = os.getenv('HF_HOME', './hf_cache')

import pandas as pd
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import base64
from io import BytesIO
from openai import OpenAI
from typing import List, Dict, Tuple
import json
import re
from torch.nn.functional import cosine_similarity


class TransformersCLIPMedicalRAG:
    """基于 HuggingFace Transformers CLIP 的医学图像RAG系统（训练集版本）"""

    def __init__(self, train_csv_path: str, data_dir: str, model_cache_dir: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")

        # ====== 模型缓存目录（通过环境变量或入参配置，避免硬编码绝对路径）======
        self.model_cache_dir = model_cache_dir or os.getenv("MODEL_CACHE_DIR", "./model_cache")
        os.makedirs(self.model_cache_dir, exist_ok=True)
        print(f"模型缓存目录: {self.model_cache_dir}")

        # ====== 尝试多个 CLIP 模型，从大到小 ======
        model_options = [
            "openai/clip-vit-large-patch14",
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
        ]

        self.clip_model = None
        self.clip_processor = None
        self.model_name = None

        for model_name in model_options:
            try:
                print(f"[INFO] 尝试加载模型: {model_name}")
                local_model_path = os.path.join(self.model_cache_dir, model_name.replace("/", "_"))

                if os.path.exists(local_model_path):
                    print(f"[CACHE] 从本地缓存加载: {local_model_path}")
                    self.clip_model = CLIPModel.from_pretrained(local_model_path).to(self.device)
                    self.clip_processor = CLIPProcessor.from_pretrained(local_model_path)
                else:
                    print(f"[DOWNLOAD] 从HuggingFace下载: {model_name}")
                    self.clip_model = CLIPModel.from_pretrained(
                        model_name,
                        cache_dir=self.model_cache_dir,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                    ).to(self.device)
                    self.clip_processor = CLIPProcessor.from_pretrained(
                        model_name,
                        cache_dir=self.model_cache_dir
                    )
                    # 首次下载后保存到本地缓存目录
                    print(f"[SAVE] 保存模型到本地: {local_model_path}")
                    self.clip_model.save_pretrained(local_model_path)
                    self.clip_processor.save_pretrained(local_model_path)

                print(f"[SUCCESS] 模型加载成功: {model_name}")
                self.model_name = model_name
                break

            except Exception as e:
                print(f"[ERROR] 模型 {model_name} 加载失败: {e}")
                continue

        if self.clip_model is None:
            raise Exception("[ERROR] 所有CLIP模型都加载失败！请检查网络/镜像或使用离线缓存")

        # ====== OpenAI 客户端改为环境变量读取（无硬编码）======
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not openai_api_key:
            raise ValueError("未设置 OPENAI_API_KEY 环境变量")

        self.client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o")

        # 数据路径（可相对路径，避免绝对路径）
        self.train_csv_path = train_csv_path
        self.data_dir = data_dir

        # 内部状态
        self.df_valid = None
        self.image_index = None
        self.text_index = None
        self.image_embeddings = None
        self.text_embeddings = None

    def encode_image_with_transformers_clip(self, image_path: str) -> np.ndarray:
        """使用 transformers CLIP 编码图像"""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            return image_features[0].cpu().numpy()
        except Exception as e:
            print(f"CLIP图像编码失败: {image_path}, 错误: {e}")
            return None

    def encode_text_with_transformers_clip(self, text: str) -> np.ndarray:
        """使用 transformers CLIP 编码文本"""
        try:
            inputs = self.clip_processor(
                text=[text], return_tensors="pt",
                padding=True, truncation=True, max_length=77
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            return text_features[0].cpu().numpy()
        except Exception as e:
            print(f"CLIP文本编码失败: {text}, 错误: {e}")
            return None

    def batch_encode_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """批量编码文本"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                inputs = self.clip_processor(
                    text=batch_texts, return_tensors="pt",
                    padding=True, truncation=True, max_length=77
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    text_features = self.clip_model.get_text_features(**inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                all_embeddings.append(text_features.cpu().numpy())

                if (i // batch_size + 1) % 10 == 0:
                    print(f"    批量编码进度: {i + len(batch_texts)}/{len(texts)}")

            except Exception as e:
                print(f"批量文本编码失败: batch {i//batch_size}, 错误: {e}")
                # 失败时降级逐条
                for text in batch_texts:
                    embedding = self.encode_text_with_transformers_clip(text)
                    if embedding is not None:
                        all_embeddings.append(embedding.reshape(1, -1))

        if all_embeddings:
            return np.vstack(all_embeddings)
        else:
            return np.array([])

    def clean_medical_text(self, text: str) -> str:
        """清洗医学文本"""
        if pd.isna(text) or text == "":
            return ""
        text = re.sub(r'[^\u4e00-\u9fff\w\s，。；：（）、]', ' ', str(text))
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def classify_image_type(self, filename: str) -> str:
        """分类图像类型 - 优先匹配更长的字符串"""
        filename_lower = filename.lower()
        if 'octa' in filename_lower:
            return 'OCTA'
        elif 'oct' in filename_lower:
            return 'OCT'
        elif 'yd' in filename_lower or 'fundus' in filename_lower:
            return 'Fundus'
        else:
            return 'Unknown'

    def get_caption_for_image(self, row: pd.Series, image_filename: str) -> str:
        """策略：总体描述 + 具体图片描述"""
        descriptions = []
        general_desc = self.clean_medical_text(row.get('文字描述', ''))
        if general_desc:
            descriptions.append(f"总体: {general_desc}")

        image_type = self.classify_image_type(image_filename)

        if image_type == 'OCT':
            oct_desc = self.clean_medical_text(row.get('oct', ''))
            if oct_desc:
                descriptions.append(f"OCT: {oct_desc}")
        elif image_type == 'OCTA':
            octa_desc = self.clean_medical_text(row.get('octa', ''))
            if octa_desc:
                descriptions.append(f"OCTA: {octa_desc}")
        elif image_type == 'Fundus':
            fundus_desc = self.clean_medical_text(row.get('眼底', ''))
            if fundus_desc:
                descriptions.append(f"眼底: {fundus_desc}")

        final_caption = " | ".join(descriptions)
        if not final_caption:
            final_caption = general_desc or ""
        return final_caption

    def load_and_process_data(self):
        """加载和预处理训练集数据"""
        print("[INFO] 加载训练集CSV数据...")
        df_train = pd.read_csv(self.train_csv_path)

        print(f"[INFO] 训练集包含 {len(df_train)} 行数据")
        print(f"[INFO] 列名: {list(df_train.columns)}")

        all_data = []

        for _, row in df_train.iterrows():
            patient_id = str(row["患者序号"])

            # 患者数据目录（相对路径，避免绝对路径）
            patient_dir = os.path.join(self.data_dir, patient_id)
            if not os.path.exists(patient_dir):
                print(f"[WARNING] 患者目录不存在: {patient_dir}")
                continue

            # 支持的图片与类型映射
            image_mappings = [
                # OCT
                ('oct01.jpg', 'OCT'), ('oct01.png', 'OCT'), ('oct01.PNG', 'OCT'),
                ('oct02.jpg', 'OCT'), ('oct02.png', 'OCT'), ('oct02.PNG', 'OCT'),
                ('oct03.jpg', 'OCT'), ('oct03.png', 'OCT'), ('oct03.PNG', 'OCT'),
                # OCTA
                ('octa01.jpg', 'OCTA'), ('octa01.png', 'OCTA'), ('octa01.PNG', 'OCTA'),
                ('octa02.jpg', 'OCTA'), ('octa02.png', 'OCTA'), ('octa02.PNG', 'OCTA'),
                ('octa03.jpg', 'OCTA'), ('octa03.png', 'OCTA'), ('octa03.PNG', 'OCTA'),
                ('octa04.jpg', 'OCTA'), ('octa04.png', 'OCTA'), ('octa04.PNG', 'OCTA'),
                # Fundus
                ('yd.bmp', 'Fundus'), ('yd.BMP', 'Fundus'),
                ('fundus.jpg', 'Fundus'), ('fundus.png', 'Fundus'), ('fundus.PNG', 'Fundus')
            ]

            for img_filename, expected_type in image_mappings:
                img_path = os.path.join(patient_dir, img_filename)

                if os.path.exists(img_path):
                    caption = self.get_caption_for_image(row, img_filename)
                    if caption:
                        all_data.append({
                            "patient_id": patient_id,
                            "image_path": img_path,
                            "image_name": img_filename,
                            "image_type": expected_type,
                            "caption": caption,
                            "grade": row.get('分级', ''),
                            "text_description": self.clean_medical_text(row.get('文字描述', '')),
                            "fundus_description": self.clean_medical_text(row.get('眼底', '')),
                            "oct_description": self.clean_medical_text(row.get('oct', '')),
                            "octa_description": self.clean_medical_text(row.get('octa', ''))
                        })
                        print(f"[SUCCESS] 添加: {patient_id}/{img_filename} -> {expected_type}")
                        print(f"   [DESC] 完整描述: {caption[:100]}{'...' if len(caption) > 100 else ''}")
                    else:
                        print(f"[WARNING] 跳过: {patient_id}/{img_filename} (无有效描述)")
                else:
                    print(f"[WARNING] 图片不存在: {img_path}")

        df_all = pd.DataFrame(all_data)
        print(f"[INFO] 最终收集到 {len(df_all)} 个有效的图片-描述对")

        if len(df_all) > 0:
            print(f"\n[STATS] 图片类型分布:")
            print(df_all['image_type'].value_counts())
            print(f"\n[STATS] 唯一患者数: {df_all['patient_id'].nunique()}")
            print(f"\n[STATS] 分级分布:")
            print(df_all['grade'].value_counts())

        return df_all

    def build_embeddings_database(self, save_cache: bool = True):
        """构建嵌入数据库（支持缓存）"""
        cache_file = os.path.join(self.model_cache_dir, "train_embeddings_cache.npz")

        if os.path.exists(cache_file) and save_cache:
            try:
                print("[CACHE] 发现训练集嵌入缓存，正在加载...")
                cache_data = np.load(cache_file, allow_pickle=True)

                self.image_embeddings = cache_data['image_embeddings']
                self.text_embeddings = cache_data['text_embeddings']
                self.df_valid = pd.DataFrame(cache_data['df_valid'].item())

                print(f"[SUCCESS] 从缓存加载完成: {len(self.df_valid)} 条数据")

                embedding_dim = self.image_embeddings.shape[1]
                self.image_index = faiss.IndexFlatIP(embedding_dim)
                self.image_index.add(self.image_embeddings)

                self.text_index = faiss.IndexFlatIP(embedding_dim)
                self.text_index.add(self.text_embeddings)
                return

            except Exception as e:
                print(f"[WARNING] 缓存加载失败，重新构建: {e}")

        print("[BUILD] 构建训练集 CLIP 嵌入数据库...")
        df_all = self.load_and_process_data()

        if len(df_all) == 0:
            raise Exception("[ERROR] 没有找到有效的训练数据！")

        image_embeddings = []
        text_embeddings_list = []
        valid_indices = []

        print("[ENCODE] 编码训练集图像...")
        for i, row in df_all.iterrows():
            img_embedding = self.encode_image_with_transformers_clip(row["image_path"])
            if img_embedding is not None:
                image_embeddings.append(img_embedding)
                text_embeddings_list.append(row["caption"])
                valid_indices.append(i)

            if (i + 1) % 20 == 0:
                print(f"  已处理 {i + 1}/{len(df_all)} 张图片")

        if not image_embeddings:
            raise Exception("[ERROR] 没有成功编码任何图像！")

        print("[ENCODE] 批量编码训练集文本...")
        text_embeddings = self.batch_encode_texts(text_embeddings_list, batch_size=32)

        self.image_embeddings = np.stack(image_embeddings).astype("float32")
        self.text_embeddings = text_embeddings.astype("float32")
        self.df_valid = df_all.iloc[valid_indices].reset_index(drop=True)

        if save_cache:
            print("[SAVE] 保存训练集嵌入缓存...")
            np.savez(cache_file,
                     image_embeddings=self.image_embeddings,
                     text_embeddings=self.text_embeddings,
                     df_valid=self.df_valid.to_dict())
            print(f"[SUCCESS] 缓存已保存到: {cache_file}")

        embedding_dim = self.image_embeddings.shape[1]
        self.image_index = faiss.IndexFlatIP(embedding_dim)
        self.image_index.add(self.image_embeddings)

        self.text_index = faiss.IndexFlatIP(embedding_dim)
        self.text_index.add(self.text_embeddings)

        print(f"[SUCCESS] 训练集嵌入数据库构建完成:")
        print(f"   - 图像嵌入: {self.image_embeddings.shape}")
        print(f"   - 文本嵌入: {self.text_embeddings.shape}")
        print(f"   - 有效数据: {len(self.df_valid)} 条")
        print(f"   - 嵌入维度: {embedding_dim}")
        print(f"   - 使用模型: {self.model_name}")

        print(f"\n[STATS] 数据库内容统计:")
        print(f"   - OCT图片: {len(self.df_valid[self.df_valid['image_type']=='OCT'])} 张")
        print(f"   - OCTA图片: {len(self.df_valid[self.df_valid['image_type']=='OCTA'])} 张")
        print(f"   - 眼底图片: {len(self.df_valid[self.df_valid['image_type']=='Fundus'])} 张")
        print(f"   - 唯一患者: {self.df_valid['patient_id'].nunique()} 人")

    def multi_modal_retrieve(self, query_image_path: str, k: int = 8, alpha: float = 0.6) -> List[Dict]:
        """多模态检索：图像相似度 + 文本语义相似度（融合）"""
        query_img_embedding = self.encode_image_with_transformers_clip(query_image_path)
        if query_img_embedding is None:
            return []

        query_img_embedding = query_img_embedding.reshape(1, -1).astype("float32")

        D_img, I_img = self.image_index.search(query_img_embedding, k=k * 2)
        D_text, I_text = self.text_index.search(query_img_embedding, k=k * 2)

        candidates_scores = {}

        for idx, score in zip(I_img[0], D_img[0]):
            candidates_scores[idx] = candidates_scores.get(idx, 0) + alpha * score

        for idx, score in zip(I_text[0], D_text[0]):
            candidates_scores[idx] = candidates_scores.get(idx, 0) + (1 - alpha) * score

        sorted_candidates = sorted(candidates_scores.items(), key=lambda x: x[1], reverse=True)[:k * 2]

        query_type = self.classify_image_type(os.path.basename(query_image_path))
        similar_images = []

        # 优先返回与查询同类型的图像
        for idx, combined_score in sorted_candidates:
            row = self.df_valid.iloc[idx]
            if row["image_type"] == query_type and len(similar_images) < k:
                similar_images.append(self._create_result_dict(
                    row, combined_score, D_img, I_img, D_text, I_text, idx, True
                ))

        # 不足则补充其他类型
        for idx, combined_score in sorted_candidates:
            if len(similar_images) >= k:
                break
            row = self.df_valid.iloc[idx]
            if row["image_type"] != query_type:
                similar_images.append(self._create_result_dict(
                    row, combined_score, D_img, I_img, D_text, I_text, idx, False
                ))

        return similar_images[:k]

    def _create_result_dict(self, row, combined_score, D_img, I_img, D_text, I_text, idx, type_match):
        """创建结果字典"""
        img_sim = float(D_img[0][np.where(I_img[0] == idx)[0][0]]) if idx in I_img[0] else 0.0
        text_sim = float(D_text[0][np.where(I_text[0] == idx)[0][0]]) if idx in I_text[0] else 0.0

        return {
            "image_path": row["image_path"],
            "image_name": row["image_name"],
            "image_type": row["image_type"],
            "caption": row["caption"],
            "patient_id": row["patient_id"],
            "grade": row["grade"],
            "combined_similarity": float(combined_score),
            "image_similarity": img_sim,
            "semantic_similarity": text_sim,
            "type_match": type_match
        }

    def create_enhanced_prompt(self, similar_images: List[Dict], query_image_path: str) -> str:
        """创建增强提示词（将检索结果注入提示）"""
        query_type = self.classify_image_type(os.path.basename(query_image_path))
        same_type_count = sum(1 for img in similar_images if img['type_match'])

        prompt = f"""你是专业的眼科医学影像分析专家，专精于{query_type}图像分析。

【当前任务】为{query_type}图像生成准确的医学描述

【参考病例】基于训练集CLIP多模态检索的{len(similar_images)}个相关病例（同类型{query_type}: {same_type_count}个）：
"""
        for i, img in enumerate(similar_images, 1):
            type_indicator = "【同类型】" if img['type_match'] else "【参考】"
            prompt += (
                f"{type_indicator} 病例 {i}：\n"
                f"  类型：{img['image_type']} - {img['image_name']}\n"
                f"  患者：{img['patient_id']} (分级: {img['grade']})\n"
                f"  相似度：综合{img['combined_similarity']:.3f}\n"
                f"  完整描述：{img['caption']}\n\n"
            )

        prompt += f"""【生成要求】
1. 参考上述训练集病例的专业表达和医学术语
2. 先描述总体病情概况，再描述{query_type}图像的具体表现
3. 保持客观性，基于图像内容描述
4. 使用标准医学术语，150-250字
5. 结构：总体病情 + {query_type}具体特征

请生成专业医学影像描述："""
        return prompt

    def image_to_base64(self, image_path: str) -> str:
        """图像转 base64（用于多模态 API）"""
        with Image.open(image_path) as image:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=95)
            buffer.seek(0)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def generate_caption_with_clip_rag(self, query_image_path: str, k: int = 5, alpha: float = 0.6) -> Dict:
        """生成图像标注（RAG增强）"""
        print(f"\n[PROCESS] CLIP RAG 处理: {os.path.basename(query_image_path)}")

        if not os.path.exists(query_image_path):
            return {"error": f"图片文件不存在: {query_image_path}"}

        similar_images = self.multi_modal_retrieve(query_image_path, k=k, alpha=alpha)
        if not similar_images:
            return {"error": "检索失败"}

        query_type = self.classify_image_type(os.path.basename(query_image_path))
        same_type_count = sum(1 for img in similar_images if img['type_match'])
        print(f"[RETRIEVE] 检索完成: {len(similar_images)}个病例 ({same_type_count}个{query_type}同类型)")

        for img in similar_images:
            print(f"   - {img['patient_id']}/{img['image_name']} (相似度: {img['combined_similarity']:.3f})")

        enhanced_prompt = self.create_enhanced_prompt(similar_images, query_image_path)

        try:
            query_image_base64 = self.image_to_base64(query_image_path)
        except Exception as e:
            return {"error": f"图片编码失败: {e}"}

        # ====== 调用多模态聊天模型（模型名通过环境变量 LLM_MODEL 控制）======
        try:
            print(f"[API] 调用 {self.llm_model}...")
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是专业眼科医师，基于训练集CLIP检索的参考病例生成医学影像描述。"
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": enhanced_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{query_image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800,
                temperature=0.05,
                top_p=0.9
            )
            generated_caption = response.choices[0].message.content.strip()
            print("[SUCCESS] 描述生成成功")

            return {
                "query_image": query_image_path,
                "query_type": query_type,
                "generated_caption": generated_caption,
                "similar_images": similar_images,
                "same_type_matches": same_type_count,
                "model_info": f"transformers {self.model_name}",
                "success": True
            }

        except Exception as e:
            return {"error": f"多模态模型调用失败: {e}"}

    def generate_patient_summary(self, results: List[Dict], patient_id: str, directory_path: str) -> Dict:
        """生成患者级别的综合总结"""
        print(f"\n[SUMMARY] 正在生成患者 {patient_id} 的综合医学报告...")

        successful_results = [r for r in results if "error" not in r]
        if not successful_results:
            return {"error": "没有成功的分析结果可用于生成患者总结"}

        type_groups = {}
        for result in successful_results:
            img_type = result['query_type']
            type_groups.setdefault(img_type, []).append(result)

        summary_prompt = (
            f"你是资深眼科专家，现在需要基于多模态医学图像分析结果，为患者 {patient_id} 生成一份综合性的眼科诊断报告。\n\n"
            f"【患者信息】\n"
            f"患者ID: {patient_id}\n"
            f"检查类型: 多模态眼科影像检查（包含 {', '.join(type_groups.keys())}）\n"
            f"图像总数: {len(successful_results)} 张\n\n"
            f"【各类型图像分析结果】\n"
        )

        for img_type, type_results in type_groups.items():
            summary_prompt += f"\n=== {img_type} 类型图像 ({len(type_results)} 张) ===\n"
            for i, result in enumerate(type_results, 1):
                img_name = os.path.basename(result['query_image'])
                summary_prompt += f"\n{i}. {img_name}:\n   {result['generated_caption']}\n"

        summary_prompt += (
            "\n【报告要求】\n"
            "请基于上述多模态医学图像分析结果，生成一份专业的眼科诊断报告，包括以下部分：\n\n"
            "1. 患者概况：简要总结患者的总体眼部状况\n"
            "2. 主要发现：各类型检查的关键发现，按严重程度排序\n"
            "3. 病理分析：结合多模态检查结果的综合分析\n"
            "4. 诊断建议：可能的诊断方向或需要进一步检查的建议\n"
            "5. 随访建议：后续治疗和随访的专业建议\n\n"
            "要求：使用专业医学术语，保持客观和准确；突出多模态检查的互补优势；字数控制在400-600字；结构清晰，便于临床使用。\n"
            "请生成专业的眼科诊断报告："
        )

        try:
            print(f"[API] 调用 {self.llm_model} 生成患者综合报告...")
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是资深眼科专家，擅长综合多模态医学影像结果生成专业诊断报告。"
                    },
                    {
                        "role": "user",
                        "content": summary_prompt
                    }
                ],
                max_tokens=1200,
                temperature=0.1,
                top_p=0.9
            )
            patient_summary = response.choices[0].message.content.strip()
            print("[SUCCESS] 患者综合报告生成成功")

            return {
                "patient_id": patient_id,
                "patient_summary": patient_summary,
                "image_count": len(successful_results),
                "image_types": list(type_groups.keys()),
                "type_distribution": {k: len(v) for k, v in type_groups.items()},
                "generation_success": True
            }

        except Exception as e:
            print(f"[ERROR] 患者总结生成失败: {e}")
            return {"error": f"患者总结生成失败: {e}"}

    def interactive_mode(self):
        """交互式模式"""
        print("\n" + "=" * 60)
        print("医学图像RAG系统已准备就绪！")
        print("=" * 60)
        print("使用说明:")
        print("   - 输入单个图像路径进行分析")
        print("   - 输入患者目录路径批量分析多个图像")
        print("   - 支持的图像类型: OCT, OCTA, Fundus")
        print("   - 输入 'quit' 或 'exit' 退出")
        print("   - 输入 'help' 查看帮助")
        print("   - 输入 'stats' 查看数据库统计")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n请输入图像路径或患者目录: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("再见！")
                    break
                elif user_input.lower() in ['help', 'h']:
                    self._show_help()
                    continue
                elif user_input.lower() in ['stats', 'stat']:
                    self._show_stats()
                    continue
                elif not user_input:
                    print("[WARNING] 请输入有效的路径")
                    continue

                if not os.path.exists(user_input):
                    print(f"[ERROR] 路径不存在: {user_input}")
                    continue

                if os.path.isdir(user_input):
                    print(f"\n[INFO] 检测到目录，开始批量分析: {user_input}")
                    self._process_directory(user_input)
                elif os.path.isfile(user_input):
                    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP']
                    if not any(user_input.lower().endswith(ext.lower()) for ext in valid_extensions):
                        print(f"[WARNING] 不支持的文件格式，支持的格式: {', '.join(valid_extensions)}")
                        continue

                    print(f"\n[PROCESS] 正在分析图像: {user_input}")
                    result = self.generate_caption_with_clip_rag(user_input, k=5)
                    self._display_single_result(result, user_input)
                else:
                    print(f"[WARNING] 无效的路径类型: {user_input}")

            except KeyboardInterrupt:
                print("\n\n用户中断，再见！")
                break
            except Exception as e:
                print(f"[ERROR] 发生错误: {e}")

    def _process_directory(self, directory_path: str):
        """处理目录中的所有图像文件"""
        try:
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP']

            image_files = []
            for file in os.listdir(directory_path):
                if any(file.lower().endswith(ext.lower()) for ext in valid_extensions):
                    image_files.append(os.path.join(directory_path, file))

            if not image_files:
                print(f"[WARNING] 目录中没有找到支持的图像文件")
                return

            image_files.sort()

            print(f"[INFO] 发现 {len(image_files)} 个图像文件:")
            for i, img_path in enumerate(image_files, 1):
                filename = os.path.basename(img_path)
                img_type = self.classify_image_type(filename)
                print(f"   {i}. {filename} ({img_type})")

            choice = input(f"\n处理方式选择:\n   1. 逐个分析并显示结果\n   2. 批量分析并生成报告\n   请选择 (1/2): ").strip()

            if choice == '1':
                self._process_images_individually(image_files, directory_path)
            elif choice == '2':
                self._process_images_batch(image_files, directory_path)
            else:
                print("[WARNING] 无效选择，默认逐个分析")
                self._process_images_individually(image_files, directory_path)

        except Exception as e:
            print(f"[ERROR] 处理目录失败: {e}")

    def _process_images_individually(self, image_files: List[str], directory_path: str):
        """逐个处理图像文件"""
        results = []

        for i, image_path in enumerate(image_files, 1):
            filename = os.path.basename(image_path)
            print(f"\n{'=' * 60}")
            print(f"[PROCESS] 处理第 {i}/{len(image_files)} 个图像: {filename}")
            print(f"{'=' * 60}")

            result = self.generate_caption_with_clip_rag(image_path, k=5)
            results.append(result)

            self._display_single_result(result, image_path, show_save_option=False)

            if i < len(image_files):
                continue_choice = input(f"\n继续处理下一个图像? (y/n/q): ").strip().lower()
                if continue_choice in ['n', 'no']:
                    print("[PAUSE] 处理暂停")
                    break
                elif continue_choice in ['q', 'quit']:
                    print("[STOP] 处理中止")
                    return

        self._offer_batch_save(results, directory_path)

    def _process_images_batch(self, image_files: List[str], directory_path: str):
        """批量处理图像文件并生成报告"""
        print(f"\n[BATCH] 开始批量处理 {len(image_files)} 个图像...")

        results = []
        successful_count = 0
        failed_count = 0

        for i, image_path in enumerate(image_files, 1):
            filename = os.path.basename(image_path)
            print(f"[PROCESS] 处理进度: {i}/{len(image_files)} - {filename}")

            result = self.generate_caption_with_clip_rag(image_path, k=5)
            results.append(result)

            if "error" not in result:
                successful_count += 1
                print(f"   [SUCCESS] 成功")
            else:
                failed_count += 1
                print(f"   [ERROR] 失败: {result['error']}")

        patient_id = os.path.basename(directory_path)
        patient_summary_result = self.generate_patient_summary(results, patient_id, directory_path)

        self._generate_batch_report(results, directory_path, successful_count, failed_count, patient_summary_result)

    def _display_single_result(self, result: Dict, image_path: str, show_save_option: bool = True):
        """显示单个分析结果"""
        if "error" not in result:
            print("\n" + "=" * 50)
            print(f"图像: {os.path.basename(image_path)}")
            print(f"类型: {result['query_type']}")
            print(f"生成描述:")
            print("-" * 50)
            print(result['generated_caption'])
            print("-" * 50)
            print(f"检索信息: {result['same_type_matches']}/{len(result['similar_images'])} 同类型参考病例")
            print(f"使用模型: {result['model_info']}")
            print("=" * 50)

            if show_save_option:
                save_choice = input("\n是否保存结果到文件? (y/n): ").strip().lower()
                if save_choice in ['y', 'yes']:
                    self._save_result(result, image_path)
        else:
            print(f"[ERROR] 分析失败: {result['error']}")

    def _generate_batch_report(self, results: List[Dict], directory_path: str,
                               successful_count: int, failed_count: int, patient_summary_result: Dict):
        """生成批量处理报告"""
        patient_id = os.path.basename(directory_path)

        print(f"\n{'=' * 60}")
        print(f"批量处理报告 - 患者: {patient_id}")
        print(f"{'=' * 60}")
        print(f"处理统计:")
        print(f"   [SUCCESS] 成功: {successful_count} 个")
        print(f"   [ERROR] 失败: {failed_count} 个")
        print(f"   [TOTAL] 总计: {len(results)} 个")

        # 患者综合总结
        if "error" not in patient_summary_result:
            print(f"\n{'=' * 60}")
            print(f"患者 {patient_id} 综合医学报告")
            print(f"{'=' * 60}")
            print(patient_summary_result['patient_summary'])
            print(f"{'=' * 60}")
        else:
            print(f"\n[WARNING] 患者综合报告生成失败: {patient_summary_result['error']}")

        successful_results = [r for r in results if "error" not in r]
        if successful_results:
            type_groups = {}
            for result in successful_results:
                img_type = result['query_type']
                type_groups.setdefault(img_type, []).append(result)

            print(f"\n分析结果详情:")
            for img_type, type_results in type_groups.items():
                print(f"\n{img_type} 类型图像 ({len(type_results)} 张):")
                for i, result in enumerate(type_results, 1):
                    img_name = os.path.basename(result['query_image'])
                    preview = result['generated_caption'][:100]
                    ell = '...' if len(result['generated_caption']) > 100 else ''
                    print(f"   {i}. {img_name}")
                    print(f"      {preview}{ell}")

        failed_results = [r for r in results if "error" in r]
        if failed_results:
            print(f"\n[ERROR] 处理失败的图像:")
            for i, result in enumerate(failed_results, 1):
                print(f"   {i}. {result.get('query_image', 'Unknown')}: {result['error']}")

        print("=" * 60)

        self._offer_batch_save(results, directory_path, patient_summary_result)

    def _offer_batch_save(self, results: List[Dict], directory_path: str, patient_summary_result: Dict = None):
        """提供批量保存选项"""
        save_choice = input(
            f"\n保存选项:\n   1. 保存完整报告（包含患者总结）\n   2. 仅保存成功结果\n   3. 不保存\n   请选择 (1/2/3): "
        ).strip()

        if save_choice == '1':
            self._save_batch_results(results, directory_path, include_failed=True,
                                     patient_summary=patient_summary_result)
        elif save_choice == '2':
            successful_results = [r for r in results if "error" not in r]
            if successful_results:
                self._save_batch_results(successful_results, directory_path, include_failed=False,
                                         patient_summary=patient_summary_result)
            else:
                print("[WARNING] 没有成功的结果可保存")
        elif save_choice == '3':
            print("结果未保存")
        else:
            print("[WARNING] 无效选择，结果未保存")

    def _save_batch_results(self, results: List[Dict], directory_path: str,
                            include_failed: bool = True, patient_summary: Dict = None):
        """保存批量处理结果"""
        try:
            patient_id = os.path.basename(directory_path)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f"batch_analysis_{patient_id}_{timestamp}.json"

            save_data = {
                'patient_id': patient_id,
                'directory_path': directory_path,
                'timestamp': timestamp,
                'total_images': len(results),
                'successful_count': len([r for r in results if "error" not in r]),
                'failed_count': len([r for r in results if "error" in r]),
                'include_failed_results': include_failed,
                'results': results
            }

            if patient_summary:
                save_data['patient_summary'] = patient_summary

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            print(f"[SUCCESS] 批量结果已保存到: {filename}")
            if patient_summary and "error" not in patient_summary:
                print(f"   包含患者综合医学报告")

        except Exception as e:
            print(f"[ERROR] 保存失败: {e}")

    def _show_help(self):
        """显示帮助信息"""
        print("\n" + "=" * 60)
        print("帮助信息")
        print("=" * 60)
        print("支持的输入:")
        print("  单个图像路径  - 分析指定路径的医学图像")
        print("  患者目录路径  - 批量分析目录中的所有图像")
        print("  stats        - 显示数据库统计信息")
        print("  help         - 显示此帮助信息")
        print("  quit/exit    - 退出程序")
        print("\n支持的图像格式:")
        print("  JPG, JPEG, PNG, BMP (大小写不敏感)")
        print("\n支持的图像类型:")
        print("  OCT    - 光学相干断层扫描")
        print("  OCTA   - 光学相干断层扫描血管造影")
        print("  Fundus - 眼底照片")
        print("\n批量处理模式:")
        print("  逐个分析 - 逐个显示结果，可中途暂停")
        print("  批量分析 - 快速处理并生成汇总报告 + 患者综合总结")
        print("\n新功能:")
        print("  患者综合报告 - 基于多模态检查结果生成专业医学报告")
        print("  多层次分析 - 单图分析 + 患者级别综合分析")
        print("\n输入示例:")
        print("  单个图像: ./data/1240_os/oct01.jpg")
        print("  患者目录: ./data/1240_os")
        print("  绝对路径: /path/to/patient/directory（不推荐在仓库中硬编码）")
        print("=" * 60)

    def _show_stats(self):
        """显示数据库统计信息"""
        print("\n" + "=" * 60)
        print("数据库统计信息")
        print("=" * 60)
        if self.df_valid is None or self.image_embeddings is None:
            print("数据库尚未构建。请先运行 build_embeddings_database()")
            print("=" * 60)
            return

        print(f"总数据量: {len(self.df_valid)} 条")
        print(f"患者数量: {self.df_valid['patient_id'].nunique()} 人")
        print(f"嵌入维度: {self.image_embeddings.shape[1]}")
        print(f"使用模型: {self.model_name}")
        print("\n图像类型分布:")
        type_counts = self.df_valid['image_type'].value_counts()
        for img_type, count in type_counts.items():
            percentage = (count / len(self.df_valid)) * 100
            print(f"  {img_type}: {count} 张 ({percentage:.1f}%)")

        print("\n分级分布:")
        grade_counts = self.df_valid['grade'].value_counts()
        for grade, count in grade_counts.items():
            percentage = (count / len(self.df_valid)) * 100
            print(f"  {grade}: {count} 个 ({percentage:.1f}%)")
        print("=" * 60)

    def _save_result(self, result, image_path):
        """保存单个分析结果"""
        try:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f"analysis_result_{timestamp}.json"

            result_to_save = result.copy()
            result_to_save['timestamp'] = timestamp
            result_to_save['original_input_path'] = image_path

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result_to_save, f, ensure_ascii=False, indent=2)

            print(f"[SUCCESS] 结果已保存到: {filename}")

        except Exception as e:
            print(f"[ERROR] 保存失败: {e}")


def main():
    """主函数"""
    # 使用相对路径，避免绝对路径；可通过环境变量覆盖
    train_csv_path = os.getenv("TRAIN_CSV_PATH", "./train_data_by_patient.csv")
    data_dir = os.getenv("DATA_DIR", "./data")
    model_cache_dir = os.getenv("MODEL_CACHE_DIR", "./model_cache")

    print("初始化训练集 CLIP 医学图像RAG系统...")

    try:
        clip_rag = TransformersCLIPMedicalRAG(
            train_csv_path=train_csv_path,
            data_dir=data_dir,
            model_cache_dir=model_cache_dir
        )

        clip_rag.build_embeddings_database(save_cache=True)
        clip_rag.interactive_mode()

    except Exception as e:
        print(f"[ERROR] 系统初始化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
