#!/usr/bin/env python3
"""
医学图像归档解压工具
专门用于解压 /root/autodl-tmp/medical_images/归档.zip
"""

import os
import sys
import zipfile
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import shutil

def print_with_flush(*args, **kwargs):
    """确保立即输出的print函数"""
    print(*args, **kwargs)
    sys.stdout.flush()

class MedicalArchiveExtractor:
    """医学图像归档解压器"""
    
    def __init__(self, zip_path: str, extract_base_dir: str = None):
        """
        初始化解压器
        
        Args:
            zip_path: ZIP文件路径
            extract_base_dir: 解压基础目录，默认为ZIP文件所在目录
        """
        self.zip_path = Path(zip_path)
        
        if extract_base_dir:
            self.extract_base_dir = Path(extract_base_dir)
        else:
            self.extract_base_dir = self.zip_path.parent
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'images_extracted': 0,
            'annotations_extracted': 0,
            'other_files': 0,
            'errors': 0,
            'total_size': 0,
            'extraction_time': None
        }
        
        # 支持的文件类型
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.dcm', '.dicom', '.gif'}
        self.annotation_extensions = {'.txt', '.json', '.xml', '.csv', '.yaml', '.yml'}
        
        print_with_flush(f"📦 ZIP文件: {self.zip_path}")
        print_with_flush(f"📁 解压目录: {self.extract_base_dir}")
    
    def check_zip_file(self) -> bool:
        """检查ZIP文件是否存在且有效"""
        print_with_flush("\n🔍 检查ZIP文件...")
        
        if not self.zip_path.exists():
            print_with_flush(f"❌ ZIP文件不存在: {self.zip_path}")
            return False
        
        if not self.zip_path.is_file():
            print_with_flush(f"❌ 路径不是文件: {self.zip_path}")
            return False
        
        # 检查文件大小
        file_size = self.zip_path.stat().st_size
        print_with_flush(f"📊 文件大小: {self._format_size(file_size)}")
        
        # 检查是否为有效的ZIP文件
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zf:
                # 测试ZIP文件完整性
                bad_file = zf.testzip()
                if bad_file:
                    print_with_flush(f"❌ ZIP文件损坏，损坏文件: {bad_file}")
                    return False
                
                # 统计文件数量
                file_list = zf.namelist()
                self.stats['total_files'] = len(file_list)
                print_with_flush(f"📊 包含文件: {self.stats['total_files']} 个")
                
                # 预览前几个文件
                print_with_flush("📋 文件预览 (前5个):")
                for i, filename in enumerate(file_list[:5]):
                    print_with_flush(f"  {i+1}. {filename}")
                
                if len(file_list) > 5:
                    print_with_flush(f"  ... 还有 {len(file_list) - 5} 个文件")
                
        except zipfile.BadZipFile:
            print_with_flush("❌ 不是有效的ZIP文件")
            return False
        except Exception as e:
            print_with_flush(f"❌ 检查ZIP文件时出错: {e}")
            return False
        
        print_with_flush("✅ ZIP文件检查通过")
        return True
    
    def create_directory_structure(self):
        """创建解压目录结构"""
        print_with_flush("\n📁 创建目录结构...")
        
        directories = [
            self.extract_base_dir / "images",
            self.extract_base_dir / "annotations", 
            self.extract_base_dir / "metadata",
            self.extract_base_dir / "extracted_files"  # 原始解压文件
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print_with_flush(f"  ✅ {directory}")
        
        print_with_flush("✅ 目录结构创建完成")
    
    def extract_and_organize_files(self) -> bool:
        """解压并组织文件"""
        print_with_flush("\n🔓 开始解压文件...")
        
        start_time = datetime.now()
        
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zf:
                file_list = zf.namelist()
                
                # 首先解压所有文件到临时目录
                temp_extract_dir = self.extract_base_dir / "temp_extracted"
                temp_extract_dir.mkdir(exist_ok=True)
                
                print_with_flush(f"📦 解压 {len(file_list)} 个文件...")
                
                for i, filename in enumerate(file_list, 1):
                    try:
                        # 跳过目录条目
                        if filename.endswith('/'):
                            continue
                        
                        # 解压文件
                        zf.extract(filename, temp_extract_dir)
                        
                        # 显示进度
                        if i % 100 == 0 or i == len(file_list):
                            print_with_flush(f"  📥 进度: {i}/{len(file_list)} ({i/len(file_list)*100:.1f}%)")
                    
                    except Exception as e:
                        print_with_flush(f"  ❌ 解压失败: {filename} - {e}")
                        self.stats['errors'] += 1
                
                print_with_flush("✅ 文件解压完成")
                
                # 组织文件到相应目录
                self._organize_extracted_files(temp_extract_dir)
                
                # 清理临时目录
                print_with_flush("🧹 清理临时文件...")
                shutil.rmtree(temp_extract_dir, ignore_errors=True)
                
        except Exception as e:
            print_with_flush(f"❌ 解压过程中出错: {e}")
            return False
        
        end_time = datetime.now()
        self.stats['extraction_time'] = (end_time - start_time).total_seconds()
        
        return True
    
    def _organize_extracted_files(self, temp_dir: Path):
        """组织解压后的文件"""
        print_with_flush("\n📂 组织文件到相应目录...")
        
        # 遍历临时目录中的所有文件
        for file_path in temp_dir.rglob('*'):
            if file_path.is_file():
                try:
                    self._categorize_and_move_file(file_path, temp_dir)
                except Exception as e:
                    print_with_flush(f"  ❌ 移动文件失败: {file_path.name} - {e}")
                    self.stats['errors'] += 1
    
    def _categorize_and_move_file(self, file_path: Path, temp_dir: Path):
        """分类并移动文件"""
        extension = file_path.suffix.lower()
        relative_path = file_path.relative_to(temp_dir)
        
        # 确定目标目录
        if extension in self.image_extensions:
            target_dir = self.extract_base_dir / "images"
            self.stats['images_extracted'] += 1
        elif extension in self.annotation_extensions:
            target_dir = self.extract_base_dir / "annotations"
            self.stats['annotations_extracted'] += 1
        else:
            target_dir = self.extract_base_dir / "metadata"
            self.stats['other_files'] += 1
        
        # 构建目标路径，保持原有的目录结构
        target_path = target_dir / relative_path
        
        # 确保目标目录存在
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 处理文件名冲突
        if target_path.exists():
            counter = 1
            stem = target_path.stem
            suffix = target_path.suffix
            while target_path.exists():
                new_name = f"{stem}_{counter}{suffix}"
                target_path = target_path.parent / new_name
                counter += 1
        
        # 移动文件
        shutil.move(str(file_path), str(target_path))
        
        # 更新文件大小统计
        self.stats['total_size'] += target_path.stat().st_size
    
    def create_extraction_report(self):
        """创建解压报告"""
        print_with_flush("\n📋 生成解压报告...")
        
        report = {
            'extraction_info': {
                'timestamp': datetime.now().isoformat(),
                'zip_file': str(self.zip_path),
                'extract_directory': str(self.extract_base_dir),
                'extraction_time_seconds': self.stats['extraction_time']
            },
            'statistics': self.stats,
            'directory_structure': {
                'images': str(self.extract_base_dir / "images"),
                'annotations': str(self.extract_base_dir / "annotations"),
                'metadata': str(self.extract_base_dir / "metadata")
            }
        }
        
        # 添加详细的文件统计
        report['file_counts'] = {
            'images': self._count_files_in_dir(self.extract_base_dir / "images"),
            'annotations': self._count_files_in_dir(self.extract_base_dir / "annotations"),
            'metadata': self._count_files_in_dir(self.extract_base_dir / "metadata")
        }
        
        # 保存报告
        report_path = self.extract_base_dir / "extraction_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print_with_flush(f"✅ 解压报告已保存: {report_path}")
        return report
    
    def _count_files_in_dir(self, directory: Path) -> Dict[str, int]:
        """统计目录中的文件"""
        if not directory.exists():
            return {'total': 0, 'by_extension': {}}
        
        file_counts = {'total': 0, 'by_extension': {}}
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                file_counts['total'] += 1
                ext = file_path.suffix.lower()
                file_counts['by_extension'][ext] = file_counts['by_extension'].get(ext, 0) + 1
        
        return file_counts
    
    def print_summary(self):
        """打印解压摘要"""
        print_with_flush("\n" + "="*60)
        print_with_flush("📊 解压完成摘要")
        print_with_flush("="*60)
        print_with_flush(f"📦 原始文件: {self.zip_path.name}")
        print_with_flush(f"📁 解压目录: {self.extract_base_dir}")
        print_with_flush(f"⏱️  解压用时: {self.stats['extraction_time']:.1f} 秒")
        print_with_flush(f"📊 总文件数: {self.stats['total_files']} 个")
        print_with_flush(f"🖼️  图像文件: {self.stats['images_extracted']} 个")
        print_with_flush(f"📝 标注文件: {self.stats['annotations_extracted']} 个")
        print_with_flush(f"📄 其他文件: {self.stats['other_files']} 个")
        print_with_flush(f"❌ 错误数量: {self.stats['errors']} 个")
        print_with_flush(f"💾 总大小: {self._format_size(self.stats['total_size'])}")
        print_with_flush("="*60)
        
        # 显示目录结构
        print_with_flush("\n📂 解压后的目录结构:")
        print_with_flush(f"{self.extract_base_dir}/")
        print_with_flush(f"├── images/          # {self.stats['images_extracted']} 个图像文件")
        print_with_flush(f"├── annotations/     # {self.stats['annotations_extracted']} 个标注文件")
        print_with_flush(f"├── metadata/        # {self.stats['other_files']} 个其他文件")
        print_with_flush(f"└── extraction_report.json  # 解压报告")
    
    def _format_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    def extract_archive(self) -> bool:
        """执行完整的解压流程"""
        print_with_flush("🚀 开始医学图像归档解压")
        print_with_flush("="*60)
        
        try:
            # 1. 检查ZIP文件
            if not self.check_zip_file():
                return False
            
            # 2. 创建目录结构
            self.create_directory_structure()
            
            # 3. 解压并组织文件
            if not self.extract_and_organize_files():
                return False
            
            # 4. 生成报告
            self.create_extraction_report()
            
            # 5. 打印摘要
            self.print_summary()
            
            print_with_flush("\n🎉 解压完成！")
            return True
            
        except Exception as e:
            print_with_flush(f"\n❌ 解压过程中发生严重错误: {e}")
            import traceback
            print_with_flush(f"🔍 错误详情: {traceback.format_exc()}")
            return False


def main():
    """主函数"""
    print_with_flush("📦 医学图像归档解压工具")
    print_with_flush("="*50)
    
    # 默认路径配置
    zip_file_path = "/root/autodl-tmp/medical_images/归档.zip"
    extract_directory = "/root/autodl-tmp/medical_images"
    
    print_with_flush("🔧 解压配置:")
    print_with_flush(f"  📦 ZIP文件: {zip_file_path}")
    print_with_flush(f"  📁 解压目录: {extract_directory}")
    
    # 检查输入参数
    if len(sys.argv) > 1:
        zip_file_path = sys.argv[1]
        print_with_flush(f"  📝 使用命令行指定的ZIP文件: {zip_file_path}")
    
    if len(sys.argv) > 2:
        extract_directory = sys.argv[2]
        print_with_flush(f"  📝 使用命令行指定的解压目录: {extract_directory}")
    
    # 创建解压器并执行解压
    extractor = MedicalArchiveExtractor(zip_file_path, extract_directory)
    success = extractor.extract_archive()
    
    if success:
        print_with_flush(f"\n✅ 医学图像归档已成功解压到: {extract_directory}")
        print_with_flush("🎯 现在可以在RAG系统中使用这些医学图像了！")
        
        # 显示下一步建议
        print_with_flush("\n💡 建议下一步操作:")
        print_with_flush("1. 检查解压后的文件结构")
        print_with_flush("2. 运行医学RAG系统进行图像标注")
        print_with_flush("3. 查看 extraction_report.json 了解详细信息")
        
    else:
        print_with_flush("\n❌ 解压失败，请检查错误信息并重试")
        sys.exit(1)


def quick_extract():
    """快速解压模式（最小化输出）"""
    zip_path = "/root/autodl-tmp/medical_images/归档.zip"
    extract_dir = "/root/autodl-tmp/medical_images"
    
    if not os.path.exists(zip_path):
        print_with_flush(f"❌ 文件不存在: {zip_path}")
        return
    
    print_with_flush("⚡ 快速解压模式")
    extractor = MedicalArchiveExtractor(zip_path, extract_dir)
    
    if extractor.extract_archive():
        print_with_flush("✅ 快速解压完成")
    else:
        print_with_flush("❌ 快速解压失败")


if __name__ == "__main__":
    # 检查是否指定了快速模式
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_extract()
    else:
        main()