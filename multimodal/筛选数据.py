import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

def filter_patient_data():
    """
    读取CSV文件，根据./data目录下的患者序号进行筛选
    """
    try:
        # 1. 读取CSV文件
        csv_file = 'patient_diagnosis_result_and_caption copy.csv'
        df = pd.read_csv(csv_file, encoding='utf-8')
        
        print(f"原始CSV数据行数: {len(df)}")
        print(f"列名: {list(df.columns)}")
        
        # 2. 读取./data目录下的文件名（患者序号）
        data_dir = './data'
        
        if not os.path.exists(data_dir):
            print(f"警告: {data_dir} 目录不存在")
            return None
            
        # 获取./data目录下所有文件的文件名（不包含扩展名）
        data_files = os.listdir(data_dir)
        data_patient_ids = []
        
        for file in data_files:
            # 去掉文件扩展名，获取患者序号
            patient_id = Path(file).stem
            data_patient_ids.append(patient_id)
        
        print(f"./data目录下患者序号数量: {len(data_patient_ids)}")
        print(f"前10个患者序号: {data_patient_ids[:10]}")
        
        # 3. 筛选CSV数据，只保留在./data目录中存在的患者
        filtered_df = df[df['患者序号'].isin(data_patient_ids)]
        
        print(f"筛选后数据行数: {len(filtered_df)}")
        
        # 4. 保存新的CSV文件
        output_file = 'filtered_patient_data.csv'
        filtered_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"筛选完成，新文件已保存为: {output_file}")
        
        # 显示筛选结果统计
        print(f"\n筛选统计:")
        print(f"- 原始数据: {len(df)} 行")
        print(f"- ./data目录文件: {len(data_patient_ids)} 个")
        print(f"- 筛选后数据: {len(filtered_df)} 行")
        print(f"- 匹配率: {len(filtered_df)/len(df)*100:.1f}%")
        
        return filtered_df
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        return None
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        return None

def split_patient_data(df=None, csv_file='filtered_patient_data.csv'):
    """
    将数据拆分为训练集(160行)和测试集(40行)
    """
    try:
        # 如果没有传入数据，则从文件读取
        if df is None:
            df = pd.read_csv(csv_file, encoding='utf-8')
        
        print(f"\n=== 数据集拆分 ===")
        print(f"总数据行数: {len(df)}")
        
        # 检查数据量是否足够
        if len(df) < 200:
            print(f"警告: 数据量不足200行，当前只有{len(df)}行")
            # 按比例调整
            test_size = min(40, int(len(df) * 0.2))
            train_size = len(df) - test_size
            print(f"调整为: 训练集{train_size}行, 测试集{test_size}行")
        else:
            train_size = 160
            test_size = 40
        
        # 检查分级分布
        grade_counts = df['分级'].value_counts()
        print(f"\n分级分布:")
        print(grade_counts)
        
        # 按分级进行分层抽样，保证训练集和测试集的分级比例一致
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size,        # 测试集大小
            train_size=train_size,      # 训练集大小
            stratify=df['分级'],         # 按分级分层抽样
            random_state=42             # 固定随机种子，保证结果可复现
        )
        
        print(f"\n拆分结果:")
        print(f"训练集行数: {len(train_df)}")
        print(f"测试集行数: {len(test_df)}")
        
        # 保存训练集和测试集
        train_df.to_csv('train_data.csv', index=False, encoding='utf-8')
        test_df.to_csv('test_data.csv', index=False, encoding='utf-8')
        
        print(f"\n文件已保存:")
        print(f"- 训练集: train_data.csv ({len(train_df)} 行)")
        print(f"- 测试集: test_data.csv ({len(test_df)} 行)")
        
        # 验证分层抽样效果
        print(f"\n训练集分级分布:")
        train_grade_dist = train_df['分级'].value_counts().sort_index()
        print(train_grade_dist)
        
        print(f"\n测试集分级分布:")
        test_grade_dist = test_df['分级'].value_counts().sort_index()
        print(test_grade_dist)
        
        # 计算分级比例
        print(f"\n分级比例对比:")
        for grade in df['分级'].unique():
            total_ratio = grade_counts[grade] / len(df) * 100
            train_ratio = train_grade_dist.get(grade, 0) / len(train_df) * 100
            test_ratio = test_grade_dist.get(grade, 0) / len(test_df) * 100
            print(f"{grade}: 总体{total_ratio:.1f}% | 训练集{train_ratio:.1f}% | 测试集{test_ratio:.1f}%")
        
        return train_df, test_df
        
    except Exception as e:
        print(f"拆分过程中出现错误: {e}")
        return None, None

def split_by_patient(df=None, csv_file='filtered_patient_data.csv'):
    """按患者编号拆分 - 确保同一患者的左右眼数据在同一集合中"""
    try:
        if df is None:
            df = pd.read_csv(csv_file, encoding='utf-8')
        
        print(f"\n=== 按患者拆分数据集 ===")
        
        # 提取患者编号（去掉_od/_os后缀）
        df['患者编号'] = df['患者序号'].str.replace('_od|_os', '', regex=True)
        
        # 获取唯一患者列表
        unique_patients = df['患者编号'].unique()
        np.random.seed(42)
        np.random.shuffle(unique_patients)
        
        print(f"唯一患者数量: {len(unique_patients)}")
        
        # 按患者分配到训练集和测试集
        train_patients = []
        test_patients = []
        train_count = 0
        test_count = 0
        
        target_train = 160 if len(df) >= 200 else int(len(df) * 0.8)
        target_test = 40 if len(df) >= 200 else len(df) - target_train
        
        for patient in unique_patients:
            patient_data = df[df['患者编号'] == patient]
            
            if train_count + len(patient_data) <= target_train:
                train_patients.append(patient)
                train_count += len(patient_data)
            elif test_count + len(patient_data) <= target_test:
                test_patients.append(patient)
                test_count += len(patient_data)
        
        # 生成训练集和测试集
        train_df = df[df['患者编号'].isin(train_patients)].drop('患者编号', axis=1)
        test_df = df[df['患者编号'].isin(test_patients)].drop('患者编号', axis=1)
        
        # 保存
        train_df.to_csv('train_data_by_patient.csv', index=False, encoding='utf-8')
        test_df.to_csv('test_data_by_patient.csv', index=False, encoding='utf-8')
        
        print(f"按患者拆分完成:")
        print(f"- 训练集: {len(train_df)}行 ({len(train_patients)}名患者)")
        print(f"- 测试集: {len(test_df)}行 ({len(test_patients)}名患者)")
        
        return train_df, test_df
        
    except Exception as e:
        print(f"按患者拆分过程中出现错误: {e}")
        return None, None

def show_data_summary(df, title="数据摘要"):
    """显示数据摘要信息"""
    if df is not None:
        print(f"\n=== {title} ===")
        print(f"数据行数: {len(df)}")
        print(f"数据列数: {len(df.columns)}")
        print(f"\n前5行数据:")
        print(df.head())
        print(f"\n各分级统计:")
        print(df['分级'].value_counts())

def main():
    """主函数 - 完整的数据处理流程"""
    print("=== 患者数据处理完整流程 ===\n")
    
    # 步骤1: 筛选数据
    print("步骤1: 根据./data目录筛选患者数据")
    filtered_data = filter_patient_data()
    
    if filtered_data is None:
        print("数据筛选失败，程序退出")
        return
    
    # 显示筛选后的数据摘要
    show_data_summary(filtered_data, "筛选后数据摘要")
    
    # 步骤2: 选择拆分方式
    print(f"\n步骤2: 选择数据集拆分方式")
    print("1. 分层抽样拆分（推荐）- 保持分级比例")
    print("2. 按患者拆分 - 同一患者左右眼数据不分离")
    
    choice = input("请选择拆分方式 (1/2) [默认为1]: ").strip()
    
    if choice == "2":
        # 按患者拆分
        train_data, test_data = split_by_patient(filtered_data)
    else:
        # 分层抽样拆分（默认）
        train_data, test_data = split_patient_data(filtered_data)
    
    if train_data is not None and test_data is not None:
        print(f"\n=== 处理完成 ===")
        print("生成的文件:")
        print("- filtered_patient_data.csv (筛选后的完整数据)")
        print("- train_data.csv (训练集)")
        print("- test_data.csv (测试集)")
        
        # 显示最终统计
        show_data_summary(train_data, "训练集摘要")
        show_data_summary(test_data, "测试集摘要")
    else:
        print("数据拆分失败")

# 简化版本 - 一行代码实现筛选
def simple_filter():
    """简化版本的筛选函数"""
    df = pd.read_csv('patient_diagnosis_result_and_caption copy.csv')
    data_ids = [Path(f).stem for f in os.listdir('./data')]
    
    filtered = df[df['患者序号'].isin(data_ids)]
    filtered.to_csv('filtered_patient_data.csv', index=False)
    
    print(f"筛选完成: {len(filtered)}/{len(df)} 行保留")
    return filtered

# 快速执行版本
def quick_process():
    """快速处理 - 自动完成整个流程"""
    print("=== 快速处理模式 ===")
    
    # 筛选数据
    filtered_data = filter_patient_data()
    if filtered_data is None:
        return
    
    # 自动使用分层抽样拆分
    train_data, test_data = split_patient_data(filtered_data)
    
    print(f"\n快速处理完成!")
    print(f"- 筛选数据: {len(filtered_data)} 行")
    print(f"- 训练集: {len(train_data)} 行") 
    print(f"- 测试集: {len(test_data)} 行")

if __name__ == "__main__":
    # 可以选择运行方式:
    # main()          # 交互式完整流程
    # quick_process() # 快速自动处理
    
    main()  # 默认运行交互式流程