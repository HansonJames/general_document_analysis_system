import os
import hashlib
from datasets import Dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate
from document_analysis import DocumentAnalysis
import pandas as pd
import asyncio

async def evaluate_document(document_name: str):
    """评测指定文档的RAG系统性能"""
    print(f"\nEvaluating document: {document_name}")
    
    # 初始化文档分析对象
    document_analysis_obj = DocumentAnalysis()
    
    # 加载指定的文档
    files = await document_analysis_obj.load_knowledge(document_name)
    if not files:
        print(f"Failed to load document: {document_name}")
        return
        
    # 设置当前知识库
    collection_hash = hashlib.md5(document_name.encode()).hexdigest()
    # 使用_DocumentAnalysis__retrievers来访问私有属性
    document_analysis_obj._current_collection = getattr(
        document_analysis_obj,
        '_DocumentAnalysis__retrievers'
    ).get(collection_hash)
    
    if not document_analysis_obj._current_collection:
        print("Failed to set current collection")
        return
        
    # 根据文件名创建测试数据目录
    test_data_dir = f'./ragas_data/{document_name.split(".")[0]}'
    os.makedirs(test_data_dir, exist_ok=True)
    
    # 测试数据文件路径
    test_data_path = f'{test_data_dir}/test_data_2.csv'
    
    try:
        # 检查测试数据文件
        if not os.path.exists(test_data_path):
            print(f"Error: Test data file not found: {test_data_path}")
            print("Please create a test data file with 'question' and 'ground_truth' columns")
            return
        
        # 读取测试数据
        df_data_2 = pd.read_csv(test_data_path, encoding='utf-8')
        if len(df_data_2) == 0:
            print(f"Error: No questions found in {test_data_path}")
            return
            
        if 'question' not in df_data_2.columns or 'ground_truth' not in df_data_2.columns:
            print(f"Error: Test data file must contain 'question' and 'ground_truth' columns")
            return
            
    except Exception as e:
        print(f"Error reading test data file: {str(e)}")
        return
    
    # 遍历问题进行评测
    for index, row in df_data_2.iterrows():
        print(f"\nProcessing question {index + 1}/{len(df_data_2)}")
        
        questions = [row['question']]  # 问题
        response = await document_analysis_obj.query(row['question'], [document_name])
        # 替换答案和上下文中的英文逗号为中文逗号
        answer = response['answer'].replace(',', '，')
        contexts_list = [doc.replace(',', '，') for doc in response['sources']]
        ground_truth = row['ground_truth'].replace(',', '，')

        answers = [answer]  # 答案
        contexts = [contexts_list]  # 检索内容
        ground_truths = [ground_truth]  # 真实答案

        data_samples = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }

        dataset = Dataset.from_dict(data_samples)

        '''
        评估指标说明:
        Retrieval检索相关:
            - Context Precision: 检索内容与问题的相关性
            - Context Recall: 检索内容与真实答案的覆盖度
        Generation生成相关:
            - Faithfulness: 生成答案与检索内容的一致性
            - Answer Relevancy: 生成答案与问题的相关性
        '''

        result = evaluate(
            dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
        )
        
        # 格式化输出结果
        print("\n" + "="*50)
        print(f"Question {index + 1}: {row['question']}")
        print("-"*50)
        print("Evaluation Results:")
        print(result)
        print("\nDetailed Metrics:")
        df = result.to_pandas()
        print(df)
        
        # 保存结果到CSV
        output_path = f'./ragas_data/{document_name.split(".")[0]}/evaluate_data_4.csv'
        
        # 替换DataFrame中的英文逗号
        for col in df.columns:
            if df[col].dtype == 'object':  # 只处理字符串类型的列
                df[col] = df[col].astype(str).apply(lambda x: x.replace(',', '，'))
        
        # 如果是第一个问题且文件不存在，直接写入
        if index == 0 and not os.path.exists(output_path):
            df.to_csv(output_path, index=False, encoding='utf-8')
        else:
            # 否则追加到现有文件，不包含header
            df.to_csv(output_path, mode='a', header=False, index=False, encoding='utf-8')

async def main():
    """主函数"""
    # 要评测的文档列表
    documents = [
        # "新编十万个为什么--01--动物.pdf",
        "人事管理流程.docx",
    ]
    
    # 依次评测每个文档
    for doc in documents:
        await evaluate_document(doc)

if __name__ == "__main__":
    asyncio.run(main())
