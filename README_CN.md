# 通用文档分析系统

[English Documentation](README.md)

基于LangChain构建的强大文档分析系统，支持多种文档格式、异步处理和多知识库搜索。

## 功能特点

- **多格式文档支持**
  - PDF文件
  - Word文档 (.docx)
  - 文本文件 (.txt)
  - Markdown文件 (.md)
  - 最大文件大小：30MB

- **异步文档处理**
  - 异步文档加载和向量化
  - 大文档批处理
  - 内存高效处理，自动垃圾回收
  - 进度跟踪和错误恢复

- **高级搜索功能**
  - 多知识库并发搜索
  - 上下文感知检索
  - 结果重排序优化
  - 流式响应提升用户体验

- **使用的LangChain技术**
  - 文档加载器：PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
  - 文本分割：RecursiveCharacterTextSplitter
  - 向量存储：FAISS
  - 嵌入模型：OpenAI Embeddings
  - 检索器：
    - ContextualCompressionRetriever（上下文压缩检索）
    - MultiVectorRetriever（多向量检索）
    - EnsembleRetriever（集成检索）
    - BM25Retriever（BM25检索）
  - 重排序：CohereRerank
  - 大语言模型：ChatOpenAI
  - 索引技术：
    - VectorStoreIndexWrapper（向量存储索引包装器）
    - VectorStoreIndexCreator（向量存储索引创建器）
    - 文档索引管理

## Ragas评测功能

系统包含使用Ragas的全面评测指标：
- **上下文相关性**：评估检索文档的相关程度
- **答案忠实度**：衡量答案与提供的上下文的一致性
- **答案相关性**：评估答案与问题的相关程度
- **上下文精确度**：评估检索上下文的精确性
- **上下文召回率**：衡量检索信息的完整性

评测结果用于优化：
- 文档分块策略
- 检索方法
- 答案生成质量
- 系统整体性能

### 运行评测
```bash
# 准备测试数据
mkdir -p ragas_data/your_document_name
# 创建test_data_2.csv文件，包含列：question, ground_truth

# 运行评测
python ragas_test.py
```

评测结果将保存在：
- `ragas_data/your_document_name/evaluate_data_4.csv`

测试数据格式示例（test_data_2.csv）：
```csv
question,ground_truth
"招聘流程是什么？","招聘流程包括部门需求申请、人事核实和总经理审批。"
```

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/general_document_analysis_system.git
cd general_document_analysis_system
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
创建 `.env` 文件并添加API密钥：
```
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
```

## 使用方法

1. 启动应用：
```bash
python main.py
```

2. 通过Web界面上传文档（http://localhost:7860）

3. 开始提问文档相关问题

## 技术细节

### 异步处理
- 使用Python的asyncio实现并发操作
- 采用ThreadPoolExecutor处理CPU密集任务
- 可配置批处理大小
- 自动内存管理和垃圾回收

### 向量存储
- 使用FAISS实现高效相似度搜索
- 持久化存储，支持自动恢复
- 大文档增量更新

### 检索系统
- 多阶段检索管道
- 结合密集和稀疏检索的混合搜索
- 上下文压缩提高相关性
- 结果重排序优化质量

### 错误处理
- 完整的错误恢复机制
- 失败批次跟踪
- 自动重试临时失败
- 详细日志记录

### 性能优化
- 文档异步加载和处理
- 批量向量化减少内存使用
- 定期垃圾回收
- 检索结果缓存

### 扩展性
- 模块化设计
- 易于添加新的文档格式支持
- 可配置的检索策略
- 灵活的模型选择

## 贡献指南

欢迎提交Pull Request来改进项目！

## 许可证

本项目采用Apache许可证 - 详见LICENSE文件