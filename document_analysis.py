import hashlib
import os.path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from tqdm import tqdm
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    RePhraseQueryRetriever,
)
from langchain_community.retrievers import BM25Retriever
from langchain_core.document_loaders import BaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_cohere import CohereRerank
from unstructured.file_utils.filetype import FileType, detect_filetype
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import logging
import shutil
from config import (
    faiss_path,
    knowledge_path,
    llm_models,
)
from loguru import logger

# 配置日志记录
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
logging.getLogger("langchain.retrievers.re_phraser").setLevel(logging.INFO)

# 加载环境变量
load_dotenv()


class MyCustomLoader(BaseLoader):
    # 支持加载的文件类型
    file_type = {
        FileType.TXT: (TextLoader, {"autodetect_encoding": True}),
        FileType.DOC: (UnstructuredWordDocumentLoader, {}),
        FileType.DOCX: (UnstructuredWordDocumentLoader, {}),
        FileType.PDF: (PyPDFLoader, {}),
        FileType.MD: (UnstructuredMarkdownLoader, {}),
    }

    def __init__(self, file_path: str):
        loader_class, params = self.file_type[detect_filetype(file_path)]
        logger.info(f"loader_class: {loader_class}")

        self.loader: BaseLoader = loader_class(file_path, **params)
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )

    async def load(self) -> List[Dict]:
        """异步加载和切分文档"""

        def load_and_split():
            try:
                logger.info("开始读取文档内容...")
                documents = self.loader.load()
                if not documents:
                    logger.error("文档加载结果为空")
                    return None

                logger.info(f"文档加载完成，开始分块处理...")
                chunks = []
                for doc in tqdm(documents, desc="Processing document"):
                    splits = self.text_splitter.split_documents([doc])
                    chunks.extend(splits)

                logger.info(f"文档分块完成，共 {len(chunks)} 个块")
                return chunks

            except Exception as e:
                logger.error(f"文档处理失败: {str(e)}")
                raise

        with ThreadPoolExecutor() as executor:
            try:
                logger.info("开始异步处理...")
                result = await asyncio.get_event_loop().run_in_executor(
                    executor, load_and_split
                )

                if result:
                    logger.info("异步处理完成")
                    return result
                else:
                    logger.error("异步处理失败")
                    raise ValueError("文档处理失败")

            except Exception as e:
                logger.error(f"异步加载失败: {str(e)}")
                raise


class DocumentProcessor:
    def __init__(
        self, chunk_size: int = 500, chunk_overlap: int = 50, temperature: float = 0.3
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化OpenAI嵌入模型
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # 初始化ChatGPT模型
        self.llm = ChatOpenAI(temperature=temperature, model_name=llm_models[1])

        # 初始化Cohere Rerank
        self.reranker = CohereRerank(model="rerank-multilingual-v3.0")

        # 初始化向量存储
        self.vector_store = None

        # 初始化检索器
        self.compression_retriever = None

        # 初始化问答链的模板
        self.qa_prompt = ChatPromptTemplate.from_template(
            """
        你是一个专业的问答助手。请基于以下上下文信息，回答用户的问题。
        如果上下文信息不足以回答问题，请明确说明。请保持回答的准确性和客观性。

        上下文信息:
        {context}

        用户问题: {question}

        请给出详细的回答：
        """
        )

    async def process_document(
        self,
        file_path: str,
        persist_directory: str,
    ) -> None:
        """异步处理文档，支持批处理和错误恢复"""
        try:
            # 检查向量存储是否已存在
            if os.path.exists(os.path.join(persist_directory, "index.faiss")):
                logger.info(f"Loading existing vector store from {persist_directory}")
                try:
                    self.vector_store = FAISS.load_local(
                        persist_directory,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    logger.info("Vector store loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading vector store: {str(e)}")
                    # 如果加载失败，删除可能损坏的文件并重新创建
                    import shutil
                    shutil.rmtree(persist_directory, ignore_errors=True)
                    logger.info("Removed corrupted vector store, will recreate")
                    raise
            
            if not self.vector_store:
                # 创建持久化目录
                os.makedirs(persist_directory, exist_ok=True)

                # 异步加载文档
                loader = MyCustomLoader(file_path)
                documents = await loader.load()
                if not documents:
                    raise ValueError("No documents loaded")

                logger.info(f"开始创建向量存储，共有 {len(documents)} 个文档块")

                # 优化批处理大小
                batch_size = min(8, max(1, len(documents) // 20))  # 动态调整批大小
                batches = [
                    documents[i : i + batch_size]
                    for i in range(0, len(documents), batch_size)
                ]
                total_batches = len(batches)

                # 使用第一批创建向量存储
                logger.info(f"Creating vector store with first batch (1/{total_batches})")
                self.vector_store = FAISS.from_documents(
                    documents=batches[0], embedding=self.embeddings
                )

                # 使用ThreadPoolExecutor处理剩余批次
                with ThreadPoolExecutor() as executor:
                    for i, batch in enumerate(batches[1:], 1):
                        try:
                            logger.info(f"Processing batch {i+1}/{total_batches}")
                            # 异步处理每个批次
                            await asyncio.get_event_loop().run_in_executor(
                                executor,
                                self.vector_store.add_documents,
                                batch
                            )
                            # 内存管理
                            if i % 3 == 0:  # 每3个批次后进行内存管理
                                import gc
                                gc.collect()
                                await asyncio.sleep(0.5)
                                
                            # 定期保存进度
                            if i % 5 == 0:  # 每5个批次保存一次
                                self.vector_store.save_local(persist_directory)
                                logger.info(f"Progress saved at batch {i+1}")
                                
                        except Exception as e:
                            logger.error(f"Error processing batch {i+1}: {str(e)}")
                            # 记录失败的批次以便后续重试
                            with open(os.path.join(persist_directory, "failed_batches.txt"), "a") as f:
                                f.write(f"Batch {i+1}: {str(e)}\n")
                            continue

                # 最终保存
                self.vector_store.save_local(persist_directory)
                logger.info("Vector store saved successfully")

                # 创建检索器
                retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
                
                # 创建压缩检索器
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=self.reranker,
                    base_retriever=retriever
                )
                
                return compression_retriever
                logger.info("Vector store saved successfully")

            def create_retrievers(docs=None):
                """创建检索器"""
                # 创建FAISS检索器
                faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

                if docs is not None:
                    # 创建BM25检索器
                    bm25_retriever = BM25Retriever.from_documents(docs)
                    bm25_retriever.k = 4
                else:
                    # 如果没有文档，只使用FAISS检索器
                    return faiss_retriever

                # 创建RePhraseQuery检索器
                from langchain.prompts import PromptTemplate
                
                prompt_template = """Below is a user question. Please rephrase it to be more descriptive while keeping its original meaning.
                Question: {question}
                Rephrased question:"""
                
                prompt = PromptTemplate(
                    input_variables=["question"],
                    template=prompt_template,
                )
                
                # 使用新的Runnable语法
                rephrase_chain = prompt | self.llm | StrOutputParser()
                
                rephrase_retriever = RePhraseQueryRetriever(
                    retriever=faiss_retriever,
                    llm_chain=rephrase_chain,
                    k=4
                )

                # 组合所有检索器
                retrievers = [
                    faiss_retriever,
                    bm25_retriever,
                    rephrase_retriever
                ]
                weights = [0.5, 0.3, 0.2]  # 权重之和为1
                
                # 创建集成检索器
                ensemble_retriever = EnsembleRetriever(
                    retrievers=retrievers,
                    weights=weights
                )

                return ensemble_retriever

            # 创建检索器
            base_retriever = create_retrievers(documents if 'documents' in locals() else None)

            # 创建压缩检索器
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.reranker,
                base_retriever=base_retriever
            )

            logger.info("向量存储创建完成")

            return self.compression_retriever

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return None

    async def search_and_answer(self, query: str) -> Dict:
        """搜索相关文档并回答问题"""
        if self.compression_retriever is None:
            raise ValueError(
                "Retriever not initialized. Please process documents first."
            )

        try:
            # 获取压缩后的相关文档
            compressed_docs = await asyncio.get_event_loop().run_in_executor(
                None, self.compression_retriever.get_relevant_documents, query
            )

            # 组合上下文
            context = "\n\n".join(doc.page_content for doc in compressed_docs)

            # 构建问答链
            qa_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            # 生成答案
            answer = await qa_chain.ainvoke({"context": context, "question": query})

            return {
                "answer": answer,
                "sources": [doc.page_content for doc in compressed_docs[:2]],
            }

        except Exception as e:
            logger.error(f"Error in search and answer: {str(e)}")
            return {"answer": "抱歉，处理您的问题时出现错误。", "sources": []}

    def cleanup(self):
        """清理资源"""
        try:
            # Chroma会自动管理资源，不需要手动关闭
            self.vector_store = None
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


class DocumentAnalysis:
    def __init__(self):
        self.processor = DocumentProcessor()
        self._current_collection = None
        self.chat_history = []
        self.max_history = 10
        self.__retrievers = {}  # 存储所有检索器的字典
        self.__embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            temperature=0.3,
            model_name="gpt-4"
        )
        
        # 初始化reranker
        self.reranker = CohereRerank(
            model="rerank-multilingual-v3.0"
        )

    def add_to_history(self, question: str, answer: str):
        """添加对话到历史记录"""
        self.chat_history.append([question, answer])
        # 保持历史记录在最大限制内
        if len(self.chat_history) > self.max_history:
            self.chat_history.pop(0)

    def clear_history(self):
        """清除历史记录"""
        self.chat_history = []
        return []

    def get_history_message(self):
        """获取历史消息"""
        return self.chat_history

    async def stream(self, question, collection_names=None, model=None, max_length=512, temperature=0.1):
        """流式生成回答"""
        try:
            # 如果没有选择知识库，直接使用大语言模型回答
            if not collection_names or not isinstance(collection_names, list) or len(collection_names) == 0:
                logger.info("No knowledge base selected, using LLM directly")
                async for chunk in self.llm.astream(question):
                    yield {"answer": chunk}
                return

            # 获取所有选中文档的检索器
            retrievers = []
            for name in collection_names:
                # 加载每个知识库
                await self.load_knowledge(name)
                collection_hash = hashlib.md5(name.encode()).hexdigest()
                retriever = self.__retrievers.get(collection_hash)
                if retriever:
                    retrievers.append(retriever)

            if not retrievers:
                yield {"answer": "所选知识库尚未加载，请重新选择"}
                return

            logger.info(f"Using retrievers for collections: {collection_names}")

            from langchain.retrievers import MultiVectorRetriever
            from langchain.storage import InMemoryStore
            from langchain_community.embeddings import OpenAIEmbeddings
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            try:
                # 创建存储
                docstore = InMemoryStore()
                id_key = "doc_id"
                
                # 创建文本分割器
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
                
                # 为每个检索器创建MultiVectorRetriever
                all_docs = []
                for retriever in retrievers:
                    try:
                        # 获取检索器的向量存储
                        base_retriever = retriever.base_retriever
                        vectorstore = base_retriever.vectorstore
                        vectorstore_str = str(vectorstore)
                        
                        # 匹配知识库
                        source = None
                        for name in collection_names:
                            collection_hash = hashlib.md5(name.encode()).hexdigest()
                            if collection_hash in self.__retrievers:
                                current_retriever = self.__retrievers[collection_hash]
                                current_vectorstore = current_retriever.base_retriever.vectorstore
                                if current_vectorstore == vectorstore:
                                    source = name
                                    break
                                
                        if source is None:
                            logger.warning(f"Could not find matching collection for vectorstore: {vectorstore_str}")
                            continue
                            
                        logger.info(f"Processing retriever for {source}")
                        
                        # 获取文档
                        docs = await retriever.aget_relevant_documents(question)
                        logger.info(f"Retrieved {len(docs)} documents from {source}")
                        
                        # 记录文档得分
                        for doc in docs:
                            try:
                                # 使用base_retriever的vectorstore计算相似度
                                results = retriever.base_retriever.vectorstore.similarity_search_with_score(doc.page_content, k=1)
                                if results:
                                    similarity = results[0][1]
                                    doc.metadata['score'] = similarity
                                    doc.metadata['source'] = source
                                    logger.info(f"Document from {source} has similarity score: {similarity}")
                                    all_docs.append(doc)
                            except Exception as e:
                                logger.error(f"Error calculating similarity: {str(e)}")
                                # 如果无法计算相似度，使用默认分数
                                doc.metadata['score'] = 1.0
                                doc.metadata['source'] = source
                                all_docs.append(doc)
                    except Exception as e:
                        logger.error(f"Error processing retriever: {str(e)}")
                        continue
                
                if not all_docs:
                    yield {"answer": "未能从知识库中检索到相关信息"}
                    return
                    
                # 并发检索所有知识库
                all_docs = []
                async with asyncio.TaskGroup() as tg:
                    tasks = []
                    for retriever in retrievers:
                        try:
                            # 异步获取相关文档
                            docs = await retriever.aget_relevant_documents(question)
                            logger.info(f"Retrieved {len(docs)} documents from {retriever}")
                            
                            # 计算相似度分数
                            for doc in docs:
                                try:
                                    # 使用base_retriever的vectorstore计算相似度
                                    results = retriever.base_retriever.vectorstore.similarity_search_with_score(
                                        doc.page_content, k=1
                                    )
                                    if results:
                                        similarity = results[0][1]
                                        doc.metadata['score'] = similarity
                                        doc.metadata['source'] = str(retriever)
                                        logger.info(f"Document has similarity score: {similarity}")
                                        all_docs.append(doc)
                                except Exception as e:
                                    logger.error(f"Error calculating similarity: {str(e)}")
                                    # 使用默认分数
                                    doc.metadata['score'] = 1.0
                                    doc.metadata['source'] = str(retriever)
                                    all_docs.append(doc)
                        except Exception as e:
                            logger.error(f"Error retrieving from {retriever}: {str(e)}")
                            continue

                if not all_docs:
                    yield {"answer": "未能从知识库中检索到相关信息"}
                    return

                # 记录所有文档的得分情况
                logger.info("\nAll documents scores:")
                for doc in all_docs:
                    source = doc.metadata.get('source', 'unknown')
                    score = doc.metadata.get('score', 0)
                    logger.info(f"Document from {source} has score: {score}")
                    logger.info(f"Content preview: {doc.page_content[:100]}...")

                # 使用Cohere重排序
                try:
                    reranker = CohereRerank()
                    reranked_docs = await reranker.arerank(
                        query=question,
                        documents=[doc.page_content for doc in all_docs],
                    )
                    
                    # 更新文档的相似度分数
                    for i, doc in enumerate(all_docs):
                        doc.metadata['score'] = reranked_docs[i].relevance_score
                except Exception as e:
                    logger.error(f"Error during reranking: {str(e)}")

                # 按相似度得分排序（分数高的更相关）
                all_docs = sorted(
                    all_docs,
                    key=lambda x: x.metadata.get('score', 0),
                    reverse=True
                )
                
                # 取最相关的文档
                relevant_docs = all_docs[:4]
                logger.info(f"\nSelected {len(relevant_docs)} most relevant documents:")
                
                # 记录选中的文档来源和得分
                for doc in relevant_docs:
                    source = doc.metadata.get('source', 'unknown')
                    score = doc.metadata.get('score', 0)
                    logger.info(f"Selected document from: {source}, score: {score}")
                    logger.info(f"Content: {doc.page_content}\n")

                # 生成答案
                try:
                    context = "\n\n".join(doc.page_content for doc in relevant_docs)
                    try:
                        # 创建提示模板
                        prompt = ChatPromptTemplate.from_template(
                            """基于以下上下文回答问题。如果上下文信息不足，请说明。
                            
                            上下文：
                            {context}
                            
                            问题：{question}
                            
                            请用中文回答。
                            """
                        )
                        
                        # 创建链
                        chain = prompt | self.llm | StrOutputParser()
                        
                        # 设置超时时间为30秒
                        async with asyncio.timeout(30):
                            # 生成答案
                            answer = await chain.ainvoke({
                                "context": context,
                                "question": question
                            })
                            
                            # 分段返回答案
                            chunk_size = 50  # 每次返回50个字符
                            for i in range(0, len(answer), chunk_size):
                                chunk = answer[i:i + chunk_size]
                                yield {"answer": chunk}
                                await asyncio.sleep(0.05)  # 减少延迟
                                
                    except asyncio.TimeoutError:
                        logger.error("Answer generation timed out")
                        yield {"answer": "生成答案超时，请重试"}
                    except Exception as e:
                        logger.error(f"Error generating answer: {str(e)}")
                        yield {"answer": "生成答案时出现错误"}
                    
                    # 添加到历史记录
                    self.add_to_history(question, answer)
                    
                    # 分段返回答案
                    chunk_size = 50  # 每次返回50个字符
                    for i in range(0, len(answer), chunk_size):
                        chunk = answer[i:i + chunk_size]
                        yield {"answer": chunk}
                        await asyncio.sleep(0.05)  # 减少延迟
                except Exception as e:
                    logger.error(f"Error generating answer: {str(e)}")
                    yield {"answer": "生成答案时出现错误"}
                    
            except Exception as e:
                logger.error(f"Error in stream: {str(e)}")
                yield {"answer": "处理查询时出现错误"}

        except Exception as e:
            logger.error(f"Error in stream: {str(e)}")
            yield {"answer": "抱歉，处理您的问题时出现错误。"}

    async def load_knowledge(self, target_file=None):
        """异步加载知识库文件
        Args:
            target_file: 指定要加载的文件名，如果为None则加载所有文件
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(knowledge_path), exist_ok=True)

            # 获取要处理的文件列表
            all_files = os.listdir(knowledge_path)
            logger.info(f'Available files: {all_files}')

            files_to_process = []
            if target_file:
                if target_file in all_files:
                    files_to_process = [target_file]
                    logger.info(f'Loading specific file: {target_file}')
                else:
                    logger.error(f'Target file not found: {target_file}')
                    return []
            else:
                files_to_process = [f for f in all_files if str(f).strip()]

            loaded_files = []
            # 使用ThreadPoolExecutor进行并发处理
            with ThreadPoolExecutor() as executor:
                async def process_file(file):
                    try:
                        # 得到知识库的路径
                        file_path = os.path.join(knowledge_path, file)
                        logger.info(f'file_path: {file_path}')

                        # 知识库文件名进行md5编码
                        collection_name = hashlib.md5(file.encode()).hexdigest()
                        logger.info(f'collection_name: {collection_name}')

                        logger.info(f'self.__retrievers: {self.__retrievers}')

                        # 检查向量存储是否已存在
                        persist_dir = os.path.abspath(os.path.join(faiss_path, collection_name)).replace("\\", "/")
                        if os.path.exists(persist_dir) and os.path.exists(os.path.join(persist_dir, "index.faiss")):
                            logger.info(f"Loading existing vector store from {persist_dir}")
                            # 异步加载已存在的向量存储
                            if collection_name not in self.__retrievers:
                                retriever = await self.processor.process_document(file_path, persist_dir)
                                if retriever:
                                    self.__retrievers[collection_name] = retriever
                                    logger.info(f"Successfully loaded vector store for {file}")
                                else:
                                    logger.error(f"Failed to load vector store for {file}")
                                    return None
                        else:
                            # 异步创建新的向量存储
                            logger.info(f"Creating new vector store in {persist_dir}")
                            loader = MyCustomLoader(file_path)
                            logger.info(f'loader: {loader}')
                            retriever = await self.processor.process_document(file_path, persist_dir)
                            if retriever:
                                self.__retrievers[collection_name] = retriever
                                logger.info(f"Successfully created vector store for {file}")
                            else:
                                logger.error(f"Failed to create vector store for {file}")
                                return None

                        return file
                    except Exception as e:
                        logger.error(f"Error processing file {file}: {str(e)}")
                        return None

                # 并发处理所有文件
                tasks = [process_file(file) for file in files_to_process]
                results = await asyncio.gather(*tasks)
                loaded_files = [f for f in results if f is not None]

            logger.info(f"Loaded files: {loaded_files}")
            logger.info(f"self.__retrievers: {self.__retrievers}")
            return loaded_files

        except Exception as e:
            logger.error(f"Error in load_knowledge: {str(e)}")
            return []

        # 确保文件名格式正确并排序
        loaded_files = sorted([str(f).strip() for f in loaded_files if f])
        logger.info(f'Loaded files: {loaded_files}')
        logger.info(f'self.__retrievers: {self.__retrievers}')

        return loaded_files

    async def upload_knowledge(self, temp_file):
        """上传并处理文档"""
        try:
            # 检查文件大小
            file_size = os.path.getsize(temp_file)
            if file_size > 55 * 1024 * 1024:  # 55MB
                return f"文件大小超过55MB限制 ({file_size / 1024 / 1024:.2f}MB)", None

            file_name = os.path.basename(temp_file)
            file_path = os.path.join(knowledge_path, file_name)

            # 如果文件不存在就copy
            if not os.path.exists(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                shutil.copy(temp_file, file_path)

            # 更新知识库选项
            import gradio as gr
            return None, gr.update(choices=await self.load_knowledge())

        except Exception as e:
            logger.error(f"Error uploading knowledge: {str(e)}")
            return f"文件处理失败: {str(e)}", None

    async def query(self, question: str, collection_names: list = None, chatbot=None) -> Dict:
        """查询知识库"""
        try:
            if not collection_names:
                return {"answer": "请先选择知识库", "sources": []}

            # 获取所有选中文档的检索器
            retrievers = []
            for name in collection_names:
                collection_hash = hashlib.md5(name.encode()).hexdigest()
                retriever = self.__retrievers.get(collection_hash)
                if retriever:
                    retrievers.append(retriever)

            if not retrievers:
                return {"answer": "所选知识库尚未加载，请重新选择", "sources": []}

            # 合并所有检索器的结果
            all_docs = []
            final_result = None
            for retriever in retrievers:
                # 设置当前检索器
                self._current_collection = retriever
                # 执行查询
                result = await self.processor.search_and_answer(question)
                all_docs.extend(result.get("sources", []))
                final_result = result  # 保存最后一个结果

            if final_result:
                # 更新最终结果的源文档
                final_result["sources"] = all_docs[:4]  # 限制返回的源文档数量
                # 添加到历史记录
                self.add_to_history(question, final_result["answer"])
                
                # 如果提供了chatbot，更新它的状态
                if chatbot is not None:
                    chatbot = chatbot + [(question, final_result["answer"])]
                    return final_result, chatbot
                return final_result
            
            return {"answer": "处理查询时出现错误", "sources": []}

        except Exception as e:
            logger.error(f"Error querying knowledge base: {str(e)}")
            error_msg = "抱歉，处理您的问题时出现错误。"
            if chatbot is not None:
                chatbot = chatbot + [(question, error_msg)]
                return {"answer": error_msg, "sources": []}, chatbot
            return {"answer": error_msg, "sources": []}

    def cleanup(self):
        """清理资源"""
        try:
            self.processor.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


async def main():
    # 初始化处理器
    doc_analysis = DocumentAnalysis()

    try:
        # 处理文档
        file_path = (
            "example_documents/新编十万个为什么--第6版--12册/新编十万个为什么--01--动物.pdf"
        )
        result = await doc_analysis.upload_knowledge(file_path)
        if result[0] is not None:
            logger.error(f"Error uploading document: {result[0]}")
            return

        # 测试问答
        questions = ["为什么鸡吃小石子", "为什么鸽子会成双成对", "为什么燕子低飞要下雨"]

        for question in questions:
            logger.info(f"\n问题: {question}")
            result = await doc_analysis.query(question)
            logger.info("回答：")
            logger.info(result["answer"])

    finally:
        doc_analysis.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
