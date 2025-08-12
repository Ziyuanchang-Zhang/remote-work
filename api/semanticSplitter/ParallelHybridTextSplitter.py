import csv
import os
import threading
from typing import List, Dict, Any, Union, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from tree_sitter import Language, Parser
from adalflow.core.types import Document

# 导入不同语言的Tree-sitter模块
try:
    import tree_sitter_c as tsc
except ImportError:
    tsc = None
    
try:
    import tree_sitter_cpp as tscpp
except ImportError:
    tscpp = None
    
try:
    import tree_sitter_python as tspy
except ImportError:
    tspy = None

import logging
logger = logging.getLogger(__name__)


class MultiLanguageTextSplitter:
    """
    多语言文本分割器：
    - 对C/C++文件进行函数级别分割
    - 对Python文件进行函数/类级别分割
    - 对其他文件使用普通文本分割器
    - 支持并行处理
    """
    
    def __init__(self, text_splitter, configs: Dict[str, Any] = None,file_path:Optional[str] = None):
        """初始化多语言分割器"""
        self.text_splitter = text_splitter
        self.configs = configs or {}
        self.file_path = file_path
        # 初始化并行处理配置
        self._init_parallel_config()
        
        # 初始化CSV配置
        self._csv_lock = threading.Lock()
        self.csv_buffer_size = 131072  # 128KB缓冲区
        
        # 初始化解析器
        self._init_parsers()

    def _init_parallel_config(self):
        """初始化并行处理配置"""
        cpu_cores = cpu_count()
        is_development = os.environ.get("NODE_ENV") != "production"
        
        # 设置默认worker数量
        if is_development:
            default_workers = min(6, max(2, cpu_cores // 2))
        else:
            default_workers = min(28, cpu_cores // 3)
        
        self.max_workers = self.configs.get('max_workers', default_workers)
        self.use_multiprocessing = self.configs.get('use_multiprocessing', True)
        
        # 设置批次大小
        base_batch_size = 200 if is_development else 2000
        self.batch_size = min(
            500 if not is_development else 50,
            max(10 if is_development else 100, base_batch_size // self.max_workers)
        )
        
        logger.info(f"Environment: {'Development' if is_development else 'Production'}")
        logger.info(f"Using {self.max_workers} workers, batch_size: {self.batch_size}")

    def _init_parsers(self):
        """初始化各种语言的解析器"""
        self.parsers = {}
        self.languages = {}
        
        # 语言配置
        lang_configs = [
            ('c', tsc, "C"),
            ('cpp', tscpp, "C++"),
            ('python', tspy, "Python")
        ]
        
        for lang_key, lang_module, lang_name in lang_configs:
            if lang_module:
                try:
                    self.languages[lang_key] = Language(lang_module.language())
                    self.parsers[lang_key] = Parser(self.languages[lang_key])
                    logger.info(f"{lang_name} parser initialized successfully")
                except Exception as e:
                    logger.warning(f"{lang_name} parser initialization failed: {e}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表的主入口方法"""
        if not documents:
            return []
        
        if len(documents) == 1:
            return self._process_single_document(documents[0])
        
        # 根据配置选择处理方式
        if self.use_multiprocessing:
            return self._split_documents_parallel(documents)
        elif self.configs.get('use_threading', False):
            return self._split_documents_threaded(documents)
        else:
            return self._split_documents_sequential(documents)

    def _split_documents_parallel(self, documents: List[Document]) -> List[Document]:
        """使用多进程并行分割文档列表"""
        doc_count = len(documents)
        logger.info(f"Processing {doc_count} documents using parallel processing")
        
        # 分批处理
        batches = [documents[i:i + self.batch_size] 
                  for i in range(0, doc_count, self.batch_size)]
        
        logger.info(f"Processing {doc_count} documents in {len(batches)} batches")
        
        result_chunks = []
        all_func_calls = {}
        
        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # 准备并提交批量任务
                future_to_batch = {
                    executor.submit(process_document_batch, {
                        'documents': batch,
                        'text_splitter': self.text_splitter,
                        'configs': self.configs,
                        'batch_id': batch_idx
                    }): batch_idx for batch_idx, batch in enumerate(batches)
                }
                
                # 收集结果
                self._collect_parallel_results(future_to_batch, result_chunks, all_func_calls)
                
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            raise RuntimeError(f"Failed to process {doc_count} documents in parallel: {e}")

        # 异步写入CSV
        if all_func_calls:
            threading.Thread(target=self._write_csv_async, args=(all_func_calls,), daemon=True).start()
    
        logger.info(f"Parallel processing completed. Generated {len(result_chunks)} chunks")
        return result_chunks

    def _collect_parallel_results(self, future_to_batch, result_chunks, all_func_calls):
        """收集并行处理结果"""
        completed = 0
        progress_interval = max(1, len(future_to_batch) // 20)
        
        for future in as_completed(future_to_batch):
            batch_id = future_to_batch[future]
            try:
                result = future.result()
                
                if result.get('success', False):
                    result_chunks.extend(result['chunks'])
                    if result.get('func_calls'):
                        all_func_calls.update(result['func_calls'])
                
                completed += 1
                if completed % progress_interval == 0:
                    progress = (completed / len(future_to_batch)) * 100
                    completed_docs = completed * self.batch_size
                    logger.info(f"Progress: {progress:.1f}% ({completed}/{len(future_to_batch)} batches, ~{completed_docs} documents)")
                    
            except Exception as e:
                logger.error(f"Batch {batch_id} failed: {e}")

    def _split_documents_threaded(self, documents: List[Document]) -> List[Document]:
        """使用多线程并行分割文档列表"""
        result_chunks = []
        all_func_calls = {}
        results_lock = threading.Lock()
        
        def process_doc_thread(doc):
            try:
                chunks, func_calls = self._process_document_internal(doc)
                
                with results_lock:
                    result_chunks.extend(chunks)
                    if func_calls:
                        file_path = doc.meta_data.get('file_path', 'unknown')
                        all_func_calls[file_path] = func_calls
                        
            except Exception as e:
                logger.error(f"Error processing document {doc.meta_data.get('file_path', 'unknown')}: {e}")
                # 回退处理
                fallback_chunks = self._fallback_process(doc, str(e))
                with results_lock:
                    result_chunks.extend(fallback_chunks)

        logger.info(f"Starting threaded processing of {len(documents)} documents")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_doc_thread, doc) for doc in documents]
            
            completed = 0
            for future in as_completed(futures):
                try:
                    future.result()
                    completed += 1
                    if completed % 10 == 0:
                        logger.info(f"Completed processing {completed}/{len(documents)} documents")
                except Exception as e:
                    logger.error(f"Thread execution failed: {e}")
        
        # 批量写入函数调用数据
        if all_func_calls:
            self._write_csv_async(all_func_calls)
        
        logger.info(f"Threaded processing completed. Total chunks generated: {len(result_chunks)}")
        return result_chunks

    def _split_documents_sequential(self, documents: List[Document]) -> List[Document]:
        """顺序处理文档"""
        result_chunks = []
        
        for doc in documents:
            try:
                chunks = self._process_single_document(doc)
                result_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing document {doc.meta_data.get('file_path', 'unknown')}: {e}")
                fallback_chunks = self._fallback_process(doc, str(e))
                result_chunks.extend(fallback_chunks)
        
        logger.info(f"Sequential processing completed. Total chunks generated: {len(result_chunks)}")
        return result_chunks

    def _process_single_document(self, doc: Document) -> List[Document]:
        """处理单个文档"""
        chunks, _ = self._process_document_internal(doc)
        return chunks

    def _process_document_internal(self, doc: Document) -> tuple[List[Document], Dict[str, List[str]]]:
        """内部文档处理逻辑"""
        language = self.detect_file_language(doc)
        file_path = doc.meta_data.get('file_path', 'unknown')
        
        if language:
            logger.info(f"Processing {language.upper()} file: {file_path}")
            chunks = self.build_language_chunks(doc.text, doc.meta_data, language)
            func_calls = self.extract_function_calls_data(doc.text, language) if language == "c" else {}
            logger.info(f"Generated {len(chunks)} chunks for {language.upper()} file")
        else:
            logger.info(f"Processing text file: {file_path}")
            chunks = self._process_text_document(doc)
            func_calls = {}
        
        return chunks, func_calls

    def _process_text_document(self, doc: Document) -> List[Document]:
        """处理普通文本文档"""
        text_chunks = self.text_splitter.call([doc])
        
        for i, chunk_text in enumerate(text_chunks):
            chunk_text.meta_data.update({
                "chunk_type": "text",
                "chunk_id": f"text_{i}",
                "token_count": len(chunk_text.text.split())
            })
        
        return text_chunks

    def _fallback_process(self, doc: Document, error_msg: str) -> List[Document]:
        """回退处理逻辑"""
        try:
            text_chunks = self.text_splitter.call([doc])
            
            for i, chunk_text in enumerate(text_chunks):
                chunk_text.meta_data.update({
                    "chunk_type": "text_fallback",
                    "chunk_id": f"fallback_{i}",
                    "token_count": len(chunk_text.text.split()),
                    "error": error_msg
                })
            
            return text_chunks
        except Exception as fallback_error:
            logger.error(f"Fallback splitting also failed: {fallback_error}")
            return []

    def extract_function_calls_data(self, code: str, language: str) -> Dict[str, List[str]]:
        """提取函数调用数据"""
        if language != "c" or "c" not in self.parsers:
            return {}
        
        items, _ = self.extract_functions_and_classes_unified(code, language)
        return {item["name"]: item["calls"] for item in items if item["calls"]}

    def _write_csv_async(self, all_func_calls):
        """异步CSV写入"""
        try:
            self._write_func_calls_to_csv_optimized(all_func_calls)
        except Exception as e:
            logger.error(f"Async CSV write failed: {e}")

    def _write_func_calls_to_csv_optimized(self, all_func_calls: Dict[str, Dict[str, List[str]]]):
        """优化的CSV写入"""
        if not all_func_calls:
            return
        dir_name = os.path.join(os.getcwd(), "function_call")
        
        # 确保目录存在
        os.makedirs(dir_name, exist_ok=True)

        # 修复：使用实例变量而不是参数中的同名变量
        csv_filename = f"{self.file_path}_func_calls.csv" if self.file_path else "func_calls.csv"
        csv_file_path = os.path.join(dir_name, csv_filename)
        
        write_header = not os.path.exists(csv_file_path)
        
        
        # 预处理所有数据
        rows_to_write = []
        if write_header:
            rows_to_write.append(["subject", "predicate", "object"])
        
        for file_path_key, func_calls in all_func_calls.items():
            for func, calls in func_calls.items():
                for callee in calls:
                    rows_to_write.append([func, "calls", callee])
        
        if len(rows_to_write) <= 1:
            return
        
        total_calls = len(rows_to_write) - (1 if write_header else 0)
        logger.info(f"Writing {total_calls} function calls to CSV")
        
        with self._csv_lock:
            try:
                with open(csv_file_path, "a", newline='', buffering=self.csv_buffer_size) as f:
                    writer = csv.writer(f)
                    writer.writerows(rows_to_write)
                    f.flush()
                
                logger.info(f"Successfully wrote {total_calls} function calls to CSV")
            except Exception as e:
                logger.error(f"CSV writing failed: {e}")

    def detect_file_language(self, document: Document) -> Optional[str]:
        """检测文件语言类型"""
        file_type = document.meta_data.get("type", "").lower()
        file_path = document.meta_data.get("file_path", "").lower()
        
        # 语言扩展名映射
        language_extensions = {
            "c": ["c", "h"],
            "cpp": ["cpp", "cc", "cxx", "hpp", "hxx", "h++", "c++"],
            "python": ["py", "pyx", "pyi"]
        }
        
        for language, extensions in language_extensions.items():
            if (file_type in extensions or 
                any(file_path.endswith(f".{ext}") for ext in extensions)):
                return language if language in self.parsers else None
        
        return None

    def extract_functions_and_classes_unified(self, code: str, language: str) -> tuple[List[Dict[str, Any]], List[tuple]]:
        """统一的函数和类提取方法"""
        lines = code.splitlines(keepends=True)
        parser = self.parsers[language]
        tree = parser.parse(code.encode("utf8"))
        root_node = tree.root_node
        items = []
        item_ranges = []
        
        # 定义不同语言的目标节点类型
        # target_node_types = {
        #     'c': ['function_definition'],
        #     'cpp': ['function_definition', 'method_definition'],
        #     'python': ['function_definition', 'class_definition']
        # }

        # 定义不同语言的目标节点类型（新增函数声明节点）
        target_node_types = {
            'c': ['function_definition', 'declaration'],  # 新增 declaration 识别C函数声明
            'cpp': ['function_definition', 'method_definition', 'declaration'],  # 新增C++声明
            'python': ['function_definition', 'class_definition']
        }
        
        target_types = target_node_types.get(language, ['function_definition'])
        
        def traverse(node, parent_class=None):
            if node.type in target_types:
                item_data = self._extract_item_data(node, lines, code, language, parent_class)
                if item_data:
                    items.append(item_data)
                    item_ranges.append((item_data["start_line"], item_data["end_line"]))
                
                # 如果是类，递归处理其中的方法
                if node.type == "class_definition":
                    for child in node.children:
                        traverse(child, item_data["original_name"])
            else:
                for child in node.children:
                    traverse(child, parent_class)

        traverse(root_node)
        return items, item_ranges

    def _find_function_name_in_declarator(self, declarator_node):
        """从function_declarator节点中精准提取函数名（忽略返回值中的标识符）"""
        # 递归遍历，但优先处理函数名所在的路径
        def traverse(n):
            # 函数名的identifier通常直接属于function_declarator或其直接子节点
            if n.type == "identifier":
                # 检查父节点是否为function_declarator或指针声明（排除结构体名）
                parent_type = n.parent.type if n.parent else ""
                if parent_type in ["function_declarator", "pointer_declarator", "array_declarator"]:
                    return n
            
            # 优先遍历可能包含函数名的子节点（跳过返回值类型相关节点）
            for child in n.children:
                # 忽略返回值中的结构体类型节点（如struct_specifier）
                if child.type in ["struct_specifier", "union_specifier", "enum_specifier"]:
                    continue
                result = traverse(child)
                if result:
                    return result
            return None
        
        return traverse(declarator_node)

    def _extract_item_data(self, node, lines, code, language, parent_class=None):
        """提取单个项目（函数/类/函数声明）的数据"""
        # 1. 处理函数声明（针对头文件中的 declaration 节点）
        if node.type == "declaration" and language in ['c', 'cpp']:
            # 从声明节点中查找函数名（简化逻辑，可根据实际语法调整）
            func_declarator = None
            for child in node.children:
                if child.type == "function_declarator":
                    func_declarator = child
                    break
            if not func_declarator:
                return None
            
            # 提取函数名
            # name_node = self._find_identifier(func_declarator)
            name_node = self._find_function_name_in_declarator(func_declarator)  # 新增专用方法
            if not name_node:
                return None
            item_name = self._extract_code_by_position(lines, name_node.start_point, name_node.end_point).strip()
            
            # 提取声明的位置信息
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            return {
                "name": item_name,
                "original_name": item_name,
                "start_line": start_line,
                "end_line": end_line,
                "comments": "",  # 声明可能没有注释，可按需提取
                "code": self._extract_code_by_position(lines, node.start_point, node.end_point).strip(),
                "calls": [],  # 声明没有函数体，无调用
                "type": "function_declaration",  # 标记为声明
                "parent_class": parent_class
            }
        
        # 2. 原有处理函数定义和类的逻辑（保持不变）
        if node.type in ["function_definition", "method_definition", "class_definition"]:
            # （此处省略原有代码，保持不变）
            # ...
            """提取单个项目（函数/类）的数据"""
            # 获取名称
            name_node = self._find_identifier(node)
            if not name_node:
                return None
            
            item_name = self._extract_code_by_position(lines, name_node.start_point, name_node.end_point).strip()
            item_text = self._extract_code_by_position(lines, node.start_point, node.end_point)
            
            # 获取注释
            comments, start_line = self._get_comments_above_with_tree_sitter(code, node.start_point[0], language)
            end_line = node.end_point[0] + 1
            
            # 获取函数调用
            called_funcs = self._get_called_functions_unified(node, lines, language)
            
            # 确定项目类型和完整名称
            item_type = "class" if node.type == "class_definition" else "function"
            full_name = f"{parent_class}.{item_name}" if parent_class else item_name
            
            return {
                "name": full_name,
                "original_name": item_name,
                "start_line": start_line,
                "end_line": end_line,
                "comments": comments,
                "code": item_text.strip(),
                "calls": called_funcs,
                "type": item_type,
                "parent_class": parent_class
            }
    
    def build_language_chunks(self, code: str, original_meta: Dict[str, Any], language: str) -> List[Document]:
        """构建特定语言文件的分块"""
        lines = code.splitlines(keepends=True)
        items, item_ranges = self.extract_functions_and_classes_unified(code, language)
        
        chunks = []
        
        # 处理非函数/类块
        non_function_docs = self._extract_non_function_blocks(lines, item_ranges,original_meta.copy())
        for i, block in enumerate(non_function_docs):
            block.meta_data.update({
                "chunk_type": "non_function",
                "chunk_id": f"non_func_{i}",
                "token_count": len(block.text.split()),
                "language": language
            })
        chunks.extend(non_function_docs)
        
        # 处理函数/类块
        for item in items:
            content = self._build_item_content(item)
            
            meta_data = original_meta.copy()
            meta_data.update({
                "chunk_type": item["type"],
                "function_name": item["name"],
                "original_name": item["original_name"],
                "start_line": item["start_line"],
                "end_line": item["end_line"],
                "calls": item["calls"],
                "comments": item["comments"],
                "chunk_id": f"{item['type']}_{item['name']}_{item['start_line']}",
                "token_count": len(content.split()),
                "language": language,
                "parent_class": item.get("parent_class")
            })
            chunks.append(Document(text=content, meta_data=meta_data))
        
        return chunks

    def _build_item_content(self, item):
        """构建项目内容字符串"""
        item_type = item["type"].capitalize()
        content_parts = [f"{item_type}: {item['name']}"]
        
        if item["comments"]:
            content_parts.append(f"comments: {item['comments']}")
        
        if item["code"]:
            content_parts.append(f"function_body: {item['code'].strip()}")
        
        if item["calls"]:
            content_parts.append(f"function_calls: {item['calls']}")
        
        content_parts.append(f"Start Line: {item['start_line']}, End Line: {item['end_line']}")
        
        return "  ".join(content_parts)

    # 保留的辅助方法
    def _extract_code_by_position(self, lines: List[str], start_point: tuple, end_point: tuple) -> str:
        """根据位置提取代码"""
        start_row, start_col = start_point
        end_row, end_col = end_point
        if start_row == end_row:
            return lines[start_row][start_col:end_col]
        else:
            extracted = lines[start_row][start_col:]
            for i in range(start_row + 1, end_row):
                extracted += lines[i]
            extracted += lines[end_row][:end_col]
            return extracted

    def _get_comments_above_with_tree_sitter(self, code: str, func_start_line: int, language: str) -> tuple[str, int]:
        """使用tree-sitter统一提取函数定义前的注释"""
        lines = code.splitlines(keepends=True)
        parser = self.parsers[language]
        tree = parser.parse(code.encode("utf8"))
        root_node = tree.root_node
        
        # 收集所有注释节点
        comment_nodes = []
        
        def collect_comments(node):
            if node.type == "comment":
                comment_nodes.append(node)
            for child in node.children:
                collect_comments(child)
        
        collect_comments(root_node)
        
        # 找到函数开始行之前的注释
        relevant_comments = []
        
        for comment_node in comment_nodes:
            comment_end_line = comment_node.end_point[0] + 1
            if comment_end_line < func_start_line + 1:
                comment_text = self._extract_code_by_position(lines, comment_node.start_point, comment_node.end_point)
                relevant_comments.append({
                    'text': comment_text,
                    'end_line': comment_end_line,
                    'start_line': comment_node.start_point[0] + 1
                })
        
        # 按行号排序并找到连续的注释块
        relevant_comments.sort(key=lambda x: x['start_line'])
        
        if not relevant_comments:
            return "", func_start_line
        
        # 从最后一个注释开始，向前找连续的注释块
        final_comments = []
        last_end_line = func_start_line
        
        for comment in reversed(relevant_comments):
            gap = last_end_line - comment['end_line']
            if gap <= 3:  # 允许最多2行空行的间隔
                final_comments.insert(0, comment['text'].strip())
                last_end_line = comment['start_line']
            else:
                break
        
        return ' '.join(final_comments).strip(), last_end_line

    def _get_called_functions_unified(self, node, lines: List[str], language: str) -> List[str]:
        """统一的函数调用提取方法"""
        called_funcs = []

        def walk(n):
            if language in ['c', 'cpp'] and n.type == "call_expression":
                for child in n.children:
                    if child.type == "identifier":
                        func_name = self._extract_code_by_position(lines, child.start_point, child.end_point).strip()
                        called_funcs.append(func_name)
                        break
            elif language == 'python' and n.type == "call":
                if n.children and n.children[0].type in ["identifier", "attribute"]:
                    func_name = self._extract_code_by_position(lines, n.children[0].start_point, n.children[0].end_point).strip()
                    called_funcs.append(func_name)
            
            for child in n.children:
                walk(child)

        walk(node)
        return list(set(called_funcs))

    def _find_identifier(self, node):
        """查找标识符节点"""
        if node.type == "identifier":
            return node
        for child in node.children:
            result = self._find_identifier(child)
            if result:
                return result
        return None

    def _extract_non_function_blocks(self, lines: List[str], function_ranges: List[tuple],meta_data: dict):
        """根据函数行范围提取非函数区域代码块"""
        total_lines = len(lines)
        line_flags = [True] * total_lines

        for start, end in function_ranges:
            for i in range(start - 1, end):
                if 0 <= i < total_lines:
                    line_flags[i] = False

        # 组合连续的非函数段
        blocks = []
        current_block = []

        for flag, line in zip(line_flags, lines):
            if flag:
                current_block.append(line)
            else:
                if current_block:
                    blocks.append(''.join(current_block).strip())
                    current_block = []
        
        if current_block:
            blocks.append(''.join(current_block).strip())

        # 过滤空块并创建Document对象
        non_empty_blocks = [block for block in blocks if block]
        if not non_empty_blocks:
            return []
        
        combined_text = "".join(non_empty_blocks)
        updated_meta_data = meta_data.copy()
        updated_meta_data.update({"split_method": "non_function"})
        return self.text_splitter.call([Document(text=combined_text, meta_data=updated_meta_data)])


def process_document_batch(batch_data):
    """批量处理文档函数"""
    try:
        documents = batch_data['documents']
        text_splitter = batch_data['text_splitter']
        configs = batch_data['configs']
        batch_id = batch_data['batch_id']
        
        # 在worker进程中创建分割器实例
        temp_splitter = MultiLanguageTextSplitter(text_splitter, configs)
        
        batch_chunks = []
        batch_func_calls = {}
        successful_count = 0
        
        for doc in documents:
            try:
                chunks, func_calls = temp_splitter._process_document_internal(doc)
                batch_chunks.extend(chunks)
                
                if func_calls:
                    file_path = doc.meta_data.get('file_path', f'unknown_{id(doc)}')
                    batch_func_calls[file_path] = func_calls
                
                successful_count += 1
                
            except Exception as e:
                logger.error(f"Batch {batch_id}: Document processing failed: {e}")
                continue
        
        return {
            'success': True,
            'chunks': batch_chunks,
            'func_calls': batch_func_calls,
            'batch_id': batch_id,
            'processed_count': successful_count,
            'total_count': len(documents)
        }
        
    except Exception as e:
        logger.error(f"Batch {batch_id} processing completely failed: {e}")
        return {
            'success': False,
            'chunks': [],
            'func_calls': {},
            'batch_id': batch_data.get('batch_id', -1),
            'error': str(e)
        }