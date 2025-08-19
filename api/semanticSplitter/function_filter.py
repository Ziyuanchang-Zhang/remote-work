import os
import json
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from adalflow.core.types import Document
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from tree_sitter import Language, Parser


logger = logging.getLogger(__name__)  # 新增：创建模块级logger



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


class MultiLanguageTextSplitter:
    """
    精简版多语言文本分割器：仅保留FunctionFilterProcessor所需功能
    """
    
    def __init__(self, text_splitter, configs: Dict[str, Any] = None, file_path: Optional[str] = None):
        self.text_splitter = text_splitter
        self.configs = configs or {}
        self.file_path = file_path
        self._init_parallel_config()
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
        self.batch_size = 50 if is_development else 500

    def _init_parsers(self):
        """初始化各种语言的解析器"""
        self.parsers = {}
        self.languages = {}
        
        # 仅保留必要的语言配置
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
        
        if self.configs.get('use_threading', False):
            return self._split_documents_threaded(documents)
        else:
            return self._split_documents_sequential(documents)

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
            return chunks, func_calls
        else:
            logger.info(f"Processing text file: {file_path}")
            chunks = self._process_text_document(doc)
            return chunks, {}

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
        target_node_types = {
            'c': ['function_definition', 'declaration'],
            'cpp': ['function_definition', 'method_definition', 'declaration'],
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
        """从function_declarator节点中精准提取函数名"""
        def traverse(n):
            if n.type == "identifier":
                parent_type = n.parent.type if n.parent else ""
                if parent_type in ["function_declarator", "pointer_declarator", 
                                "array_declarator", "postfix_expression"]:
                    return n
            
            for child in n.children:
                if child.type == "pointer_declarator":
                    result = traverse(child)
                    if result:
                        return result
                elif child.type in ["parameter_list", "identifier_list"]:
                    continue
                else:
                    result = traverse(child)
                    if result:
                        return result
            return None
        
        return traverse(declarator_node)
    
    def _extract_item_data(self, node, lines, code, language, parent_class=None):
        """提取单个项目（函数/类/函数声明）的数据"""
        # 处理函数声明
        if node.type == "declaration" and language in ['c', 'cpp']:
            def find_function_declarator(current_node):
                if current_node.type == "function_declarator":
                    return current_node
                for child in current_node.children:
                    found = find_function_declarator(child)
                    if found:
                        return found
                return None
            
            func_declarator = find_function_declarator(node)
            if not func_declarator:
                return None
            
            name_node = self._find_function_name_in_declarator(func_declarator)
            if not name_node:
                return None
            item_name = self._extract_code_by_position(lines, name_node.start_point, name_node.end_point).strip()
            
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            return {
                "name": item_name,
                "original_name": item_name,
                "start_line": start_line,
                "end_line": end_line,
                "comments": "",
                "code": self._extract_code_by_position(lines, node.start_point, node.end_point).strip(),
                "calls": [],
                "type": "function_declaration",
                "parent_class": parent_class
            }
        
        # 处理函数定义和类
        if node.type in ["function_definition", "method_definition", "class_definition"]:
            name_node = self._find_identifier(node)
            if not name_node:
                return None
            
            item_name = self._extract_code_by_position(lines, name_node.start_point, name_node.end_point).strip()
            item_text = self._extract_code_by_position(lines, node.start_point, node.end_point)
            
            comments, start_line = self._get_comments_above_with_tree_sitter(code, node.start_point[0], language)
            end_line = node.end_point[0] + 1
            
            called_funcs = self._get_called_functions_unified(node, lines, language)
            
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
        non_function_docs = self._extract_non_function_blocks(lines, item_ranges, original_meta.copy())
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
        """提取函数定义前的注释"""
        lines = code.splitlines(keepends=True)
        parser = self.parsers[language]
        tree = parser.parse(code.encode("utf8"))
        root_node = tree.root_node
        
        comment_nodes = []
        
        def collect_comments(node):
            if node.type == "comment":
                comment_nodes.append(node)
            for child in node.children:
                collect_comments(child)
        
        collect_comments(root_node)
        
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
        
        relevant_comments.sort(key=lambda x: x['start_line'])
        
        if not relevant_comments:
            return "", func_start_line
        
        final_comments = []
        last_end_line = func_start_line
        
        for comment in reversed(relevant_comments):
            gap = last_end_line - comment['end_line']
            if gap <= 3:
                final_comments.insert(0, comment['text'].strip())
                last_end_line = comment['start_line']
            else:
                break
        
        return ' '.join(final_comments).strip(), last_end_line

    def _get_called_functions_unified(self, node, lines: List[str], language: str) -> List[str]:
        """提取函数调用"""
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

    def _extract_non_function_blocks(self, lines: List[str], function_ranges: List[tuple], meta_data: dict):
        """提取非函数区域代码块"""
        total_lines = len(lines)
        line_flags = [True] * total_lines

        for start, end in function_ranges:
            for i in range(start - 1, end):
                if 0 <= i < total_lines:
                    line_flags[i] = False

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

        non_empty_blocks = [block for block in blocks if block]
        if not non_empty_blocks:
            return []
        
        combined_text = "".join(non_empty_blocks)
        updated_meta_data = meta_data.copy()
        updated_meta_data.update({"split_method": "non_function"})
        return self.text_splitter.call([Document(text=combined_text, meta_data=updated_meta_data)])

class FunctionFilterProcessor(MultiLanguageTextSplitter):
    """识别头文件函数和跨模块调用函数"""
    
    def __init__(self, text_splitter, configs: Dict[str, Any] = None, file_path: Optional[str] = None):
        # 强制使用多线程（避免多进程导致的子类实例问题）
        modified_configs = configs.copy() if configs else {}
        modified_configs['use_multiprocessing'] = False  # 禁用多进程
        modified_configs['use_threading'] = True         # 启用多线程
        super().__init__(text_splitter, modified_configs, file_path)
        self._lock = threading.Lock()
        self.header_functions = []  # 头文件中的函数
        self.cross_module_calls = []  # 跨模块调用
        self.all_functions = {}  # 所有文件的函数映射 {file_path: [function_details]}
        self.func_data_dir = os.path.join(os.getcwd(), "function_data")
        os.makedirs(self.func_data_dir, exist_ok=True)
        # 新增跨模块调用结果文件路径
        self.cross_calls_json_path = os.path.join(self.func_data_dir, "cross_module_calls.json")

    def _is_header_file(self, file_path: str) -> bool:
        """判断是否为头文件（支持C/C++头文件格式）"""
        header_extensions = ['.h', '.hpp', '.hxx', '.h++']
        return any(file_path.endswith(ext) for ext in header_extensions)
    
    def _get_file_base_name(self, file_path: str) -> str:
            """提取文件名主体（不含路径和扩展名）"""
            # 示例："path/to/fall.h" -> "fall"；"path/to/fall.c" -> "fall"
            file_name = os.path.basename(file_path)  # 先获取文件名（含扩展名）
            return os.path.splitext(file_name)[0]    # 移除扩展名

    def _is_header_function(self, callee_name: str) -> bool:
        """新增：判断被调用函数是否为头文件中的函数"""
        with self._lock:
            return any(hf['function_name'] == callee_name for hf in self.header_functions)

    def _is_cross_module_call(self, caller_file: str, callee_name: str) -> Optional[str]:
        if not self.all_functions:
            return None
        
        caller_base_name = self._get_file_base_name(caller_file)
        
        # 直接遍历所有头文件，寻找包含目标函数且主体名不同的文件
        for file_path, functions in self.all_functions.items():
            # 只处理头文件
            if not self._is_header_file(file_path):
                continue
            # 检查该头文件中是否包含目标函数
            if any(f.get('name') == callee_name for f in functions):
                callee_base_name = self._get_file_base_name(file_path)
                if caller_base_name != callee_base_name:
                    return file_path
        
        return None

    def _write_header_functions_to_json(self):
        """将头文件函数信息写入JSON文件（提前生成）"""
        with self._lock:
            json_data = {
                'header_functions': self.header_functions,
                'generated_at': datetime.now().isoformat()
            }
            json_path = os.path.join(self.func_data_dir, "header_functions.json")
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                logger.info(f"头文件函数JSON文件已生成: {json_path}")
            except Exception as e:
                logger.error(f"生成头文件函数JSON文件失败: {e}")
        return json_path
    
    def _write_functions_to_json(self):
        """将所有函数信息写入JSON文件"""
        with self._lock:
            json_data = {
                'all_functions': self.all_functions,
                # 'header_functions': self.header_functions,
                'generated_at': datetime.now().isoformat()
            }
            json_path = os.path.join(self.func_data_dir, "all_functions.json")
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                logger.info(f"函数信息JSON文件已生成: {json_path}")  # 修复：使用模块级logger
            except Exception as e:
                logger.error(f"生成函数JSON文件失败: {e}")  # 修复：使用模块级logger
    
    def _write_cross_module_calls_to_json(self):
        """将跨模块调用结果写入JSON文件"""
        with self._lock:
            json_data = {
                'cross_module_calls': self.cross_module_calls,
                'generated_at': datetime.now().isoformat()
            }
            try:
                with open(self.cross_calls_json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                logger.info(f"跨模块调用结果JSON文件已生成: {self.cross_calls_json_path}")
            except Exception as e:
                logger.error(f"生成跨模块调用JSON文件失败: {e}")
    
    def _process_document_internal(self, doc: Document) -> Tuple[List[Document], Dict[str, List[str]]]:
        """重写内部处理逻辑，添加头文件和跨模块调用识别"""
        # 调用父类方法获取基础分割结果
        chunks, func_calls = super()._process_document_internal(doc)
        file_path = doc.meta_data.get('file_path', 'unknown')
        language = self.detect_file_language(doc)

        # 收集当前文件的所有函数详情（仅处理C/C++/Python）
        if language in ['c', 'cpp', 'python']:
            current_functions = [
                {
                    'name': chunk.meta_data.get('function_name'),
                    'original_name': chunk.meta_data.get('original_name'),
                    'type': chunk.meta_data.get('chunk_type'),
                    'start_line': chunk.meta_data.get('start_line'),
                    'end_line': chunk.meta_data.get('end_line'),
                    'parent_class': chunk.meta_data.get('parent_class'),
                    'calls': chunk.meta_data.get('calls', [])
                }
                for chunk in chunks 
                if chunk.meta_data.get('chunk_type') in ['function', 'class', 'function_declaration']  # 新增声明类型
            ]

            current_functions = [f for f in current_functions if f['name']]  # 过滤空值

            # 线程安全地更新全局函数映射
            with self._lock:
                self.all_functions[file_path] = current_functions

            # 识别头文件中的函数
            if self._is_header_file(file_path):
                for func in current_functions:
                    with self._lock:
                        self.header_functions.append({
                            'function_name': func['name'],
                            'file_path': file_path,
                            'start_line': func['start_line'],
                            'end_line': func['end_line']
                        })

            # 识别跨模块调用（延迟到所有函数收集后再判断）
            for caller, callees in func_calls.items():
                for callee in callees:
                    # 临时存储调用关系，后续统一处理跨模块判断
                    with self._lock:
                        self.cross_module_calls.append({
                            'caller': caller,
                            'callee': callee,
                            'caller_file': file_path,
                            'pending': True  # 标记为待验证
                        })

        return chunks, func_calls

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """重写入口方法，确保处理完成后生成所有文件"""
        # 重置存储结构（避免多次调用时数据残留）
        self.header_functions = []
        self.cross_module_calls = []
        self.all_functions = {}

        # 调用父类的分割逻辑（已强制为多线程模式）
        result_chunks = super().split_documents(documents)

        # 先生成头文件函数JSON
        self._write_header_functions_to_json()

        # 处理跨模块调用的验证与去重（放宽标准：同一callee且同一callee_file视为重复）
        with self._lock:
            # 1. 先筛选有效的跨模块调用
            valid_calls = []
            for call in self.cross_module_calls:
                if call.get('pending'):
                    callee_file = self._is_cross_module_call(call['caller_file'], call['callee'])
                    if callee_file:
                        valid_calls.append({
                            'caller': call['caller'],
                            'callee': call['callee'],
                            'caller_file': call['caller_file'],
                            'callee_file': callee_file
                        })
            
            # 2. 去重逻辑：以(callee, callee_file)作为唯一标识
            call_counts = {}
            for call in valid_calls:
                # 唯一键：被调用函数 + 被调用函数所在文件
                call_key = (call['callee'], call['callee_file'])
                
                if call_key in call_counts:
                    # 重复调用则累加次数
                    call_counts[call_key]['times'] += 1
                    # 可选项：记录所有调用者信息（如果需要追溯来源）
                    call_counts[call_key]['callers'].append({
                        'caller': call['caller'],
                        'caller_file': call['caller_file']
                    })
                else:
                    call_counts[call_key] = {
                        'callee': call['callee'],
                        'callee_file': call['callee_file'],
                        'times': 1,
                        'callers': [{
                            'caller': call['caller'],
                            'caller_file': call['caller_file']
                        }]
                    }
            
            # 3. 转换为列表格式（如需精简可移除callers字段）
            self.cross_module_calls = list(call_counts.values())

        # 生成其他文件
        self._write_functions_to_json()
        # 生成跨模块调用结果JSON
        self._write_cross_module_calls_to_json()
        return result_chunks
