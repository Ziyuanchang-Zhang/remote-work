import os
import json
import csv
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from adalflow.core.types import Document
from api.semanticSplitter.ParallelHybridTextSplitter import MultiLanguageTextSplitter
import logging  

logger = logging.getLogger(__name__)  # 新增：创建模块级logger

class FunctionFilterProcessor(MultiLanguageTextSplitter):
    """识别头文件函数和跨模块调用函数"""
    
    def __init__(self, text_splitter, configs: Dict[str, Any] = None, file_path: Optional[str] = None):
        # 强制使用多线程（避免多进程导致的子类实例问题）
        modified_configs = configs.copy() if configs else {}
        modified_configs['use_multiprocessing'] = False  # 禁用多进程
        modified_configs['use_threading'] = True         # 启用多线程
        super().__init__(text_splitter, modified_configs, file_path)
        
        # 新增存储结构（使用线程锁确保并行安全）
        self._lock = threading.Lock()
        self.header_functions = []  # 头文件中的函数
        self.cross_module_calls = []  # 跨模块调用
        self.all_functions = {}  # 所有文件的函数映射 {file_path: [function_details]}
        self.temp_file_path = os.path.join(os.getcwd(), "temp_functions.json")
        # 新增函数数据文件路径
        self.func_data_dir = os.path.join(os.getcwd(), "function_data")
        os.makedirs(self.func_data_dir, exist_ok=True)

    def _is_header_file(self, file_path: str) -> bool:
        """判断是否为头文件（支持C/C++头文件格式）"""
        header_extensions = ['.h', '.hpp', '.hxx', '.h++']
        return any(file_path.endswith(ext) for ext in header_extensions)
    
    def _get_file_base_name(self, file_path: str) -> str:
            """提取文件名主体（不含路径和扩展名）"""
            # 示例："path/to/fall.h" -> "fall"；"path/to/fall.c" -> "fall"
            file_name = os.path.basename(file_path)  # 先获取文件名（含扩展名）
            return os.path.splitext(file_name)[0]    # 移除扩展名

    def _is_cross_module_call(self, caller_file: str, callee_name: str) -> Optional[str]:
        """判断是否为跨模块调用（基于文件名主体是否不同）"""
        if not self.all_functions:
            return None
        
        # 获取调用者文件的主体名称（不含扩展名）
        caller_base_name = self._get_file_base_name(caller_file)
        
        # 遍历所有文件，寻找被调用函数
        for file_path, functions in self.all_functions.items():
            # 被调用函数所在文件的主体名称
            callee_base_name = self._get_file_base_name(file_path)
            
            # 仅当文件名主体不同，且存在匹配的函数时，才视为跨模块调用
            if (caller_base_name != callee_base_name and 
                any(f.get('name') == callee_name for f in functions)):
                return file_path  # 返回被调用函数所在的文件路径
        
        return None  # 非跨模块调用

    def _generate_temp_file(self):
        """生成包含头文件函数和跨模块调用的临时文件"""
        with self._lock:
            temp_data = {
                'header_functions': self.header_functions,
                'cross_module_calls': self.cross_module_calls,
                'generated_at': datetime.now().isoformat()
            }
            try:
                with open(self.temp_file_path, 'w') as f:
                    json.dump(temp_data, f, indent=2)
                logger.info(f"临时文件已生成: {self.temp_file_path}")  # 修复：使用模块级logger
            except Exception as e:
                logger.error(f"生成临时文件失败: {e}")  # 修复：使用模块级logger

    def _write_functions_to_json(self):
        """将所有函数信息写入JSON文件"""
        with self._lock:
            json_data = {
                'all_functions': self.all_functions,
                'header_functions': self.header_functions,
                'generated_at': datetime.now().isoformat()
            }
            json_path = os.path.join(self.func_data_dir, "all_functions.json")
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                logger.info(f"函数信息JSON文件已生成: {json_path}")  # 修复：使用模块级logger
            except Exception as e:
                logger.error(f"生成函数JSON文件失败: {e}")  # 修复：使用模块级logger

    def _write_functions_to_csv(self):
        """将所有函数信息写入CSV文件"""
        with self._lock:
            csv_path = os.path.join(self.func_data_dir, "all_functions.csv")
            try:
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # 写入表头
                    writer.writerow([
                        'file_path', 'function_name', 'type', 'start_line', 
                        'end_line', 'parent_class', 'is_header_function'
                    ])
                    
                    # 写入所有函数数据
                    for file_path, functions in self.all_functions.items():
                        for func in functions:
                            is_header = any(
                                hf['function_name'] == func['name'] and hf['file_path'] == file_path
                                for hf in self.header_functions
                            )
                            writer.writerow([
                                file_path,
                                func['name'],
                                func['type'],
                                func['start_line'],
                                func['end_line'],
                                func.get('parent_class', ''),
                                'Yes' if is_header else 'No'
                            ])
                logger.info(f"函数信息CSV文件已生成: {csv_path}")  # 修复：使用模块级logger
            except Exception as e:
                logger.error(f"生成函数CSV文件失败: {e}")  # 修复：使用模块级logger

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

        # 处理跨模块调用的延迟验证（确保所有函数已收集）
        with self._lock:
            # 过滤并验证跨模块调用
            valid_calls = []
            for call in self.cross_module_calls:
                if call.get('pending'):
                    # 调用修改后的方法，获取被调用函数的文件路径
                    callee_file = self._is_cross_module_call(call['caller_file'], call['callee'])
                    if callee_file:
                        valid_calls.append({
                            'caller': call['caller'],
                            'callee': call['callee'],
                            'caller_file': call['caller_file'],
                            'callee_file': callee_file  # 添加被调用函数的文件路径
                        })
            self.cross_module_calls = valid_calls

        # 所有文档处理完成后生成文件
        self._generate_temp_file()
        self._write_functions_to_json()
        self._write_functions_to_csv()
        return result_chunks