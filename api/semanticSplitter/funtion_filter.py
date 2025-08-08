import os
import json
import csv
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from adalflow.core.types import Document
from api.semanticSplitter.ParallelHybridTextSplitter import MultiLanguageTextSplitter

class FunctionFilterProcessor(MultiLanguageTextSplitter):
    """识别头文件函数和跨模块调用函数"""
    
    def __init__(self, text_splitter, configs: Dict[str, Any] = None, file_path: Optional[str] = None):
        super().__init__(text_splitter, configs, file_path)
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

    def _is_cross_module_call(self, caller_file: str, callee_name: str) -> bool:
        """判断是否为跨模块调用"""
        if not self.all_functions:
            return False
        # 检查被调用函数是否存在于其他文件中
        for file_path, functions in self.all_functions.items():
            if file_path != caller_file and any(f.get('name') == callee_name for f in functions):
                return True
        return False

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
                self.logger.info(f"临时文件已生成: {self.temp_file_path}")
            except Exception as e:
                self.logger.error(f"生成临时文件失败: {e}")

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
                self.logger.info(f"函数信息JSON文件已生成: {json_path}")
            except Exception as e:
                self.logger.error(f"生成函数JSON文件失败: {e}")

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
                self.logger.info(f"函数信息CSV文件已生成: {csv_path}")
            except Exception as e:
                self.logger.error(f"生成函数CSV文件失败: {e}")

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
                if chunk.meta_data.get('chunk_type') in ['function', 'class']
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

            # 识别跨模块调用
            for caller, callees in func_calls.items():
                for callee in callees:
                    if self._is_cross_module_call(file_path, callee):
                        with self._lock:
                            self.cross_module_calls.append({
                                'caller': caller,
                                'callee': callee,
                                'caller_file': file_path
                            })

        return chunks, func_calls

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """重写入口方法，确保处理完成后生成所有文件"""
        # 重置存储结构（避免多次调用时数据残留）
        self.header_functions = []
        self.cross_module_calls = []
        self.all_functions = {}

        # 调用父类的分割逻辑（支持并行/线程/顺序处理）
        result_chunks = super().split_documents(documents)

        # 所有文档处理完成后生成文件
        self._generate_temp_file()
        self._write_functions_to_json()  # 新增JSON文件生成
        self._write_functions_to_csv()   # 新增CSV文件生成
        return result_chunks