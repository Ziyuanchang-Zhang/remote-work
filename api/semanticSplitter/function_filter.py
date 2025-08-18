import os
import json
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
