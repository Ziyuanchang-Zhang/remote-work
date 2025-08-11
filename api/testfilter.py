import sys
import os
import logging  # 新增：添加日志配置

# 配置日志输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 获取当前脚本所在目录的上级目录
root_path = os.path.dirname(os.path.abspath(__file__))  
sys.path.append(os.path.dirname(root_path))  # 调整路径层级，确保能导入api模块

from adalflow.core.types import Document
from api.semanticSplitter.ParallelHybridTextSplitter import MultiLanguageTextSplitter
from api.semanticSplitter.function_filter import FunctionFilterProcessor  # 修复导入路径
from adalflow.components.data_process import TextSplitter

def load_repo_documents(repo_path: str) -> list[Document]:
    """从代码仓路径加载所有代码文件并生成Document对象"""
    documents = []
    # 支持的代码文件扩展名
    supported_extensions = {
        'c': ['.c', '.h'],
        'cpp': ['.cpp', '.hpp', '.cc', '.cxx'],
        'python': ['.py', '.pyi'],
        'java': ['.java'],
        'javascript': ['.js', '.jsx'],
        'typescript': ['.ts', '.tsx']
    }
    
    # 构建扩展名到语言的映射
    ext_to_lang = {}
    for lang, exts in supported_extensions.items():
        for ext in exts:
            ext_to_lang[ext] = lang
    
    # 递归遍历目录
    for root, _, files in os.walk(repo_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in ext_to_lang:
                continue  # 跳过不支持的文件类型
            
            file_path = os.path.join(root, file)
            try:
                # 读取文件内容（尝试多种编码）
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                except Exception as e:
                    print(f"无法读取文件 {file_path}: {e}")
                    continue
            
            # 创建Document对象
            documents.append(Document(
                text=content,
                meta_data={
                    "file_path": file_path,
                    "relative_path": os.path.relpath(file_path, repo_path),
                    "type": ext.lstrip('.'),
                    "language": ext_to_lang[ext]
                }
            ))
    
    print(f"成功加载 {len(documents)} 个代码文件")
    return documents

# 配置代码仓路径（请替换为实际路径）
REPO_PATH = "/hitai/zhangziyuanchang/mongoose/src"

# 加载代码仓文档
test_docs = load_repo_documents(REPO_PATH)
if not test_docs:
    print("未找到任何可处理的代码文件，请检查路径和文件类型")
    exit(1)

# 初始化基础文本分割器
base_splitter = TextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    split_by="word"
)

# ------------------------------
# 验证ParallelHybridTextSplitter
# ------------------------------
print("\n===== 开始验证ParallelHybridTextSplitter =====")
splitter = MultiLanguageTextSplitter(
    text_splitter=base_splitter,
    configs={
        'max_workers': 4,
        'use_multiprocessing': True  # 基础分割器可使用多进程
    }
)

# 执行分割
split_chunks = splitter.split_documents(test_docs)

# 打印分割结果统计
chunk_type_count = {}
for chunk in split_chunks:
    chunk_type = chunk.meta_data.get('chunk_type', 'unknown')
    chunk_type_count[chunk_type] = chunk_type_count.get(chunk_type, 0) + 1

print(f"分割完成，共生成 {len(split_chunks)} 个块")
print("块类型分布：", chunk_type_count)

# 打印前5个块的简要信息
print("\n前5个块的信息：")
for i, chunk in enumerate(split_chunks[:5]):
    print(f"\nChunk {i+1}:")
    print(f"文件路径: {chunk.meta_data.get('file_path', 'unknown')[:50]}...")
    print(f"类型: {chunk.meta_data.get('chunk_type')}")
    print(f"语言: {chunk.meta_data.get('language')}")
    print(f"内容预览: {chunk.text[:100]}...")

# ------------------------------
# 验证FunctionFilterProcessor
# ------------------------------
print("\n===== 开始验证FunctionFilterProcessor =====")
processor = FunctionFilterProcessor(
    text_splitter=base_splitter,
    configs={
        'max_workers': 4  # 这里的多进程会被子类强制转为多线程
    }
)

# 处理文档
processor_chunks = processor.split_documents(test_docs)

# 打印扩展结果统计
print(f"\n处理完成，共生成 {len(processor_chunks)} 个块")
print(f"识别到 {len(processor.header_functions)} 个头文件函数")
print(f"识别到 {len(processor.cross_module_calls)} 个跨模块调用")

# 打印部分头文件函数
print("\n部分头文件函数：")
for func in list(processor.header_functions)[:5]:
    print(f"- {func['function_name']} (文件: {os.path.basename(func['file_path'])})")  # 修复key名称

# 打印部分跨模块调用
print("\n部分跨模块调用：")
for call in list(processor.cross_module_calls)[:5]:
    print(f"- {call['caller']} 调用了 {call['callee']} (来自: {os.path.basename(call['caller_file'])})")