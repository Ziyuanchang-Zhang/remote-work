import os
from adalflow.core.types import Document
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
REPO_PATH = "/hitai/zhangziyuanchang/mongoose"

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
