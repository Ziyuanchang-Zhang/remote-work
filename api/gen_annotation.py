import os
import json
from tree_sitter import Parser
# 导入C语言解析器模块
import tree_sitter_c as tsc

def initialize_parser():
    """初始化C语言解析器"""
    parser = Parser()
    parser.set_language(tsc.language())
    return parser

def get_possible_source_files(header_path):
    """根据头文件路径生成可能的源文件路径列表"""
    possible_paths = []
    if not header_path.endswith('.h'):
        return [header_path]  # 如果不是头文件，直接返回自身
    
    dir_name = os.path.dirname(header_path)
    base_name = os.path.basename(header_path)
    
    # 1. 同目录下，将.h替换为.c
    c_name = base_name.replace('.h', '.c')
    possible_paths.append(os.path.join(dir_name, c_name))
    
    # 2. 检查上级目录的src文件夹（如include/xxx.h 对应 src/xxx.c）
    if 'include' in dir_name.split(os.sep):
        src_dir = dir_name.replace('include', 'src')
        possible_paths.append(os.path.join(src_dir, c_name))
    
    # 3. 检查同目录下的src子文件夹
    src_subdir = os.path.join(dir_name, 'src')
    if os.path.exists(src_subdir):
        possible_paths.append(os.path.join(src_subdir, c_name))
    
    # 4. 尝试替换其他可能的目录结构（根据你的项目情况调整）
    if 'hstm/include' in dir_name:
        src_alt = dir_name.replace('hstm/include', 'hstm/src')
        possible_paths.append(os.path.join(src_alt, c_name))
    
    # 5. 去掉.h后的所有后缀
    base_no_ext = os.path.splitext(base_name)[0]
    possible_paths.append(os.path.join(dir_name, f"{base_no_ext}.c"))
    
    # 去重并过滤不存在的路径
    unique_paths = list(set(possible_paths))
    existing_paths = [p for p in unique_paths if os.path.exists(p) and os.path.isfile(p)]
    
    # 如果没有找到，返回原始路径作为最后的尝试
    if not existing_paths:
        existing_paths.append(header_path)
    
    return existing_paths

def extract_function_from_file(file_path, function_name, parser):
    """从单个文件中提取函数代码"""
    try:
        with open(file_path, 'rb') as f:
            code = f.read()
        
        tree = parser.parse(code)
        root_node = tree.root_node
        
        # 递归查找函数节点
        def find_function_node(node):
            if node.type == 'function_definition':
                declarator = node.child_by_field_name('declarator')
                if declarator:
                    # 处理指针声明
                    while declarator.type == 'pointer_declarator':
                        declarator = declarator.child_by_field_name('declarator')
                    
                    if declarator.type == 'function_declarator':
                        identifier = declarator.child_by_field_name('declarator')
                        if identifier and identifier.type == 'identifier' and identifier.text == function_name.encode():
                            return node
            # 递归检查子节点
            for child in node.children:
                result = find_function_node(child)
                if result:
                    return result
            return None
        
        function_node = find_function_node(root_node)
        if function_node:
            start = function_node.start_byte
            end = function_node.end_byte
            return code[start:end].decode('utf-8', errors='replace')
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    return None

def extract_function_code(file_path, function_name, parser):
    """提取函数代码，自动处理头文件和源文件的关联"""
    # 获取可能的源文件路径
    source_files = get_possible_source_files(file_path)
    
    # 依次尝试在每个可能的文件中查找函数
    for src_file in source_files:
        code = extract_function_from_file(src_file, function_name, parser)
        if code:
            return code, src_file
    
    # 如果所有文件都找不到
    print(f"警告: 在所有可能的文件中未找到函数 {function_name}")
    return None, None

def process_json_file(json_path, output_dir=None, top_n=10):
    """
    处理整个JSON文件，按调用次数排序并提取前N个函数代码
    :param json_path: JSON文件路径
    :param output_dir: 输出目录
    :param top_n: 要提取的调用次数最多的函数数量
    """
    # 初始化C语言解析器
    parser = initialize_parser()
    
    # 创建输出目录（如果指定）
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取所有cross_module_calls条目并按times排序
    cross_calls = data.get('cross_module_calls', [])
    
    # 按调用次数降序排序
    sorted_calls = sorted(cross_calls, key=lambda x: x.get('times', 0), reverse=True)
    
    # 取前N个
    top_calls = sorted_calls[:top_n]
    print(f"已选择调用次数最多的前 {len(top_calls)} 个函数（共 {len(sorted_calls)} 个）")
    
    # 存储提取的函数信息
    extracted_functions = []
    
    # 处理每个选中的条目
    for i, call_info in enumerate(top_calls, 1):
        callee = call_info.get('callee')
        callee_file = call_info.get('callee_file')
        times = call_info.get('times')
        
        print(f"\n处理第 {i}/{len(top_calls)} 个函数: {callee} (调用次数: {times})")
        
        # 提取被调用函数的代码
        callee_code, found_callee_path = extract_function_code(callee_file, callee, parser)
        
        # 处理调用者函数
        callers = []
        for caller in call_info.get('callers', []):
            caller_name = caller.get('caller')
            caller_file = caller.get('caller_file')
            
            print(f"  处理调用者函数: {caller_name}")
            caller_code, found_caller_path = extract_function_code(caller_file, caller_name, parser)
            
            callers.append({
                'name': caller_name,
                'file': caller_file,
                'found_file': found_caller_path,
                'code': caller_code
            })
            
            # 保存调用者函数代码
            if output_dir and caller_code:
                # 文件名格式: 排名_调用次数_函数类型_函数名.c
                caller_filename = f"{i:02d}_{times}_caller_{caller_name.replace(' ', '_')}.c"
                with open(os.path.join(output_dir, caller_filename), 'w', encoding='utf-8') as f:
                    f.write(caller_code)
        
        # 存储被调用函数信息
        extracted_functions.append({
            'rank': i,
            'callee': callee,
            'callee_file': callee_file,
            'callee_found_file': found_callee_path,
            'callee_code': callee_code,
            'call_count': times,
            'callers': callers
        })
        
        # 保存被调用函数代码
        if output_dir and callee_code:
            callee_filename = f"{i:02d}_{times}_callee_{callee.replace(' ', '_')}.c"
            with open(os.path.join(output_dir, callee_filename), 'w', encoding='utf-8') as f:
                f.write(callee_code)
    
    return extracted_functions

def generate_prompts(extracted_functions, prompt_template=None):
    """为提取的函数生成用于大模型的提示词"""
    prompts = []
    
    # 默认的提示词模板
    if not prompt_template:
        prompt_template = """请分析以下函数代码，并生成详细的注释：

函数名: {function_name}
所在文件: {file_path}
被调用次数: {call_count}

代码:
{function_code}

注释要求：
1. 说明函数的整体功能和用途
2. 解释每个参数的含义和可能的取值范围
3. 说明返回值的意义
4. 解释关键的实现逻辑和算法
5. 指出可能的边界情况和注意事项
6. 如果有特殊的设计考虑或性能优化，请说明
"""
    
    # 为每个被调用函数生成提示
    for func_info in extracted_functions:
        if func_info['callee_code']:
            prompt = prompt_template.format(
                function_name=func_info['callee'],
                file_path=func_info['callee_found_file'] or func_info['callee_file'],
                call_count=func_info['call_count'],
                function_code=func_info['callee_code']
            )
            prompts.append({
                'rank': func_info['rank'],
                'type': 'callee',
                'name': func_info['callee'],
                'call_count': func_info['call_count'],
                'prompt': prompt
            })
        
        # 为每个调用者函数生成提示
        for caller in func_info['callers']:
            if caller['code']:
                prompt = prompt_template.format(
                    function_name=caller['name'],
                    file_path=caller['found_file'] or caller['file'],
                    call_count=func_info['call_count'],
                    function_code=caller['code']
                )
                prompts.append({
                    'rank': func_info['rank'],
                    'type': 'caller',
                    'name': caller['name'],
                    'callee': func_info['callee'],
                    'call_count': func_info['call_count'],
                    'prompt': prompt
                })
    
    return prompts

# 使用示例
if __name__ == "__main__":
    # 初始化解析器（如果需要在主程序中单独使用）
    # parser = initialize_parser()
    
    # 替换为你的JSON文件路径
    json_file_path = "/hitai/zhangziyuanchang/CodeWiki/function_data/cross_module_calls.json"
    # 指定输出目录保存提取的函数代码
    output_directory = "top_functions"
    # 设置要提取的前N个函数
    top_n = 10  # 可以根据需要修改这个数字
    
    # # 处理JSON文件并提取前N个调用次数最多的函数
    # extracted = process_json_file(json_file_path, output_directory, top_n)
    
    # # 生成提示词
    # prompts = generate_prompts(extracted)
    
    # # 打印生成的提示词数量
    # print(f"\n成功生成 {len(prompts)} 个提示词")
