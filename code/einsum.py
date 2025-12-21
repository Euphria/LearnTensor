import torch

def parse_equation(equation: str):
    """ 
    输入: equation (str), 例如 "ij,jk->ik"
    输出: 
        input_subs (list of str): 例如 ['ij', 'jk']
        output_sub (str): 例如 'ik'
    """
    # 去除空格
    equation = equation.replace(' ', '')
    input_subs, output_sub = equation.split('->')
    input_subs = input_subs.split(',')

    if __name__ == '__main__':
        print(f"\nInput subscripts: {input_subs}")
        print(f"Output subscript: {output_sub}")

    return input_subs, output_sub

def analyze_indices(input_subs: list[str], output_sub: str):
    """
    分析输入和输出的指标，找出求和指标和保留指标。
    
    输入:
        input_subs (list of str): 例如 ['ij', 'jk']
        output_sub (str): 例如 'ik'
    
    输出:
        all_indices (str): 所有出现的指标集合
        sum_indices (str): 需要求和的指标集合
    """
    all_indices = ''
    for sub in input_subs:
        all_indices += ''.join(sorted(set(sub) - set(all_indices)))
    sum_indices = ''.join(sorted(set(all_indices) - set(output_sub)))

    if __name__ == '__main__':
        print(f"\nAll indices: {all_indices}")
        print(f"Sum indices: {sum_indices}")
    return all_indices, sum_indices

def mapping_table(tensor: list[torch.Tensor], input_subs: list[str]):
    """
    创建一个映射表，将每个指标映射到对应的张量和维度。
    
    输入:
        tensor (list of torch.Tensor): 输入的张量列表
        input_subs (list of str): 所有出现的指标集合
    
    输出:
        mapping (dict): 指标到维度的映射表
    """
    mapping = {}
    if len(tensor) != len(input_subs):
        raise ValueError("Tensor和输入指标数量不匹配")
    for i in range(len(tensor)):
        for j in range(tensor[i].dim()):
            if input_subs[i][j] in mapping and mapping.get(input_subs[i][j]) != tensor[i].size(j):
                raise ValueError("张量维度不匹配")
            mapping[input_subs[i][j]] = tensor[i].size(j)

    if __name__ == '__main__':
        print(f"\nMapping: {mapping}")
    return mapping

def align_tensors(tensors: list[torch.Tensor], input_subs: list[str], all_indices: str, mapping: dict):
    """
    对齐张量，使其符合爱因斯坦求和的要求。
    
    输入:
        tensors (list of torch.Tensor): 输入的张量列表
        input_subs (list of str): 输入张量对应的指标列表
        all_indices (str): 所有出现的指标集合
        mapping (dict): 指标到维度的映射表
    
    输出:
        aligned_tensors (list of torch.Tensor): 对齐后的张量列表
    """
    aligned_tensors = []
    for i in range(len(tensors)):
        # 找出当前subs中字母在all_indices中的位置，然后对当前张量进行permute
        sub = input_subs[i]
        permute_idx = [idx for idx in all_indices if idx in sub]
        permute_order = [sub.find(idx) for idx in permute_idx]
        tensors[i] = tensors[i].permute(*permute_order)

        # 重塑张量形状，缺失的维度补1
        shape = []
        for idx in all_indices:
            if idx in input_subs[i]:
                shape.append(mapping[idx])
            else:
                shape.append(1)
        aligned_tensor = tensors[i].reshape(*shape)
        aligned_tensors.append(aligned_tensor)
    
    if __name__ == '__main__':
            print(f"\nAligned tensors: ")
            for i in range(len(aligned_tensors)):
                print(f"Tensor {i+1} shape: {aligned_tensors[i].size()}")
    return aligned_tensors

def einsum(equation: str, tensors: list[torch.Tensor]):
    """
    执行爱因斯坦求和操作。
    
    输入:
        equation (str): 输入的方程式
        tensors (list of torch.Tensor): 输入的张量列表
    
    输出:
        result (torch.Tensor): 结果张量
    """
    # 1. 解析方程式
    input_subs, output_sub = parse_equation(equation)
    all_indices, sum_indices = analyze_indices(input_subs, output_sub)
    mapping = mapping_table(tensors, input_subs)
    aligned_tensors = align_tensors(tensors, input_subs, all_indices, mapping)

    # 2. 执行逐元素乘法
    result = aligned_tensors[0]
    for i in range(1, len(aligned_tensors)):
        result = result * aligned_tensors[i]
    
    # 3. 求和收缩
    dims_to_sum = [all_indices.index(idx) for idx in sum_indices]
    if dims_to_sum:
        result = result.sum(dim=dims_to_sum)
    
    # 4. 调整输出形状
    remaining_indices = [idx for idx in all_indices if idx not in sum_indices]
    final_shape = [remaining_indices.index(idx) for idx in output_sub]
    result = result.permute(*final_shape)

    return result

if __name__ == '__main__':
    A = torch.tensor([[1, 1], [0, 1]])
    B = torch.tensor([[0, 0], [0, 1]])
    C = torch.tensor([[2, 1], [2, 1]])
    tensors = [A, B, C]
    equation = 'ij,jk,jl->ikl'
    print(f"A:{A}\ndimensions: {A.size()}")
    print(f"\nB:{B}\ndimensions: {B.size()}")
    print(f"\nC:{C}\ndimensions: {C.size()}")
    print(f"\nEquation: {equation}")
    
    # 模块测试
    # input_subs, output_sub = parse_equation(equation)
    # all_indices, sum_indices = analyze_indices(input_subs, output_sub)
    # mapping = mapping_tabel(tensors, input_subs)
    # aligned_tensors = align_tensors(tensors, input_subs, all_indices, mapping)

    result = einsum(equation, tensors)
    print(f"\nResult: {result}\nDimensions: {result.size()}")

    # # 专门测试涉及转置的逻辑
    # A = torch.tensor([[1, 2], [3, 4], [5, 6]]) # 形状 (3, 2)，下标定义为 ji (j=3, i=2)
    # print(f"A:{A}\ndimensions: {A.size()}")
    # equation = 'ji -> ij'
    
    # # 预期结果：[[1, 3, 5], [2, 4, 6]]
    # result = einsum(equation, [A])
    # print(f"\nResult: {result}\nDimensions: {result.size()}")
    
    # torch_result = torch.einsum(equation, A)
    # print(f"Match Torch: {torch.allclose(result, torch_result)}")
    # if not torch.allclose(result, torch_result):
    #     print("错误：数据对齐失败，请检查 align_tensors 是否使用了 permute。")

