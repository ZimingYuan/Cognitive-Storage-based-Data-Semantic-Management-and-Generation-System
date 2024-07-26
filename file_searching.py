import faiss
import torch
import bisect
from FlagEmbedding import BGEM3FlagModel
import qlinear
from utils import Utils

def file_searching(process_path, topk, device):
    bge_model = BGEM3FlagModel('BAAI/bge-m3')
    if device == 'aie':
        torch.ao.quantization.quantize_dynamic(bge_model.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        node_args = ()
        quant_mode = 1
        node_kwargs = {
                'device': 'aie', 'quant_mode':'w8a8', 'profiler':False,
                'dtype':'float32', 'impl':'v1'}
        Utils.replace_node(
                bge_model.model, 
                torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                qlinear.QLinear, node_args, node_kwargs)

    index = faiss.read_index(process_path + '/embeddings.index')
    file_lists = []
    cnt_lists = [0]
    with open(process_path + '/file_lists.index', 'r') as f:
        for i in f.readlines():
            x, y = i.strip().split(',')
            file_lists.append(x)
            cnt_lists.append(cnt_lists[-1] + int(y))
    embeddings = bge_model.encode(['personal introduction'])['dense_vecs']
    k = min(topk, cnt_lists[-1])
    _, res = index.search(embeddings, k)
    file_res = []
    for i in res[0]:
        x = bisect.bisect(cnt_lists, i)
        if x > 0:
            file_res.append(file_lists[x - 1])
    return list(set(file_res))

if __name__ == '__main__':
    print(file_searching('C:\\Users\\yzmyz\\testdir', 10, 'aie'))

