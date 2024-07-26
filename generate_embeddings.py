import glob
import faiss
import torch
import pathlib
import numpy as np
from PIL import Image
from FlagEmbedding import BGEM3FlagModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from unstructured.partition.auto import partition
import qlinear
from utils import Utils


def process_document(path, bge_model, max_char_count, max_sequence_count, max_token_count):
    print(f'Processing {str(path)}')
    text_lists = [' '.join(path.parts)]
    elements = partition(filename=str(path))
    sequence = ''
    for i in elements:
        text = str(i).replace('- ', '')
        if len(sequence) + 1 + len(text) > max_char_count:
            print(sequence)
            text_lists.append(sequence)
            sequence = ''
            if len(text_lists) >= max_sequence_count:
                break
        sequence += ' ' + text
    if len(sequence) > 0:
        text_lists.append(sequence)
    embeddings = bge_model.encode(text_lists, max_length=max_token_count)['dense_vecs']
    return embeddings

def process_image(path, bge_model, blip_processor, blip_model, max_token_count):
    print(f'Processing {str(path)}')
    text_lists = [' '.join(path.parts)]
    image = Image.open(str(path)).convert('RGB')
    prefix = 'an image of'
    inputs = blip_processor(image, prefix, return_tensors='pt')
    outputs = blip_model.generate(**inputs)
    text = blip_processor.decode(outputs[0], skip_special_tokens=True)
    print(text)
    text_lists.append(text)
    embeddings = bge_model.encode(text_lists, max_length=max_token_count)['dense_vecs']
    return embeddings

def generate_embeddings(process_path, max_char_count, max_sequence_count, max_token_count, device):
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    bge_model = BGEM3FlagModel('BAAI/bge-m3')
    if device == 'aie':
        torch.ao.quantization.quantize_dynamic(blip_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        torch.ao.quantization.quantize_dynamic(bge_model.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        node_args = ()
        quant_mode = 1
        node_kwargs = {
                'device': 'aie', 'quant_mode':'w8a8', 'profiler':False,
                'dtype':'float32', 'impl':'v1'}
        Utils.replace_node(
                blip_model, 
                torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                qlinear.QLinear, node_args, node_kwargs)
        Utils.replace_node(
                bge_model.model, 
                torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                qlinear.QLinear, node_args, node_kwargs)

    file_lists = []
    embeddings = []
    for file in glob.glob(process_path + '/**/*', recursive=True):
        path = pathlib.Path(file)
        if path.stem == 'generate_output': continue
        if path.suffix in ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.csv', '.html', '.xml', '.md', '.txt']:
            embedding = process_document(path, bge_model, max_char_count, max_sequence_count, max_token_count)
            embeddings.append(embedding)
            file_lists.append((str(path), embedding.shape[0]))
        if path.suffix in ['.png', '.jpg', '.bmp', '.webp']:
            embedding = process_image(path, bge_model, blip_processor, blip_model, max_token_count)
            embeddings.append(embedding)
            file_lists.append((str(path), embedding.shape[0]))
    if len(embeddings) > 0:
        embeddings = np.concatenate(embeddings, axis=0)
        with open(process_path + '/file_lists.index', 'w') as f:
            for i, j in file_lists:
                f.write(f'{i},{j}\n')
        index = faiss.IndexFlat(1024)
        index.add(embeddings)
        faiss.write_index(index, process_path + '/embeddings.index')

if __name__ == '__main__':
    generate_embeddings('C:\\Users\\yzmyz\\testdir', 1024, 32, 512, 'aie')

