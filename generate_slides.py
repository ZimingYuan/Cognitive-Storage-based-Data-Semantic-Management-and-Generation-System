import os
import torch
import pathlib
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration
from unstructured.partition.auto import partition
import qlinear
from utils import Utils

def process_document(path, qwen_processor, qwen_model, max_char_count, max_sequence_count, max_token_count, language):
    print(f'Processing {str(path)}')
    elements = partition(filename=str(path))
    sequence = ''
    results = ''
    cnt = 0
    for i in elements:
        text = str(i).replace('- ', '')
        if len(sequence) + 1 + len(text) > max_char_count:
            messages = [
                {'role': 'system', 'content': f'You should summarize each paragraph sent by the user into organized, clear and easy-to-understand language, and organize the summarized content into a slide page within {max_token_count} words. You should use Markdown format. You should use {language}.'},
                {'role': 'user', 'content': sequence}
            ]
            inputs = qwen_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = qwen_processor([inputs], return_tensors="pt")
            generated_ids = qwen_model.generate(
                model_inputs.input_ids,
                max_new_tokens=max_token_count
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = qwen_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(response)
            results += response + '\n\n---\n\n'
            sequence = ''
            cnt += 1
            if cnt >= max_sequence_count:
                break
        sequence += ' ' + text
    if len(sequence) > 0:
        messages = [
            {'role': 'system', 'content': f'You should summarize each paragraph sent by the user into organized, clear and easy-to-understand language, and organize the summarized content into a slide page within {max_token_count} words. You should use Markdown format. You should use {language}.'},
            {'role': 'user', 'content': sequence}
        ]
        inputs = qwen_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = qwen_processor([inputs], return_tensors="pt")
        generated_ids = qwen_model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_token_count
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = qwen_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results += response + '\n\n---\n\n'
    return results

def process_image(path, blip_processor, blip_model):
    print(f'Processing {str(path)}')
    image = Image.open(str(path)).convert('RGB')
    prefix = 'An image of'
    inputs = blip_processor(image, prefix, return_tensors='pt')
    outputs = blip_model.generate(**inputs)
    text = blip_processor.decode(outputs[0], skip_special_tokens=True)
    print(text)
    return f'![]({path})\n\n{text}\n\n---\n\n'

def generate_slides(file_list, output_path, max_char_count, max_sequence_count, max_token_count, language, device):
    qwen_processor = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
    qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct", torch_dtype="float32")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    if device == 'aie':
        torch.ao.quantization.quantize_dynamic(blip_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        torch.ao.quantization.quantize_dynamic(qwen_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
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
                qwen_model.model, 
                torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                qlinear.QLinear, node_args, node_kwargs)
    markdown = ''
    for i in file_list:
        path = pathlib.Path(i)
        if path.suffix in ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.csv', '.html', '.xml', '.md', '.txt']:
            markdown += process_document(path, qwen_processor, qwen_model, max_char_count, max_sequence_count, max_token_count, language)
        if path.suffix in ['.png', '.jpg', '.bmp', '.webp']:
            markdown += process_image(path, blip_processor, blip_model)
    with open(output_path + '/generate_output.md', 'w', encoding='utf-8') as f:
        f.write(markdown)
    os.system(f'pandoc {output_path}/generate_output.md --slide-level=1 -o {output_path}/generate_output.pptx')

if __name__ == '__main__':
    generate_slides(['C:\\Users\\yzmyz\\testdir\\1908.10084v1.pdf'], 'C:\\Users\\yzmyz\\testdir', 1024, 32, 512, 'English', 'aie')
