# Cognitive Storage-based Data Semantic Management and Generation System

This system utilizes the Ryzen AI processor to accelerate the image-to-text model [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base), the embedding extraction model [BGE](https://huggingface.co/BAAI/bge-m3), and the large language model [Qwen](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) for slide generation, achieving a semantic-based intelligent file management and generation system on AI PCs with AMD Ryzen AI. This project participated in the PC AI track of the AMD Pervasive AI Developer Contest 2023.

## Setup

Step1: Install NPU Driver as [Ryzen AI Software documentation](https://ryzenai.docs.amd.com/en/latest/inst.html). Remember to restart the terminal after installation to update the PATH environment variable.

Step2: Using conda to setup the environment:

```
conda env create --file=env.yaml
conda activate cognitive-storage-system
```

Step3: Install QLinear:

```
setup.bat
pip install ops\cpp --force-reinstall
```

## Usage

```
python frontend.py
```

Every time when activate the conda environment, `setup.bat` should be executed first. The cognitive storage system support 3 operations:

* **Embedding generation.** User can select a directory , and then enter the maximum number of characters for each segment after the document file is segmented and the total number of segments to be processed. Then, all supported file types under this folder can be segmented and the embedding can be calculated. The supported file types include: pdf, docx, doc, pptx, ppt, xlsx, csv, html, xml, md, txt, png, jpg, bmp, webp. Note that for image files, the description text is first generated using the image-to-text model and then the embedding is generated. The generated index file `embeddings.index` and the file list `file_lists.index` are stored under the selected folder.
* **File searching.**  Users can select a directory for which the index has been generated, enter the prompt words and the maximum number of results, and the system will search for the files the user wants based on the semantics of the prompt words.
* **Slide generation.** For the list of files found through the search, users can input the maximum number of characters for each segment after the document file is segmented and the total number of segments to be processed. The system will automatically read and segment these files, and summarize the file contents into Markdown documents and slides in pptx format.
