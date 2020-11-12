import json, os, logging, re
from glob import glob
from tqdm import tqdm

filepaths = glob('/home/dmlab/Dropbox/DATA/Samsung_text/UNZIPPED/2020/ipg200107_5374.json')
out_corpus_filepath = '/home/dmlab/Dropbox/DATA/PyTorch_TextGCN/text_dataset/corpus/Patent.txt'
out_label_filepath = '/home/dmlab/Dropbox/DATA/PyTorch_TextGCN/text_dataset/Patent.txt'
test_ratio = 0.3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info('Start processing..')
cnt = 0
out_corpus_file = open(out_corpus_filepath, 'wt')
out_label_file = open(out_label_filepath, 'wt')
for filepath in tqdm(filepaths, desc='file processing'):
    with open(filepath, 'r') as input_file:
        data = json.load(input_file)
        for sentences in data['abstract'].values():
            text = ' '.join(sentences)
            text = re.sub('\s+', ' ', text)
            out_corpus_file.write(text + ' \n')
            cnt += 1
        for idx, multi_labels in enumerate(data['section_class_subclasses'].values()):
            if idx <= round(len(data['abstract'])*(1-test_ratio))-1: train_or_test = 'train'
            else: train_or_test = 'test'
            label = multi_labels[0]   # TEMP. For single label classification.
            text = '%d\t%s\t%s' % (idx, train_or_test, label)
            out_label_file.write(text + ' \n')
logger.info(f'Total number of processed rows: {cnt}.')

out_label_file.close()
out_corpus_file.close()
logger.info(f'Created {out_corpus_filepath}')
logger.info(f'Created {out_label_filepath}')