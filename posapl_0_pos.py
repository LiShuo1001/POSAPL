import os
import re
import json
import spacy
import sng_parser
from tqdm import tqdm
from alisuretool.Tools import Tools
from torch.utils.data import Dataset


class SceneGraphDataset(Dataset):

    def __init__(self, ann_file, max_words=30):
        self.ann_file = ann_file
        self.ann = json.load(open(self.ann_file, 'r'))
        self.max_words = max_words
        pass

    @staticmethod
    def pre_caption(caption, max_words):
        caption = re.sub(r"([,.'!?\"()*#:;~])", '', caption.lower(),).replace(
            '-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(r"\s{2,}", ' ', caption)
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])
        if not len(caption):
            raise ValueError("pre_caption yields invalid text")
        return caption

    def get_scene_graph(self):
        count = 0
        parser = sng_parser.Parser("spacy", model="en_core_web_sm")
        for ann_one in tqdm(self.ann):
            caption = self.pre_caption(ann_one['caption'], self.max_words)
            graph = parser.parse(caption)

            entities, relations = graph["entities"], graph["relations"]

            entity_list = [entity["head"] for entity in entities]
            subject_relation_object_list = [" ".join([entities[relation_one["subject"]]["head"],
                                                      relation_one["relation"],
                                                      entities[relation_one["object"]]["span"]])
                                            for relation_one in relations]
            ann_one["entities"] = entity_list
            ann_one["relations"] = subject_relation_object_list
            if len(entity_list) == 0:
                count += 1
                print(caption)
            pass
        print(count)
        pass

    def get_scene_pos(self, result_ann_file):

        def deal_name(_pos_list):
            _pos_result = []
            pos_dict = {"DET": "determiner", "NOUN": "noun", "adposition": "noun", "ADV": "adverb",
                        "CCONJ": "conjunction", "PRON": "pronoun", "SCONJ": "conjunction", "PUNCT": "punctuation",
                        "ADJ": "adjective", "VERB": "verb", "ADP": "adposition", "INTJ": "interjection",
                        "AUX": "auxiliary", "NUM": "numeral", "PROPN": "pronoun", "PART": "participle",
                        "X": "x", "SYM": "x"}
            for pos in _pos_list:
                if pos in pos_dict:
                    # _pos_result.append(pos_dict[pos])
                    _pos_result.append(pos)
                else:
                    print(pos)
                pass
            return _pos_result

        spacy_nlp = spacy.load("en_core_web_sm")
        for ann_one in tqdm(self.ann):
            if isinstance(ann_one['caption'], list):
                ann_one["pos"] = []
                for caption_one in ann_one['caption']:
                    caption = self.pre_caption(caption_one, self.max_words)
                    ann_one["pos"].append(deal_name([token.pos_ for token in spacy_nlp(caption)]))
                    pass
            else:
                caption = self.pre_caption(ann_one['caption'], self.max_words)
                ann_one["pos"] = deal_name([token.pos_ for token in spacy_nlp(caption)])
                pass
            pass

        json.dump(self.ann, fp=open(result_ann_file, 'w'))
        pass

    pass


if __name__ == '__main__':
    for file_one in ["rsicd_test", "rsicd_train", "rsicd_val", "rsitmd_test", "rsitmd_train", "rsitmd_val"]:
        Tools.print(file_one)
        scene_graph = SceneGraphDataset(ann_file=f'data/finetune/{file_one}.json')
        scene_graph.get_scene_pos(result_ann_file=Tools.new_dir(f'data/finetune_pos/{file_one}.json'))
        pass
    pass

