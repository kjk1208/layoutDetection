import torch
import torchvision
import numpy as np
import re
import os
from pandas import read_csv, DataFrame
from itertools import count
from tqdm import tqdm

def is_list_of_strings(x):
    if isinstance(x, list):
        return all(isinstance(item, str) for item in x)
    return False

class TreeNode:
    def __init__(self, cls, box):
        self.cls = cls
        self.box = torch.tensor(box)
        self.touch = False
        self.leaf = []

    def __str__(self):
        if self.leaf:
            leaf_strs = [l.__str__() for l in self.leaf]
            return f"[{self.cls}] {self.box.numpy().tolist()} \n -> {', '.join(leaf_strs)}"
        else:
            return f"[{self.cls}] {self.box.numpy().tolist()}"

    def getWH(self):
        self.width, self.height = self.box[2:] - self.box[:2]

def isContained(box1, box2, noise=5):
    complete = (box1[0] <= box2[0]) and \
                (box1[1] <= box2[1]) and \
                (box1[2] >= box2[2]) and \
                (box1[3] >= box2[3])
    if complete:
        return complete
        
    noised = (abs(box1[0] - box2[0]) < noise) and \
                (abs(box1[1] - box2[1]) < noise) and \
                (abs(box1[2] - box2[2]) < noise) and \
                (abs(box1[3] - box2[3]) < noise)
    return complete or noised


class LayoutPlanter:
    def __init__(self, strategy, dataset_info=None, ext_save_name=None, canvas_size=(513, 750)):
        self.strategy = strategy
        
        self.canvas_ratio = (canvas_size[0] / 513, canvas_size[1] / 750)
        self.canvas_size = canvas_size
        
        if dataset_info is not None:
            pt_path = os.path.join(dataset_info['design_intent_bbox_dir'], f"{dataset_info['dataset_name']}_preprocessed_dbs_{self.strategy['structure']}_{self.strategy['injection']}{('_'+ext_save_name) if ext_save_name is not None else ''}.pt")
            if os.path.exists(pt_path):
                self.readPreprocessedDataset(pt_path, dataset_info)
                print(f'`{dataset_info["dataset_name"]}` has been loaded from {pt_path}.')
            else:
                print(f'Start preprocessing for `{dataset_info["dataset_name"]}`.')
                self.preprocessDataset(dataset_info, pt_path)
                print('Preprocess has been finished.')
                
        if strategy['structure'] == 'plain':
            self.describer = self.getPlainLayoutDescription
            self.intepreter = self.interpretPlainLayoutDescription
        elif strategy['structure'] == 'hierarchical':
            self.describer = self.getHierarchicalLayoutDescription
            self.intepreter = self.interpretHierarchicalLayoutDescription
        else:
            raise NotImplementedError("'structure' strategy must be one of ['plain', 'hierarchical']")

    def preprocessDataset(self, dataset_info, save_path):
        db_train = torch.load(os.path.join(dataset_info['design_intent_bbox_dir'], "design_intent_bbox_train.pt"), weights_only=False)
        db_valid = torch.load(os.path.join(dataset_info['design_intent_bbox_dir'], "design_intent_bbox_valid.pt"), weights_only=False)
        db_test = torch.load(os.path.join(dataset_info['design_intent_bbox_dir'], "design_intent_bbox_test.pt"), weights_only=False)

        # filter by dataset name (skip filtering when dataset_name == 'all')
        if dataset_info['dataset_name'] == 'all':
            self.db_train = db_train
            self.db_valid = db_valid
            self.db_test = db_test
        else:
            self.db_train = list(filter(lambda x: x['dataset'] == dataset_info['dataset_name'], db_train))
            self.db_valid = list(filter(lambda x: x['dataset'] == dataset_info['dataset_name'], db_valid))
            self.db_test = list(filter(lambda x: x['dataset'] == dataset_info['dataset_name'], db_test))
        
        for i in range(len(self.db_train)):
            for j in range(len(self.db_train[i]['den_box'])):
                self.db_train[i]['den_box'][j] = (int(self.db_train[i]['den_box'][j][0] * self.canvas_ratio[0]),
                                                    int(self.db_train[i]['den_box'][j][1] * self.canvas_ratio[1]),
                                                    int(self.db_train[i]['den_box'][j][2] * self.canvas_ratio[0]),
                                                    int(self.db_train[i]['den_box'][j][3] * self.canvas_ratio[1]))
        
        for i in range(len(self.db_valid)):
            for j in range(len(self.db_valid[i]['den_box'])):
                self.db_valid[i]['den_box'][j] = (int(self.db_valid[i]['den_box'][j][0] * self.canvas_ratio[0]),
                                                    int(self.db_valid[i]['den_box'][j][1] * self.canvas_ratio[1]),
                                                    int(self.db_valid[i]['den_box'][j][2] * self.canvas_ratio[0]),
                                                    int(self.db_valid[i]['den_box'][j][3] * self.canvas_ratio[1]))
        
        for i in range(len(self.db_test)):
            for j in range(len(self.db_test[i]['den_box'])):
                self.db_test[i]['den_box'][j] = (int(self.db_test[i]['den_box'][j][0] * self.canvas_ratio[0]),
                                                    int(self.db_test[i]['den_box'][j][1] * self.canvas_ratio[1]),
                                                    int(self.db_test[i]['den_box'][j][2] * self.canvas_ratio[0]),
                                                    int(self.db_test[i]['den_box'][j][3] * self.canvas_ratio[1]))

        df = read_csv(os.path.join(dataset_info['annotation_dir'], 'train.csv'))
        groups = df.groupby(df.poster_path)

        print("Train split:", end=" ")
        for i in tqdm(range(len(self.db_train))):
            query = self.db_train[i]['poster_path']
            subdf = groups.get_group(query)
            self.db_train[i]['layout'] = {
                'cls_elem': subdf.cls_elem.to_list(),
                'box_elem': [eval(b) for b in subdf.box_elem]
            }
            for j in range(len(self.db_train[i]['layout']['box_elem'])):
                self.db_train[i]['layout']['box_elem'][j] = [int(self.db_train[i]['layout']['box_elem'][j][0] * self.canvas_ratio[0]),
                                                            int(self.db_train[i]['layout']['box_elem'][j][1] * self.canvas_ratio[1]),
                                                            int(self.db_train[i]['layout']['box_elem'][j][2] * self.canvas_ratio[0]),
                                                            int(self.db_train[i]['layout']['box_elem'][j][3] * self.canvas_ratio[1])]
            subdf = DataFrame(self.db_train[i]['layout'])
            for k in range(len(subdf)):
                subdf.loc[k, 'box_elem'] = str(subdf.loc[k, 'box_elem'])
            self.db_train[i]['layout_description'] = self.getLayoutDescription(subdf, self.strategy, dataset_info['label_info'], self.db_train[i])
        
        print("Annotated test split:", end=" ")
        for i in tqdm(range(len(self.db_valid))):
            query = self.db_valid[i]['poster_path']
            subdf = groups.get_group(query)
            self.db_valid[i]['layout'] = {
                'cls_elem': subdf.cls_elem.to_list(),
                'box_elem': [eval(b) for b in subdf.box_elem]
            }
            for j in range(len(self.db_valid[i]['layout']['box_elem'])):
                self.db_valid[i]['layout']['box_elem'][j] = [int(self.db_valid[i]['layout']['box_elem'][j][0] * self.canvas_ratio[0]),
                                                            int(self.db_valid[i]['layout']['box_elem'][j][1] * self.canvas_ratio[1]),
                                                            int(self.db_valid[i]['layout']['box_elem'][j][2] * self.canvas_ratio[0]),
                                                            int(self.db_valid[i]['layout']['box_elem'][j][3] * self.canvas_ratio[1])]
            subdf = DataFrame(self.db_valid[i]['layout'])
            for k in range(len(subdf)):
                subdf.loc[k, 'box_elem'] = str(subdf.loc[k, 'box_elem'])
            self.db_valid[i]['layout_description'] = self.getLayoutDescription(subdf, self.strategy, dataset_info['label_info'], self.db_valid[i])

        print(f"Length of Train/Annotated test/Unannotated test: {len(self.db_train)}/{len(self.db_valid)}/{len(self.db_test)}")

        # save
        preprocessed_dbs_dict = {
            "db_train": self.db_train,
            "db_valid": self.db_valid,
            "db_test": self.db_test,
            "dataset_info": dataset_info,
            "strategy": self.strategy
        }
        
        torch.save(preprocessed_dbs_dict, save_path)
        print(f"The preprocessed pt file is saved at {save_path}")

    def readPreprocessedDataset(self, path, dataset_info):
        preprocessed_dbs_dict = torch.load(path, weights_only=False)
        # avoid mistakenly changing pt name outside
        assert preprocessed_dbs_dict['dataset_info'] == dataset_info, f"preprocessed_dbs_dict['dataset_info'] is not identical with the provided dataset_info: {preprocessed_dbs_dict['dataset_info']} vs {dataset_info}."

        self.db_train = preprocessed_dbs_dict["db_train"]
        self.db_valid = preprocessed_dbs_dict["db_valid"]
        self.db_test = preprocessed_dbs_dict["db_test"]

    def getLayoutDescription(self, subdf, strategy, label_info, design_intent_dict=None):
        if strategy['structure'] == 'plain':
            describer = self.getPlainLayoutDescription
        elif strategy['structure'] == 'hierarchical':
            describer = self.getHierarchicalLayoutDescription
        else:
            raise NotImplementedError("'structure' strategy must be one of ['plain', 'hierarchical']")

        if strategy['injection'] == 'none':
            return describer(subdf, label_info)
        elif strategy['injection'] == 'top':
            return describer(subdf, label_info, design_intent_dict, polygons=True)
        elif strategy['injection'] == 'pulse':
            return describer(subdf, label_info, design_intent_dict)
        elif strategy['injection'] == 'pulse_wh':
            return describer(subdf, label_info, design_intent_dict, wh=True)
        else:
            raise NotImplementedError("'injection' strategy must be one of ['none', 'top', 'pulse']")

    def checkSubdf(self, subdf):
        subdf = subdf[~(subdf['box_elem'] == '[0, 0, 0, 0]')]
        return subdf
        
    def getHierarchy(self, subdf, preview=False):
        clses = subdf.cls_elem.astype(int).to_list()
        boxes = [eval(b) for b in subdf.box_elem]
        # remove background/invalid class (0) if present
        if 0 in clses:
            keep = [i for i, c in enumerate(clses) if c != 0]
            clses = [clses[i] for i in keep]
            boxes = [boxes[i] for i in keep]
        nodes = [TreeNode(c, b) for c, b in zip(clses, boxes)]
        if len(nodes) == 0:
            return []
        nodes = sorted(nodes, key=lambda x: (-x.cls, x.box[0], x.box[1], -x.box[2], -x.box[3]))
        clses = [node.cls for node in nodes]
        boxes = torch.stack([node.box for node in nodes])
        n = len(nodes)
        ious = torchvision.ops.box_iou(boxes, boxes)
        for n_i in range(n - 1, -1, -1):
            for n_j in range(n - 1, n_i, -1):
                if nodes[n_j].touch:
                    continue
                if ious[n_i, n_j] > 0:
                    if isContained(boxes[n_i], boxes[n_j], noise=20):
                        if clses[n_i] == 3:
                            nodes[n_i].leaf.append(nodes[n_j])
                            nodes[n_j].touch = True
                        else:
                            nodes[n_j].leaf.append(nodes[n_i])
                            nodes[n_i].touch = True
                    # else:
                    #     print(ious[n_i, n_j])
        
        hierarchy = []
        for n_i in range(n):
            if not nodes[n_i].touch:
                hierarchy.append(nodes[n_i])
        if preview:
            for h in hierarchy:
                print(h)
        return hierarchy
    
    def traceHierarchy(self, h, label_info, anchor=(0, 0), index=None, indent="\t"):
        get = ""
        h.getWH() 
        x = max(0, h.box[0] - anchor[0])
        y = max(0, h.box[1] - anchor[1])
        if h.cls == 3:
            get += f'{indent}<svg x="{x}" y="{y}">\n'
            get += f'{indent}\t<rect id="{label_info[h.cls]["type"]}_{next(index)}" x="0" y="0" width="{h.width}" height="{h.height}" />\n'
            for l in h.leaf:
                get += self.traceHierarchy(l, label_info, anchor=(h.box[0], h.box[1]), index=index, indent=indent+'\t')
            get += f'{indent}</svg>\n'
        else:
            get += f'{indent}<rect id="{label_info[h.cls]["type"]}_{next(index)}" x="{x}" y="{y}" width="{h.width}" height="{h.height}" />\n'
        return get
    
    def getNestedSVG(self, hierarchy, label_info, index, indent="\t", injection=""):
        svg = f'<svg width="{self.canvas_size[0]}" height="{self.canvas_size[1]}" xmlns="http://www.w3.org/2000/svg">\n'
        svg += injection
        svg += f'{indent}<rect id="canvas_0" x="0" y="0" width="{self.canvas_size[0]}" height="{self.canvas_size[1]}" />\n'
        
        for rootH in hierarchy:
            svg += self.traceHierarchy(rootH, label_info, index=index)
        svg += '</svg>\n'
            
        return svg
    
    def getPlainSVG(self, subdf, label_info, index, indent="\t", injection=""):
        clses = subdf.cls_elem.astype(int).to_list()
        boxes = np.array([eval(b) for b in subdf.box_elem])
        # remove class 0 if present
        if any(c == 0 for c in clses):
            keep = [i for i, c in enumerate(clses) if c != 0]
            clses = [clses[i] for i in keep]
            boxes = boxes[keep]
        boxes[:, 2] -= boxes[:, 0]
        boxes[:, 3] -= boxes[:, 1]
    
        svg = f'<svg width="{self.canvas_size[0]}" height="{self.canvas_size[1]}" xmlns="http://www.w3.org/2000/svg">\n'
        svg += injection
        svg += f'{indent}<rect id="canvas_0" x="0" y="0" width="{self.canvas_size[0]}" height="{self.canvas_size[1]}" />\n'
        
        for cls, box in zip(clses, boxes):
            svg += f'{indent}<rect id="{label_info[cls]["type"]}_{next(index)}" x="{box[0]}" y="{box[1]}" width="{box[2]}" height="{box[3]}" />\n'
        svg += '</svg>\n'
        
        return svg

    def getSVGHead(self, svg, inject_area=None):
        head = f"This svg uses canvas_0 of size {self.canvas_size} "
        if inject_area is not None:
            inject_area = [str(ia) for ia in inject_area]
            head += "with available areas "
            head += ", ".join(inject_area)
            head += " "
        head += "to allocate { "
        head += ", ".join(re.findall('<rect id="(.*?)"', svg)[1:])
        head += ' }.\n'
        return head

    def getAvailableBbox(self, design_intent_dict):
        injection = ''
        boxes = design_intent_dict['den_box']
        for e_i, box in enumerate(boxes):
            injection += f'\t<polygon id="available_area" points="{box[0]},{box[1]} {box[2]},{box[1]} {box[2]},{box[3]} {box[0]},{box[3]}" />\n'
            
        return injection, boxes

    # call outside
    
    def getHierarchicalLayoutDescription(self, subdf, label_info, design_intent_dict=None, polygons=None, wh=False):
        index = count(start=1)
        
        hierarchy = self.getHierarchy(subdf)
        
        if design_intent_dict is None:
            injection, inject_area = "", None
        elif polygons is None:
            if wh:
                den_box = torch.tensor(design_intent_dict['den_box'])
                if den_box.shape[0] > 0:
                    den_box[:, 2] -= den_box[:, 0]
                    den_box[:, 3] -= den_box[:, 1]
                injection, inject_area = "", [tuple(b) for b in den_box.numpy()]
            else:
                injection, inject_area = "", design_intent_dict['den_box']
        else:
            injection, inject_area = self.getAvailableBbox(design_intent_dict)
            
        if len(hierarchy) == 0:
            return self.getSVGHead('<svg></svg>'), '<svg></svg>'
        svg = self.getNestedSVG(hierarchy, label_info, index, injection=injection)
        head = self.getSVGHead(svg, inject_area)
        
        return head, svg

    def getPlainLayoutDescription(self, subdf, label_info, design_intent_dict=None, polygons=None, wh=False):
        index = count(start=1)
        
        if design_intent_dict is None:
            injection, inject_area = "", None
        elif polygons is None:
            if wh:
                den_box = torch.tensor(design_intent_dict['den_box'])
                if den_box.shape[0] > 0:
                    den_box[:, 2] -= den_box[:, 0]
                    den_box[:, 3] -= den_box[:, 1]
                injection, inject_area = "", [tuple(b) for b in den_box.numpy()]
            else:
                injection, inject_area = "", design_intent_dict['den_box']
        else:
            injection, inject_area = self.getAvailableBbox(design_intent_dict)
            
        svg = self.getPlainSVG(subdf, label_info, index, injection=injection)
        head = self.getSVGHead(svg, inject_area)
        
        return head, svg

    def getSVGPrompt(self, labels, label_info, design_intent_dict=None):            
        prompt = f'Final: This svg uses canvas_0 of size {self.canvas_size} '
        if design_intent_dict is not None:
            injection, inject_area = self.getAvailableBbox(design_intent_dict)
            inject_area = [str(ia) for ia in inject_area]
            prompt += "with available areas "
            prompt += ", ".join(inject_area)
            prompt += " "
            
        prompt += 'to allocate { '
        if is_list_of_strings(labels):
            if 'canvas' in labels:
                labels.remove('canvas')
            prompt += ', '.join([f"{label}_{e_i + 1}" for e_i, label in enumerate(labels)])
        else:
            prompt += ', '.join([f"{label_info[label]['type']}_{e_i + 1}" for e_i, label in enumerate(labels)])
        prompt += ' }.\n'

        return prompt
        
    def cvtOffsetInt(self, found):
        for i, current in enumerate(found):
            offset_x, offset_y = current[:2]
            if offset_x and offset_y:
                offset_y = offset_y.split()[0]
                offset_x = max(0, int(float(offset_x.replace('"', ''))))
                offset_y = max(0, int(float(offset_y.replace('"', ''))))
            else:
                offset_x, offset_y = 0, 0
            found[i] = (offset_x, offset_y, *found[i][2:])
        return found
    
    def splitHierarchicalLayoutDescription(self, response):
        pattern_start = re.compile(r'<svg.*?>')
        pattern_end = re.compile(r'</svg>')
        pattern_combination = re.compile(pattern_start.pattern + '|' + pattern_end.pattern)
        
        segments = []
        stack = []
        i = 0
        
        for match in pattern_combination.finditer(response):
            if match.group() == '</svg>':
                if not stack:
                    break
                start = stack.pop()
                if not stack:
                    segments.append(response[start:match.end()])
            else:
                if not stack:
                    segments.append(response[i:match.start()])
                stack.append(match.start())
            i = match.end() + 1
                
        if response[i:]:
            segments.append(response[i:])
        return segments
    
    def interpretNestedSVG(self, segment, offsets, collects):
        sub_svg_pattern = '<svg x=(.+?) y=(.+?)>(.*)</svg>'
        rect_pattern = '<rect id="(.*?)_\d+" x=(.+?) y=(.+?) width=(.+?) height=(.+?) />'
        combined_pattern = re.compile(r'(?:' + sub_svg_pattern + r'|' + rect_pattern + r')+', re.DOTALL)
        found = re.findall(combined_pattern, segment)
        found = self.cvtOffsetInt(found)
        
        for f in found:
            if f[2]:
                offset_x = f[0] + offsets[0]
                offset_y = f[1] + offsets[1]
                self.interpretNestedSVG(f[2], (offset_x, offset_y), collects)
            else:
                rect = f[3:]
                offset_x = int(float(rect[1].replace('"', ''))) + offsets[0]
                offset_y = int(float(rect[2].replace('"', ''))) + offsets[1]
                rect = [rect[0], max(0, offset_x), max(0, offset_y)] + \
                    [max(0, int(float(r.replace('"', '')))) for r in rect[3:]]
                if rect not in collects:
                    collects.append(rect)
    
    def interpretHierarchicalLayoutDescription(self, response):
        svg_pattern = re.compile('<svg(.*)</svg>', re.DOTALL)
        svg = re.findall(svg_pattern, response)[0]
        segments = self.splitHierarchicalLayoutDescription(svg)
        
        rects = []
        for segment in segments:
            self.interpretNestedSVG(segment, (0, 0), rects)
        
        cls_elem = []
        box_elem = []
        for rect in rects:
            cls_elem.append(rect[0])
            box_elem.append(rect[1:])
                
        return {"cls_elem": cls_elem, "box_elem": box_elem}
        
    def old_interpretHierarchicalLayoutDescription(self, response):
        svg_pattern = re.compile('<svg(.*)</svg>', re.DOTALL)
        svg = re.findall(svg_pattern, response)[0]

        sub_svg_pattern = '<svg x=(.+?) y=(.+?)>(.*?)</svg>'
        rect_pattern = '<rect id="(.*?)_\d+" x=(.+?) y=(.+?) width=(.+?) height=(.+?) />'
        combined_pattern = re.compile(r'(?:' + sub_svg_pattern + r'|' + rect_pattern + r')+', re.DOTALL)
        found = re.findall(combined_pattern, svg)
        found = self.cvtOffsetInt(found)
        rects = []

        while found:
            current = found.pop(0)
            offset_x, offset_y = current[:2]
            
            if current[2]:
                sub_found = re.findall(combined_pattern, current[2])
                if sub_found:
                    sub_found = self.cvtOffsetInt(sub_found)
                for sf_i, sf in enumerate(sub_found):
                    sub_found[sf_i] = (offset_x + sub_found[sf_i][0], offset_y + sub_found[sf_i][1], *sub_found[sf_i][2:])
                found = sub_found + found
            else:
                rect = current[3:]
                rect = [rect[0]] + [max(0, int(float(rect[1].replace('"', '')))) + offset_x] + \
                       [max(0, int(float(rect[2].replace('"', '')))) + offset_y] + \
                       [max(0, int(float(r.replace('"', '')))) for r in rect[3:]]
                if rect not in rects:
                    rects.append(rect)
        
        cls_elem = []
        box_elem = []
        for rect in rects:
            cls_elem.append(rect[0])
            box_elem.append(rect[1:])
                
        return {"cls_elem": cls_elem, "box_elem": box_elem} 

    def interpretPlainLayoutDescription(self, response):
        svg_pattern = re.compile('<svg.*</svg>', re.DOTALL)
        rect_pattern = re.compile('<rect id="(.*?)_\d+" x=(.+?) y=(.+?) width=(.+?) height=(.+?) />')
        
        svg = re.findall(svg_pattern, response)[0]
        rects = re.findall(rect_pattern, svg)
        
        # remove duplicates but keep order
        new_rects = []
        for rect in rects:
            if rect not in new_rects:
                new_rects.append(rect)
        rects = new_rects
        
        cls_elem = []
        box_elem = []
        for rect in rects:
            if rect[0] == '0':
                continue
            cls_elem.append(rect[0])
            box_elem.append([max(0, int(float(r.replace('"', '')))) for r in rect[1:]])

        return {"cls_elem": cls_elem, "box_elem": box_elem}
    
if __name__ == '__main__':
    canvas_size = (513, 750) # (102, 150)
    ext_save_name = None # 'small'
    structure = ['hierarchical', 'plain']
    injection = ['none', 'top', 'pulse', 'pulse_wh']
    for s in structure:
        for i in injection:
            strategy = {
                'structure': s,
                'injection': i
            }
            
            pku = {
                'dataset_name': 'pku',
                'design_intent_bbox_dir': '/home/xuxiaoyuan/transPosterO/divs_split/extext_pku_128_1e-06_none/result/epoch100/',
                'annotation_dir': '/home/xuxiaoyuan/calg_dataset/pku/annotation/',
                'label_info': {
                    1: {'type': 'text', 'color': 'green'}, 
                    2: {'type': 'logo', 'color': 'red'},
                    3: {'type': 'underlay', 'color': 'orange'}
                }
            }
            
            pku_layout_planter = LayoutPlanter(strategy, dataset_info=pku, ext_save_name=ext_save_name, canvas_size=canvas_size)
            
            # cgl = {
            #     'dataset_name': 'cgl',
            #     'design_intent_bbox_dir': '/home/xuxiaoyuan/PosterO/divs_split/cgl_128_1e-06_none/result/epoch35/',
            #     'annotation_dir': '/home/xuxiaoyuan/calg_dataset/cgl/annotation/',
            #     'label_info': {
            #         1: {'type': 'logo', 'color': 'red'},
            #         2: {'type': 'text', 'color': 'green'},
            #         3: {'type': 'underlay', 'color': 'orange'},
            #         4: {'type': 'embellishment', 'color': 'blue'}
            #     }
            # }
            
            # cgl_layout_planter = LayoutPlanter(strategy, dataset_info=cgl, ext_save_name=ext_save_name, canvas_size=canvas_size)