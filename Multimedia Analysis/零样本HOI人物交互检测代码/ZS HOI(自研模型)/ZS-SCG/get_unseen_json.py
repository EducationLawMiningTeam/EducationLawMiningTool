import json


def generate_uo():
    anno_file = 'hicodet/instances_train2015.json'
    uo = [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                    126, 127, 128, 224, 225, 226, 227, 228, 229, 230, 231, 290, 291, 292, 293,
                    294, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 336, 337,
                    338, 339, 340, 341, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
                    429, 430, 431, 432, 433, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462,
                    463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 533, 534, 535, 536,
                    537, 558, 559, 560, 561, 595, 596, 597, 598, 599]
    with open(anno_file, 'r') as file:
        f = json.load(file)

    # _num_anno = num_anno

    _anno = f['annotation']
    _filenames = f['filenames']
    _image_sizes = f['size']
    _class_corr = f['correspondence']
    _empty_idx = f['empty']
    _objects = f['objects']
    _verbs = f['verbs']


    for i in range(len(_anno)):   # 去除未见类标签
        new_anno = {"boxes_h": [], "boxes_o": [], "hoi": [], "object": [], "verb": []}
        for index in range(len(_anno[i]["hoi"])):
            if _anno[i]["hoi"][index] not in uo:
                new_anno["boxes_h"].append(_anno[i]["boxes_h"][index])
                new_anno["boxes_o"].append(_anno[i]["boxes_o"][index])
                new_anno["object"].append(_anno[i]["object"][index])
                new_anno["verb"].append(_anno[i]["verb"][index])
                new_anno["hoi"].append(_anno[i]["hoi"][index])
        _anno[i] = new_anno
        if _anno[i]["hoi"] == []:
            _empty_idx.append(i)
    

    to_remove = []
    for corr in _class_corr:
        if corr[0] in uo:
            to_remove.append(corr)

    for corr in to_remove:
        _class_corr.remove(corr)
    

    output_file = 'hicodet/instances_train2015_uo.json'

    f['annotation'] = _anno
    f['empty'] = sorted(list(set(_empty_idx)))
    f['correspondence'] = _class_corr
    # 写入新的 JSON 文件
    with open(output_file, 'w') as file:
        json.dump(f, file, separators=(',', ':'))



def generate_uv():
    uv = [4, 6, 12, 15, 18, 25, 34, 38, 40, 49, 58, 60, 68, 69, 72, 73, 77, 82, 96, 97, 104, 113, 116, 118,
                    122, 129, 139, 147,
                    150, 153, 165, 166, 172, 175, 176, 181, 190, 202, 210, 212, 219, 227, 228, 233, 235, 243, 298, 313,
                    315, 320, 326, 336,
                    342, 345, 354, 372, 401, 404, 409, 431, 436, 459, 466, 470, 472, 479, 481, 488, 491, 494, 498, 504,
                    519, 523, 535, 536,
                    541, 544, 562, 565, 569, 572, 591, 595]
    anno_file = 'hicodet/instances_train2015.json'
    with open(anno_file, 'r') as file:
        f = json.load(file)

    idx = list(range(len(f['filenames'])))
    for empty_idx in f['empty']:
        idx.remove(empty_idx)

    # num_anno = [0 for _ in range(self.num_interation_cls)]
    # for anno in f['annotation']:
    #     for hoi in anno['hoi']:
    #         num_anno[hoi] += 1

    _idx = idx
    # _num_anno = num_anno

    _anno = f['annotation']
    _filenames = f['filenames']
    _image_sizes = f['size']
    _class_corr = f['correspondence']
    _empty_idx = f['empty']
    _objects = f['objects']
    _verbs = f['verbs']


    for i in range(len(_anno)):   # 去除未见类标签
        new_anno = {"boxes_h": [], "boxes_o": [], "hoi": [], "object": [], "verb": []}
        for index in range(len(_anno[i]["hoi"])):
            if _anno[i]["hoi"][index] not in uv:
                new_anno["boxes_h"].append(_anno[i]["boxes_h"][index])
                new_anno["boxes_o"].append(_anno[i]["boxes_o"][index])
                new_anno["object"].append(_anno[i]["object"][index])
                new_anno["verb"].append(_anno[i]["verb"][index])
                new_anno["hoi"].append(_anno[i]["hoi"][index])
        _anno[i] = new_anno
        if _anno[i]["hoi"] == []:
            _empty_idx.append(i)
    

    to_remove = []
    for corr in _class_corr:
        if corr[0] in uv:
            to_remove.append(corr)

    for corr in to_remove:
        _class_corr.remove(corr)
    
    output_file = 'hicodet/instances_train2015_uv.json'

    f['annotation'] = _anno
    f['empty'] = sorted(list(set(_empty_idx)))
    print(_class_corr)
    f['correspondence'] = _class_corr
    # 写入新的 JSON 文件
    with open(output_file, 'w') as file:
        json.dump(f, file, separators=(',', ':'))


def generate_ua():
    ua = [2, 10, 14, 20, 27, 33, 36, 42, 46, 57, 68, 81, 82, 86, 90, 92, 101, 103,
            109, 111, 116, 120, 121, 122, 123, 136, 137, 138, 140, 141, 149, 152, 155,
            160, 161, 170, 172, 174, 180, 188, 205, 208, 215, 222, 225, 236, 247, 260,
            265, 271, 273, 279, 283, 288, 295, 300, 301, 306, 310, 311, 315, 318, 319,
            337, 344, 352, 356, 363, 369, 373, 374, 419, 425, 427, 438, 453, 458, 461,
            464, 468, 471, 475, 480, 486, 489, 490, 496, 504, 506, 513, 516, 524, 528,
            533, 542, 555, 565, 576, 590, 597]
    anno_file = 'hicodet/instances_train2015.json'
    with open(anno_file, 'r') as file:
        f = json.load(file)

    idx = list(range(len(f['filenames'])))
    for empty_idx in f['empty']:
        idx.remove(empty_idx)

    # num_anno = [0 for _ in range(self.num_interation_cls)]
    # for anno in f['annotation']:
    #     for hoi in anno['hoi']:
    #         num_anno[hoi] += 1

    _idx = idx
    # _num_anno = num_anno

    _anno = f['annotation']
    _filenames = f['filenames']
    _image_sizes = f['size']
    _class_corr = f['correspondence']
    _empty_idx = f['empty']
    _objects = f['objects']
    _verbs = f['verbs']


    for i in range(len(_anno)):   # 去除未见类标签
        new_anno = {"boxes_h": [], "boxes_o": [], "hoi": [], "object": [], "verb": []}
        for index in range(len(_anno[i]["hoi"])):
            if _anno[i]["hoi"][index] not in ua:
                new_anno["boxes_h"].append(_anno[i]["boxes_h"][index])
                new_anno["boxes_o"].append(_anno[i]["boxes_o"][index])
                new_anno["object"].append(_anno[i]["object"][index])
                new_anno["verb"].append(_anno[i]["verb"][index])
                new_anno["hoi"].append(_anno[i]["hoi"][index])
        _anno[i] = new_anno
        if _anno[i]["hoi"] == []:
            _empty_idx.append(i)
    

    to_remove = []
    for corr in _class_corr:
        if corr[0] in ua:
            to_remove.append(corr)

    for corr in to_remove:
        _class_corr.remove(corr)
    
    output_file = 'hicodet/instances_train2015_ua.json'

    f['annotation'] = _anno
    f['empty'] = sorted(list(set(_empty_idx)))
    print(_class_corr)
    f['correspondence'] = _class_corr
    # 写入新的 JSON 文件
    with open(output_file, 'w') as file:
        json.dump(f, file, separators=(',', ':'))


def generate_rf_uc():
    rf_uc = [509, 279, 280, 402, 504, 286, 499, 498, 289, 485, 303, 311, 325, 439, 351, 358, 66, 427, 379, 418,
                   70, 416,
                   389, 90, 395, 76, 397, 84, 135, 262, 401, 592, 560, 586, 548, 593, 526, 181, 257, 539, 535, 260, 596,
                   345, 189,
                   205, 206, 429, 179, 350, 405, 522, 449, 261, 255, 546, 547, 44, 22, 334, 599, 239, 315, 317, 229,
                   158, 195,
                   238, 364, 222, 281, 149, 399, 83, 127, 254, 398, 403, 555, 552, 520, 531, 440, 436, 482, 274, 8, 188,
                   216, 597,
                   77, 407, 556, 469, 474, 107, 390, 410, 27, 381, 463, 99, 184, 100, 292, 517, 80, 333, 62, 354, 104,
                   55, 50,
                   198, 168, 391, 192, 595, 136, 581]
    anno_file = 'hicodet/instances_train2015.json'
    with open(anno_file, 'r') as file:
        f = json.load(file)

    # _num_anno = num_anno

    _anno = f['annotation']
    _filenames = f['filenames']
    _image_sizes = f['size']
    _class_corr = f['correspondence']
    _empty_idx = f['empty']
    _objects = f['objects']
    _verbs = f['verbs']


    for i in range(len(_anno)):   # 去除未见类标签
        new_anno = {"boxes_h": [], "boxes_o": [], "hoi": [], "object": [], "verb": []}
        for index in range(len(_anno[i]["hoi"])):
            if _anno[i]["hoi"][index] not in rf_uc:
                new_anno["boxes_h"].append(_anno[i]["boxes_h"][index])
                new_anno["boxes_o"].append(_anno[i]["boxes_o"][index])
                new_anno["object"].append(_anno[i]["object"][index])
                new_anno["verb"].append(_anno[i]["verb"][index])
                new_anno["hoi"].append(_anno[i]["hoi"][index])
        _anno[i] = new_anno
        if _anno[i]["hoi"] == []:
            _empty_idx.append(i)
    

    to_remove = []
    for corr in _class_corr:
        if corr[0] in rf_uc:
            to_remove.append(corr)

    for corr in to_remove:
        _class_corr.remove(corr)
    
    output_file = 'hicodet/instances_train2015_rf_uc.json'

    f['annotation'] = _anno
    f['empty'] = sorted(list(set(_empty_idx)))
    print(_class_corr)
    f['correspondence'] = _class_corr
    # 写入新的 JSON 文件
    with open(output_file, 'w') as file:
        json.dump(f, file, separators=(',', ':'))


def generate_nf_uc():
    anno_file = 'hicodet/instances_train2015.json'
    nf_uc = [38, 41, 20, 18, 245, 11, 19, 154, 459, 42, 155, 139, 60, 461, 577, 153, 582, 89, 141, 576, 75,
                212, 472, 61,
                457, 146, 208, 94, 471, 131, 248, 544, 515, 566, 370, 481, 226, 250, 470, 323, 169, 480, 479,
                230, 385, 73,
                159, 190, 377, 176, 249, 371, 284, 48, 583, 53, 162, 140, 185, 106, 294, 56, 320, 152, 374, 338,
                29, 594, 346,
                456, 589, 45, 23, 67, 478, 223, 493, 228, 240, 215, 91, 115, 337, 559, 7, 218, 518, 297, 191,
                266, 304, 6, 572,
                529, 312, 9, 308, 417, 197, 193, 163, 455, 25, 54, 575, 446, 387, 483, 534, 340, 508, 110, 329,
                246, 173, 506,
                383, 93, 516, 64]
    with open(anno_file, 'r') as file:
        f = json.load(file)

    # _num_anno = num_anno

    _anno = f['annotation']
    _filenames = f['filenames']
    _image_sizes = f['size']
    _class_corr = f['correspondence']
    _empty_idx = f['empty']
    _objects = f['objects']
    _verbs = f['verbs']


    for i in range(len(_anno)):   # 去除未见类标签
        new_anno = {"boxes_h": [], "boxes_o": [], "hoi": [], "object": [], "verb": []}
        for index in range(len(_anno[i]["hoi"])):
            if _anno[i]["hoi"][index] not in nf_uc:
                new_anno["boxes_h"].append(_anno[i]["boxes_h"][index])
                new_anno["boxes_o"].append(_anno[i]["boxes_o"][index])
                new_anno["object"].append(_anno[i]["object"][index])
                new_anno["verb"].append(_anno[i]["verb"][index])
                new_anno["hoi"].append(_anno[i]["hoi"][index])
        _anno[i] = new_anno
        if _anno[i]["hoi"] == []:
            _empty_idx.append(i)
    

    to_remove = []
    for corr in _class_corr:
        if corr[0] in nf_uc:
            to_remove.append(corr)

    for corr in to_remove:
        _class_corr.remove(corr)
    

    output_file = 'hicodet/instances_train2015_nf_uc.json'

    f['annotation'] = _anno
    f['empty'] = sorted(list(set(_empty_idx)))
    f['correspondence'] = _class_corr
    # 写入新的 JSON 文件
    with open(output_file, 'w') as file:
        json.dump(f, file, separators=(',', ':'))




if __name__ == "__main__":
    generate_uo()

'''
主要来着gen_vltk，但ua来自EoID
{
    uv{
        verb:[41, 100, 99, 91, 34, 42, 97, 84, 26, 106, 38, 56, 92, 79, 19, 76, 80, 2, 114, 62]
        hoi:[4, 6, 12, 15, 18, 25, 34, 38, 40, 49, 58, 60, 68, 69, 72, 73, 77, 82, 96, 97, 104, 113, 116, 118,
                    122, 129, 139, 147,
                    150, 153, 165, 166, 172, 175, 176, 181, 190, 202, 210, 212, 219, 227, 228, 233, 235, 243, 298, 313,
                    315, 320, 326, 336,
                    342, 345, 354, 372, 401, 404, 409, 431, 436, 459, 466, 470, 472, 479, 481, 488, 491, 494, 498, 504,
                    519, 523, 535, 536,
                    541, 544, 562, 565, 569, 572, 591, 595]
    }
    ua{
        verb:[8, 9,  25,  31,  37,  38,  40,  46,  47,  65,  67,  70,  72,  78,  80,  85,  86,  89, 93,  97,  98, 103]
        hoi:[
            2, 10, 14, 20, 27, 33, 36, 42, 46, 57, 68, 81, 82, 86, 90, 92, 101, 103,
            109, 111, 116, 120, 121, 122, 123, 136, 137, 138, 140, 141, 149, 152, 155,
            160, 161, 170, 172, 174, 180, 188, 205, 208, 215, 222, 225, 236, 247, 260,
            265, 271, 273, 279, 283, 288, 295, 300, 301, 306, 310, 311, 315, 318, 319,
            337, 344, 352, 356, 363, 369, 373, 374, 419, 425, 427, 438, 453, 458, 461,
            464, 468, 471, 475, 480, 486, 489, 490, 496, 504, 506, 513, 516, 524, 528,
            533, 542, 555, 565, 576, 590, 597
        ]
    }
    rf_uc{
    hoi:[509, 279, 280, 402, 504, 286, 499, 498, 289, 485, 303, 311, 325, 439, 351, 358, 66, 427, 379, 418,
                   70, 416,
                   389, 90, 395, 76, 397, 84, 135, 262, 401, 592, 560, 586, 548, 593, 526, 181, 257, 539, 535, 260, 596,
                   345, 189,
                   205, 206, 429, 179, 350, 405, 522, 449, 261, 255, 546, 547, 44, 22, 334, 599, 239, 315, 317, 229,
                   158, 195,
                   238, 364, 222, 281, 149, 399, 83, 127, 254, 398, 403, 555, 552, 520, 531, 440, 436, 482, 274, 8, 188,
                   216, 597,
                   77, 407, 556, 469, 474, 107, 390, 410, 27, 381, 463, 99, 184, 100, 292, 517, 80, 333, 62, 354, 104,
                   55, 50,
                   198, 168, 391, 192, 595, 136, 581]
    }
    nf_uc{
    hoi:[38, 41, 20, 18, 245, 11, 19, 154, 459, 42, 155, 139, 60, 461, 577, 153, 582, 89, 141, 576, 75,
                212, 472, 61,
                457, 146, 208, 94, 471, 131, 248, 544, 515, 566, 370, 481, 226, 250, 470, 323, 169, 480, 479,
                230, 385, 73,
                159, 190, 377, 176, 249, 371, 284, 48, 583, 53, 162, 140, 185, 106, 294, 56, 320, 152, 374, 338,
                29, 594, 346,
                456, 589, 45, 23, 67, 478, 223, 493, 228, 240, 215, 91, 115, 337, 559, 7, 218, 518, 297, 191,
                266, 304, 6, 572,
                529, 312, 9, 308, 417, 197, 193, 163, 455, 25, 54, 575, 446, 387, 483, 534, 340, 508, 110, 329,
                246, 173, 506,
                383, 93, 516, 64]
    }

    "unseen_object": [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                    126, 127, 128, 224, 225, 226, 227, 228, 229, 230, 231, 290, 291, 292, 293,
                    294, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 336, 337,
                    338, 339, 340, 341, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
                    429, 430, 431, 432, 433, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462,
                    463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 533, 534, 535, 536,
                    537, 558, 559, 560, 561, 595, 596, 597, 598, 599],  # 100
}
'''









# 这都是EoID


RARE_HOI_IDX = [
    8, 22, 27, 44, 50, 55, 62, 63, 66, 70, 76, 77, 80, 83, 84, 90, 99, 100,
    104, 107, 112, 127, 135, 136, 149, 158, 165, 166, 168, 172, 179, 181, 184,
    188, 189, 192, 195, 198, 205, 206, 214, 216, 222, 227, 229, 238, 239, 254,
    255, 257, 260, 261, 262, 274, 279, 280, 281, 286, 289, 292, 303, 311, 315,
    317, 325, 328, 333, 334, 345, 350, 351, 354, 358, 364, 379, 381, 389, 390,
    391, 395, 397, 398, 399, 401, 402, 403, 404, 405, 407, 410, 416, 418, 426,
    427, 429, 431, 436, 439, 440, 449, 451, 463, 469, 474, 482, 485, 498, 499,
    504, 509, 514, 517, 520, 522, 526, 531, 535, 539, 546, 547, 548, 549, 550,
    551, 552, 555, 556, 560, 578, 580, 581, 586, 592, 593, 595, 596, 597, 599
]

NON_RARE_HOI_IDX = [
    38, 41, 20, 18, 245, 11, 19, 154, 459, 42, 155, 139, 60, 461, 577, 153, 582, 89, 141, 576, 75,
    212, 472, 61,
    457, 146, 208, 94, 471, 131, 248, 544, 515, 566, 370, 481, 226, 250, 470, 323, 169, 480, 479,
    230, 385, 73,
    159, 190, 377, 176, 249, 371, 284, 48, 583, 53, 162, 140, 185, 106, 294, 56, 320, 152, 374, 338,
    29, 594, 346,
    456, 589, 45, 23, 67, 478, 223, 493, 228, 240, 215, 91, 115, 337, 559, 7, 218, 518, 297, 191,
    266, 304, 6, 572,
    529, 312, 9, 308, 417, 197, 193, 163, 455, 25, 54, 575, 446, 387, 483, 534, 340, 508, 110, 329,
    246, 173, 506,
    383, 93, 516, 64
]

UC_HOI_IDX = {
    'rare_first': RARE_HOI_IDX,
    'non_rare_first': NON_RARE_HOI_IDX,
    'uc0': [
        0, 1, 10, 29, 30, 41, 48, 50, 56, 57, 69, 72, 80, 81, 92, 93, 96, 109,
        110, 114, 127, 134, 139, 161, 170, 177, 183, 189, 191, 197, 198, 201,
        208, 209, 221, 227, 229, 232, 233, 235, 239, 242, 245, 247, 250, 252,
        260, 263, 270, 271, 280, 286, 288, 290, 299, 301, 308, 316, 325, 334,
        336, 343, 344, 352, 355, 356, 357, 363, 375, 376, 380, 384, 387, 389,
        395, 396, 397, 404, 408, 413, 414, 417, 422, 425, 433, 434, 436, 444,
        448, 452, 454, 455, 463, 480, 484, 488, 498, 503, 505, 507, 513, 516,
        527, 530, 532, 536, 537, 540, 546, 547, 550, 555, 561, 562, 566, 567,
        572, 581, 587, 598
    ],
    'uc1': [
        0, 3, 22, 29, 32, 52, 58, 63, 72, 73, 78, 89, 91, 92, 105, 106, 107,
        113, 137, 148, 163, 165, 172, 178, 179, 194, 196, 207, 209, 210, 214,
        215, 229, 231, 233, 234, 236, 240, 241, 243, 245, 247, 252, 254, 260,
        262, 269, 272, 282, 286, 289, 292, 296, 302, 310, 315, 322, 326, 333,
        335, 338, 340, 343, 347, 350, 351, 353, 354, 358, 362, 367, 368, 376,
        380, 388, 389, 393, 395, 397, 399, 410, 412, 416, 417, 419, 420, 429,
        434, 439, 441, 445, 449, 454, 467, 476, 483, 495, 503, 507, 511, 519,
        528, 529, 535, 537, 539, 547, 548, 556, 557, 561, 563, 565, 569, 579,
        587, 589, 591, 595, 597
    ],
    'uc2': [
        9, 25, 30, 49, 51, 61, 71, 74, 77, 82, 94, 108, 110, 116, 126, 131,
        143, 164, 168, 177, 185, 200, 201, 208, 212, 229, 232, 234, 239, 241,
        243, 244, 248, 255, 256, 258, 259, 266, 272, 279, 281, 287, 288, 290,
        294, 295, 301, 305, 308, 319, 322, 325, 328, 330, 332, 337, 344, 347,
        349, 350, 356, 359, 366, 367, 370, 375, 378, 380, 386, 387, 390, 391,
        400, 406, 409, 411, 416, 419, 428, 429, 431, 436, 439, 443, 445, 447,
        449, 451, 454, 457, 466, 468, 477, 479, 485, 486, 491, 497, 504, 508,
        510, 516, 527, 529, 531, 533, 536, 544, 545, 546, 549, 550, 552, 558,
        561, 568, 589, 594, 596, 599
    ],
    'uc3': [
        4, 14, 26, 27, 41, 45, 51, 53, 62, 69, 74, 80, 88, 91, 92, 93, 100,
        107, 110, 125, 127, 130, 136, 152, 153, 163, 167, 170, 177, 183, 186,
        188, 196, 200, 207, 210, 217, 220, 225, 232, 237, 242, 243, 246, 248,
        252, 253, 263, 267, 270, 280, 285, 289, 291, 292, 302, 312, 316, 325,
        335, 341, 343, 348, 355, 356, 362, 363, 368, 378, 382, 384, 385, 390,
        394, 396, 404, 406, 407, 415, 416, 426, 428, 429, 431, 435, 441, 443,
        448, 450, 452, 454, 460, 467, 469, 479, 480, 483, 498, 503, 505, 509,
        518, 524, 532, 533, 541, 549, 551, 560, 561, 566, 572, 573, 579, 580,
        585, 587, 594, 595, 599
    ],
    'uc4': [
        0, 4, 28, 29, 42, 43, 49, 53, 55, 56, 66, 72, 80, 81, 87, 90, 92, 94,
        100, 103, 109, 110, 129, 137, 149, 159, 166, 167, 170, 171, 179, 182,
        189, 193, 194, 195, 201, 206, 236, 237, 244, 245, 248, 249, 254, 255,
        257, 258, 266, 270, 290, 292, 300, 303, 316, 317, 326, 327, 331, 333,
        339, 340, 345, 347, 349, 350, 352, 353, 357, 362, 365, 366, 375, 380,
        381, 383, 385, 395, 396, 425, 426, 446, 448, 450, 451, 458, 466, 470,
        474, 476, 485, 487, 494, 495, 504, 505, 509, 515, 516, 525, 528, 529,
        536, 537, 539, 541, 546, 548, 556, 557, 568, 572, 578, 582, 585, 586,
        590, 593, 595, 597
    ]
}

UO_HOI_IDX = [
    111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
    126, 127, 128, 224, 225, 226, 227, 228, 229, 230, 231, 290, 291, 292, 293,
    294, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 336, 337,
    338, 339, 340, 341, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
    429, 430, 431, 432, 433, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462,
    463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 533, 534, 535, 536,
    537, 558, 559, 560, 561, 595, 596, 597, 598, 599
]

UA_HOI_IDX = [
    2, 10, 14, 20, 27, 33, 36, 42, 46, 57, 68, 81, 82, 86, 90, 92, 101, 103,
    109, 111, 116, 120, 121, 122, 123, 136, 137, 138, 140, 141, 149, 152, 155,
    160, 161, 170, 172, 174, 180, 188, 205, 208, 215, 222, 225, 236, 247, 260,
    265, 271, 273, 279, 283, 288, 295, 300, 301, 306, 310, 311, 315, 318, 319,
    337, 344, 352, 356, 363, 369, 373, 374, 419, 425, 427, 438, 453, 458, 461,
    464, 468, 471, 475, 480, 486, 489, 490, 496, 504, 506, 513, 516, 524, 528,
    533, 542, 555, 565, 576, 590, 597
]
# ua_verb:  [8   9  25  31  37  38  40  46  47  65  67  70  72  78  80  85  86  89 93  97  98 103] + 1
# ua_verb: ['carry','catch', 'exit', 'greet', 'hop_on', 'hose', 'hunt', 'lasso', 'launch', 'pet', 'pick_up', 
#           'pull', 'race', 'run', 'scratch', 'sip', 'sit_at', 'smell', 'stand_on', 'stop_at', 'straddle', 'text_on']