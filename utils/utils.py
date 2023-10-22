# Utils for eye tracking project


def flatten(l):
    """Returns a flattened list"""
    if len(l) == 0 or not type(l[0]) is list:
        return l
    return [item for sublist in l for item in sublist]


def all_same_length(list_of_lists):
    """Returns bool indicating whether all list are the same size"""
    assert(len(list_of_lists) > 0)
    last_length = len(list_of_lists[0])
    for sublist in list_of_lists:
        if len(sublist) != last_length:
            return False
    return True


def sim(first, second):
    """Calculates intersection over union of 1s in two lists"""
    same_ones = 0
    total_ones = 0
    for a, b in zip(first, second):
        if a == 1 or b == 1:
            total_ones += 1
        if a == 1 and b == 1:
            same_ones += 1
    return same_ones/total_ones if total_ones != 0 else 0


def move_indices(datas, idx_to_move):
    """:param datas: a list of lists
    :param idx_to_move: a list of (frm_idx, to_idx) tuples
    Mutates the lists in datas such that their frm_idx values are added to the to_idx values.
    frm_idx is then removed."""
    for data in datas:
        for frm, to in idx_to_move:
            data[to] += data[frm]
        indices_to_rm = [frm for frm, to in idx_to_move]
        for idx in sorted(indices_to_rm, reverse=True):
            del data[idx]


def tolist(item):
    """Given a space-separated string of float-like values, return List[Float]"""
    if item == 'NA' or item == '':
        return []
    result = [i if i != 'nan' else 0 for i in item.split(' ')]
    result = [float(i) for i in result if i != '']
    return result


def is_constant(measure):
    """Returns True if measure contains only one unique value"""
    if not measure:
        return True

    prev = measure[0]
    for m in measure:
        if m != prev:
            return False
    return True


def clean_word(word):
    """Returns lowercase word with any punctuation, etc removed"""
    return word.lower().strip(' ,"\'.`\n?!@#¬†;():*.\xa0_')


def get_indices(header):
    header = [h.strip() for h in header]
    id_idx = 0  # for some reason the "id" is encoding as "\ufeffid" and I can't spend time to figure it out
    label_idx = header.index('Label')
    type_idx = header.index('Type')
    text_idx = header.index('Text')
    hb_idx = header.index('Hummingbird scores')
    cp_idx = header.index('Captum Converted')
    outpt_idx = header.index('Judgment')
    conf_idx = header.index('Confidence')
    ppl_idx = header.index('gpt2 ppl')
    return [id_idx, label_idx, type_idx, text_idx, hb_idx, cp_idx, outpt_idx, conf_idx, ppl_idx]