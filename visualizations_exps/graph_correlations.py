import csv
import matplotlib.pyplot as plt
from ia_processing_helpers import (all_same_length, Criterion, get_indices, get_score, get_sub_scores, ia_indices, measures,
                                   merge_baseline, metrics, NOSTOPS, plot_jaccard, plot_pearsons, tolist)

SAVE_METRICS_NAME = '.tmpmetrics'

if __name__ == '__main__':
    criterion = Criterion.AVG
    with open(SAVE_METRICS_NAME) as f:
        lines = list(csv.reader(f))
        header = lines[0]
        id_idx, label_idx, type_idx, text_idx, hb_idx, cp_idx, outpt_idx, conf_idx, ppl_idx = get_indices(header)
        stims = lines[1:]

        for stim in stims:
            stim_id = int(stim[1])
            text = stim[text_idx]
            hb = [abs(i) for i in tolist(stim[hb_idx])]
            cptm = tolist(stim[cp_idx])
            ppl = tolist(stim[ppl_idx])
            if NOSTOPS:
                hb = merge_baseline(stim_id, text, hb)
                cptm = merge_baseline(stim_id, text, cptm)
                ppl = merge_baseline(stim_id, text, ppl)

            # for each ia, get each statistic
            have_written = False
            p = False
            for measure in measures:
                raw_subtracted = get_sub_scores(stim_id, measure, 'raw', text)
                z_subtracted = get_sub_scores(stim_id, measure, 'z_score', text)
                if 'pred' in measures[measure]:
                    pred_subtracted = get_sub_scores(stim_id, measure, 'pred', text)
                if len(z_subtracted) != len(ppl) and not p:
                    p = True
                    print(text)
                    print(stim_id)
                    print('len ias:', len(ia_indices(stim_id, text)))
                    print('len raw: {}, len ppl: {}'.format(len(raw_subtracted), len(ppl)))
                    print()

                # can only compute correlations if same num entries in all fields
                if all_same_length([z_subtracted, hb]):
                    metrics[measure]['raw'] += [raw_subtracted]
                    metrics[measure]['z_score'] += [z_subtracted]
                    if 'pred' in measures[measure]:
                      metrics[measure]['pred'] += [pred_subtracted]
                    have_written = True
                elif not p:
                    p = True
                    print('mismatch! stim {}, len raw: {}, len hb: {}, len cpt: {}, len ppl: {}'.format(stim_id,
                                                                                                        len(z_subtracted),
                                                                                                        len(hb),
                                                                                                        len(cptm),
                                                                                                        len(ppl)))

            if have_written:
                metrics['hummingbird annotations'] += [hb]
                metrics['captum scores'] += [cptm]
                metrics['surprisal'] += [ppl]

        m = {}
        for name in metrics:
            print(type(metrics[name]))
            if 'raw' in name or 'pred' in name:
                continue
            if type(metrics[name]) is not dict:
                m[name] = metrics[name]
            else:
                for subtype in metrics[name]:
                    print(subtype)
                    if 'raw' in subtype or 'pred' in subtype:
                        continue
                    m['{} {}'.format(name, subtype)] = metrics[name][subtype]
        metrics = m

        # with open(SAVE_METRICS_NAME, 'w') as savefile:
        #     savefile.write(json.dumps(metrics))
        f, ax = plt.subplots(figsize=(10, 7))
        plot_jaccard(metrics, Criterion.EXP, axes=ax)
        plt.tight_layout()
        plt.show()
        f, ax = plt.subplots(figsize=(10, 7))
        plot_pearsons(metrics, Criterion.EXP, axes=ax)
        plt.tight_layout()
        #print(venn_diagram(metrics['hummingbird annotations'], metrics['captum scores'], metrics['First Run Dwell raw'], Criterion.TOP_50))
        plt.show()


if __name__ != '__main__':
    with open(SAVE_METRICS_NAME) as f:
        lines = list(csv.reader(f))
        header = lines[0]
        id_idx, label_idx, type_idx, text_idx, hb_idx, cp_idx, outpt_idx, conf_idx, ppl_idx = get_indices(header)
        stims = lines[1:]

        for stim in stims:
            stim_id = int(stim[1])
            text = stim[text_idx]
            hb = [abs(i) for i in tolist(stim[hb_idx])]
            cptm = tolist(stim[cp_idx])
            ppl = tolist(stim[ppl_idx])
            if NOSTOPS:
                hb = merge_baseline(stim_id, text, hb)
                cptm = merge_baseline(stim_id, text, cptm)
                ppl = merge_baseline(stim_id, text, ppl)

            # for each ia, get each statistic
            have_written = False
            p = False
            for measure in measures:
                raw_subtracted = get_sub_scores(stim_id, measure, 'raw', text)
                z_raw = get_score(stim_id, measure, 'z_score', text)
                z_subtracted = get_sub_scores(stim_id, measure, 'z_score', text)
                if 'pred' in measures[measure]:
                    pred_subtracted = get_sub_scores(stim_id, measure, 'pred', text)
                if len(z_subtracted) != len(ppl) and not p:
                    p = True
                    print(text)
                    print(stim_id)
                    print('len ias:', len(ia_indices(stim_id, text)))
                    print('len raw: {}, len ppl: {}'.format(len(raw_subtracted), len(ppl)))
                    print()

                # can only compute correlations if same num entries in all fields
                if all_same_length([z_subtracted, hb]):
                    metrics[measure]['raw'] += [raw_subtracted]
                    if not 'z_score conditional' in metrics[measure]:
                        metrics[measure]['z_score conditional'] = []
                    metrics[measure]['z_score conditional'] += [z_subtracted]
                    metrics[measure]['z_score'] += [z_raw]
                    if 'pred' in measures[measure]:
                      metrics[measure]['pred'] += [pred_subtracted]
                    have_written = True
                elif not p:
                    p = True
                    print('mismatch! stim {}, len raw: {}, len hb: {}, len cpt: {}, len ppl: {}'.format(stim_id,
                                                                                                        len(z_subtracted),
                                                                                                        len(hb),
                                                                                                        len(cptm),
                                                                                                        len(ppl)))

            if have_written:
                metrics['hummingbird annotations'] += [hb]
                metrics['captum scores'] += [cptm]
                metrics['surprisal'] += [ppl]

        m = {}
        for name in metrics:
            print(type(metrics[name]))
            if 'raw' in name or 'pred' in name:
                continue
            if name in ['hummingbird annotations', 'captum scores', 'surprisal']:
                continue
            if type(metrics[name]) is not dict:
                m[name] = metrics[name]
            else:
                for subtype in metrics[name]:
                    print(subtype)
                    if 'raw' in subtype or 'pred' in subtype:
                        continue
                    m['{} {}'.format(name, subtype)] = metrics[name][subtype]
        metrics = m

        # with open(SAVE_METRICS_NAME, 'w') as savefile:
        #     savefile.write(json.dumps(metrics))
        f, ax = plt.subplots(figsize=(10, 7))
        plot_jaccard(metrics, Criterion.EXP, axes=ax)
        plt.tight_layout()
        plt.show()
        f, ax = plt.subplots(figsize=(10, 7))
        plot_pearsons(metrics, Criterion.EXP, axes=ax)
        plt.tight_layout()
        #print(venn_diagram(metrics['hummingbird annotations'], metrics['captum scores'], metrics['First Run Dwell raw'], Criterion.TOP_50))
        plt.show()
